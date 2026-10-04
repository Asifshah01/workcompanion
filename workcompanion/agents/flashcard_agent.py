"""Flashcard Agent.

Builds atomic question/answer decks from the learner's material.  Two rules
drive the implementation:

1. **Atomicity** - one card tests one idea.  Cards that need multiple hops are
   the main reason flashcard systems fail, so the prompt forbids them.
2. **Provenance** - every card keeps the chunk ids it came from, which lets the
   UI show "why is this my answer?" and lets the deck be re-generated when the
   underlying document is updated.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.rag.citations import citations_markdown
from workcompanion.rag.compressor import ContextCompressor
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.flashcards import CardDifficulty, Flashcard, FlashcardDeck
from workcompanion.schemas.retrieval import RetrievalOutcome
from workcompanion.utils.text_utils import extract_key_terms, sentences_from_text
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Flashcard Agent inside WorkCompanion AI. You build \
spaced-repetition decks for long-term retention.

ABSOLUTE RULES
1. ONE card = ONE idea. If a card needs two answers to be correct, split it.
2. Test RECOGNITION or RECALL, never reading comprehension or multi-step reasoning.
3. Every fact must come from the CONTEXT. Attach the citation markers of the \
passage each card came from in "citation_refs" (e.g. ["[2]"]).
4. The back of the card must be self-contained: a learner must be able to grade \
themselves from it alone, without the question.
5. Prefer concrete anchors: numbers, named mechanisms, definitions, contrasts, \
formulas. Avoid vague cards like "What is section 3 about?".
6. Keep fronts under 25 words and backs under 60 words.

COVERAGE
- Spread cards across the whole topic; do not generate near-duplicate cards.
- Include definition cards, mechanism/why cards and contrast cards."""

_DIFFICULTY_HINTS = {
    "easy": "Recall-level: definitions, names, direct facts.",
    "medium": "Understanding: mechanisms, purposes, relationships.",
    "hard": "Application: contrasts, exceptions, quantitative detail, multi-step links.",
    "expert": "Expert: nuances, boundary conditions, derivations, disputes.",
}


def _refs(raw: Any, mapping: Mapping[str, str]) -> list[str]:
    """Normalise citation refs to known chunk ids (same contract as the quiz)."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    output: list[str] = []
    for item in raw:
        marker = str(item).strip()
        if not marker:
            continue
        if not marker.startswith("["):
            marker = f"[{marker}]"
        chunk_id = mapping.get(marker)
        if chunk_id and chunk_id not in output:
            output.append(chunk_id)
    return output


def _dedupe_key(card: Flashcard) -> str:
    return re.sub(r"\W+", " ", card.front.lower()).strip()[:80]


class FlashcardAgent(AgentBase):
    """Grounded flashcard deck generation."""

    name = "flashcards"
    description = "Builds atomic, citation-backed flashcards for spaced repetition."

    def __init__(self, rag_agent: RagAgent, llm=None, settings=None, cache=None):
        super().__init__(llm, settings, cache)
        self._rag = rag_agent

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def generate_deck(
        self,
        topic: str,
        *,
        count: int = 15,
        difficulty: CardDifficulty = "medium",
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        title: str | None = None,
        per_topic_retrievals: Mapping[str, RetrievalOutcome] | None = None,
        skip_compression: bool = False,
    ) -> FlashcardDeck:
        """Create a deck for one topic (or several topics at once)."""
        wanted = max(1, min(int(count), 60))
        warnings: list[str] = []
        retrievals: dict[str, RetrievalOutcome] = dict(per_topic_retrievals or {})

        if not retrievals:
            retrievals = {
                topic: self._rag.retrieve(
                    topic,
                    document_ids=document_ids,
                    document_filter=document_filter,
                    top_k=max(8, wanted),
                )
            }

        if not any(outcome.chunks for outcome in retrievals.values()):
            return FlashcardDeck(
                id=uuid.uuid4().hex[:12],
                title=title or f"Deck: {topic}",
                subject=topic,
                cards=[],
                warnings=[
                    "No passages about this topic exist in your indexed material, so no "
                    "grounded deck could be built. Upload the relevant notes first."
                ],
            )

        blocks = self._evidence_blocks(retrievals, skip_compression=skip_compression)
        mapping = self._marker_map(retrievals)

        prompt = self._build_prompt(topic, blocks, wanted=wanted, difficulty=difficulty)
        request = self.build_request("flashcards", user=prompt, temperature=0.35, max_tokens=3000)

        raw_cards: list[dict[str, Any]] = []
        payload: dict[str, Any] = {}
        try:
            payload = self.structured(request) or {}
            raw_cards = [c for c in (payload.get("cards") or []) if isinstance(c, dict)]
            deck_title = str(payload.get("title") or "").strip()
        except Exception as exc:
            logger.warning("[flashcards] generation failed, using extractive cards: %s", exc)
            deck_title = ""

        cards: list[Flashcard] = []
        seen: set[str] = set()
        for spec in raw_cards:
            card = self._build_card(spec, mapping, difficulty, topic)
            if card is None:
                continue
            key = _dedupe_key(card)
            if key in seen:
                continue
            seen.add(key)
            cards.append(card)

        dropped = len(raw_cards) - len(cards)
        if dropped > 0:
            warnings.append(
                f"{dropped} generated card(s) were discarded (empty, duplicated or not "
                "grounded in your material)."
            )

        if len(cards) < max(1, wanted // 2):
            filler = self._extractive_cards(
                retrievals, mapping, needed=max(0, wanted - len(cards)), topic=topic
            )
            if filler:
                seen.update(_dedupe_key(card) for card in filler)
                cards.extend(filler)
                warnings.append(
                    "The deck was completed with cards built directly from your sentences "
                    "because the model did not return enough usable cards."
                )

        if len(cards) > wanted:
            cards = cards[:wanted]

        used = {ref for card in cards for ref in card.citation_refs}
        citations = [
            citation
            for outcome in retrievals.values()
            for citation in outcome.citations
            if citation.chunk_id in used
        ]

        return FlashcardDeck(
            id=uuid.uuid4().hex[:12],
            title=title or deck_title or f"Deck: {topic}",
            subject=topic,
            cards=cards,
            citations=citations,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def generate_from_topics(
        self,
        topics: Sequence[str],
        *,
        cards_per_topic: int = 6,
        difficulty: CardDifficulty = "medium",
        document_ids: list[str] | None = None,
    ) -> FlashcardDeck:
        """One focused retrieval per topic - used for weak-area revision decks."""
        clean = [t.strip() for t in topics if t.strip()][:8]
        if not clean:
            return FlashcardDeck(
                id=uuid.uuid4().hex[:12],
                title="Revision deck",
                cards=[],
                warnings=["No topics were supplied for this deck."],
            )

        retrievals: dict[str, RetrievalOutcome] = {}
        for topic in clean:
            outcome = self._rag.retrieve(
                topic, document_ids=document_ids, top_k=max(6, cards_per_topic)
            )
            if outcome.chunks:
                retrievals[topic] = outcome

        deck = self.generate_deck(
            ", ".join(retrievals) or clean[0],
            count=cards_per_topic * len(clean),
            difficulty=difficulty,
            title="Revision deck: " + ", ".join(list(retrievals)[:3]),
            per_topic_retrievals=retrievals,
        )
        missing = [topic for topic in clean if topic not in retrievals]
        if missing:
            deck.warnings.append(
                "No material was found for: " + ", ".join(missing[:4]) + "."
            )
        return deck

    # ------------------------------------------------------------------
    def run(
        self,
        topic: str,
        *,
        count: int = 15,
        difficulty: CardDifficulty = "medium",
        document_ids: list[str] | None = None,
    ) -> AgentResult:
        """Envelope-returning entry point for chat / crew workflows."""
        deck = self.generate_deck(
            topic, count=count, difficulty=difficulty, document_ids=document_ids
        )
        if not deck.cards:
            return AgentResult(
                success=False,
                agent=self.name,
                intent="flashcards",
                answer="\n\n".join(deck.warnings) or "I could not build a grounded deck.",
                confidence=0.0,
                metadata={"evidence": "insufficient", "deck_id": deck.id},
            )

        answer = self.render_deck(deck, reveal=True)
        return self.result(
            answer + citations_markdown(deck.citations),
            intent="flashcards",
            sources=deck.citations,
            confidence=self._deck_confidence(deck),
            grounding=[GroundingLabel.RETRIEVED_FACT, GroundingLabel.MODEL_REASONING],
            warnings=deck.warnings,
            metadata={
                "evidence": "documents",
                "deck_id": deck.id,
                "title": deck.title,
                "card_count": len(deck.cards),
                "topics": sorted({card.topic for card in deck.cards}),
                "offline": self.is_offline,
            },
        )

    # ------------------------------------------------------------------
    def render_deck(self, deck: FlashcardDeck, *, reveal: bool = False) -> str:
        """Markdown rendering used by the chat view and the deck page."""
        lines = [f"### {deck.title}", ""]
        lines.append(
            f"{len(deck.cards)} cards \u00b7 topics: "
            f"{', '.join(sorted({c.topic for c in deck.cards})) or 'n/a'}"
        )
        lines.append("")
        for index, card in enumerate(deck.cards, start=1):
            lines.append(f"**{index}.** {card.front}")
            if reveal:
                lines.append(f"   \u2192 {card.back}")
                if card.hint:
                    lines.append(f"   *Hint:* {card.hint}")
            else:
                lines.append("   _Tap to reveal_")
            lines.append("")
        for warning in deck.warnings:
            lines.append(f"> {warning}")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    def _build_prompt(
        self, topic: str, blocks: str, *, wanted: int, difficulty: CardDifficulty
    ) -> str:
        hint = _DIFFICULTY_HINTS.get(difficulty, _DIFFICULTY_HINTS["medium"])
        return (
            f"TOPIC:\n{topic}\n\n"
            f"REQUEST: exactly {wanted} cards.\n"
            f"CARD DIFFICULTY: {difficulty} - {hint}\n\n"
            f"EVIDENCE:\n{blocks}\n\n"
            "QUALITY BAR:\n"
            "- Front: one specific question or cloze prompt (<= 25 words).\n"
            "- Back: the complete answer the learner must recall (<= 60 words).\n"
            "- No card may be answerable without the evidence.\n"
            "- citation_refs lists the markers (e.g. \"[1]\") backing that card.\n\n"
            'Return STRICT JSON:\n'
            '{"title": "...", "cards": [{"front": "...", "back": "...", "topic": "...", '
            '"difficulty": "easy|medium|hard", "hint": "..." | null, "tags": ["..."], '
            '"citation_refs": ["[1]"]}]}'
        )

    def _build_card(
        self,
        spec: Mapping[str, Any],
        mapping: Mapping[str, str],
        default_difficulty: CardDifficulty,
        fallback_topic: str,
    ) -> Flashcard | None:
        front = str(spec.get("front") or "").strip()
        back = str(spec.get("back") or "").strip()
        if len(front) < 6 or len(back) < 6:
            return None

        refs = _refs(spec.get("citation_refs"), mapping)
        if not refs:
            return None

        difficulty = str(spec.get("difficulty") or default_difficulty).strip().lower()
        if difficulty not in ("easy", "medium", "hard", "expert"):
            difficulty = default_difficulty

        tags = [str(tag).strip() for tag in (spec.get("tags") or []) if str(tag).strip()]
        topic = str(spec.get("topic") or "").strip() or fallback_topic
        if topic not in tags:
            tags.append(topic)

        return Flashcard(
            id=f"c_{uuid.uuid4().hex[:10]}",
            front=front,
            back=back,
            topic=topic,
            difficulty=difficulty,  # type: ignore[arg-type]
            hint=str(spec["hint"]).strip() if spec.get("hint") else None,
            tags=tags[:6],
            citation_refs=refs,
        )

    def _extractive_cards(
        self,
        retrievals: Mapping[str, RetrievalOutcome],
        mapping: Mapping[str, str],
        *,
        needed: int,
        topic: str,
    ) -> list[Flashcard]:
        """Definition/recall cards built straight from the retrieved sentences."""
        if needed <= 0:
            return []

        cards: list[Flashcard] = []
        for label, outcome in retrievals.items():
            if len(cards) >= needed:
                break
            for index, chunk in enumerate(outcome.chunks, start=1):
                if len(cards) >= needed:
                    break
                marker = outcome.citations[index - 1].marker if index <= len(outcome.citations) else None
                chunk_id = mapping.get(marker or "")
                if not chunk_id:
                    continue
                sentence = next(
                    iter(sentences_from_text(chunk.chunk.text, min_length=55)), chunk.chunk.text.strip()
                )
                sentence = sentence[:400]
                if len(sentence) < 40:
                    continue
                terms = extract_key_terms(sentence, limit=3)
                subject = terms[0] if terms else label
                cards.append(
                    Flashcard(
                        id=f"c_{uuid.uuid4().hex[:10]}",
                        front=f"What does your material state about {subject}?",
                        back=sentence,
                        topic=label or topic,
                        citation_refs=[chunk_id],
                        tags=[subject],
                    )
                )
        return cards

    # ------------------------------------------------------------------
    def _marker_map(self, retrievals: Mapping[str, RetrievalOutcome]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for outcome in retrievals.values():
            for citation in outcome.citations:
                if citation.chunk_id:
                    mapping[citation.marker] = citation.chunk_id
        return mapping

    def _evidence_blocks(
        self,
        retrievals: Mapping[str, RetrievalOutcome],
        *,
        skip_compression: bool = False,
    ) -> str:
        """Render labelled evidence, re-using uncompressed chunk text when asked.

        Flashcard generation benefits from full sentences, so the dedicated deck
        view asks for the uncompressed passages instead of the query-compressed
        context used for question answering.
        """
        blocks: list[str] = []
        for label, outcome in retrievals.items():
            if not outcome.chunks:
                continue
            if skip_compression:
                lines: list[str] = []
                for index, chunk in enumerate(outcome.chunks, start=1):
                    header = ContextCompressor.citation_header(chunk, marker=index)
                    lines.append(f"[{index}] {header}\n{chunk.chunk.text.strip()}")
                body = "\n\n".join(lines)
            else:
                body = outcome.context
            blocks.append(f"### {label}\n{trim_to_tokens(body, 9000)}")
        return "\n\n".join(blocks) or "CONTEXT:\n(nothing relevant was retrieved)"

    @staticmethod
    def _deck_confidence(deck: FlashcardDeck) -> float:
        if not deck.cards:
            return 0.0
        grounded = sum(1 for card in deck.cards if card.citation_refs) / len(deck.cards)
        unique = len({_dedupe_key(card) for card in deck.cards}) / len(deck.cards)
        coverage = min(1.0, len({card.topic for card in deck.cards}) / 3.0)
        score = 0.5 * grounded + 0.2 * unique + 0.3 * coverage
        return round(min(0.92, score + 0.12), 4)
