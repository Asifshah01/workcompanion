"""Deterministic offline provider.

Serves two purposes:

1. **Demo mode** - the application is fully usable (grounded, extractive
   answers, quizzes, flashcards) without a Groq key, so a reviewer can run the
   end-to-end loop immediately.
2. **Tests** - deterministic, free, instant completions.

The provider dispatches on ``LLMRequest.metadata["task"]``; for unknown tasks it
falls back to an extractive answer built from the ``CONTEXT:`` block of the
prompt.  Everything it produces is clearly labelled as extractive / offline in
the agent layer, never presented as model reasoning.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Iterator, Sequence
from typing import Any

from workcompanion.llm.base import LLMProvider, LLMRequest, LLMResponse, Message
from workcompanion.utils.text_utils import (
    extract_key_terms,
    matches_keywords,
    sentences_from_text,
)

#: Block labels the agents use for retrieved evidence inside a prompt.
_CONTEXT_LABELS = (
    r"CONTEXT|EVIDENCE|SOURCES?|PASSAGES?|PAPER TEXT|DOCUMENT TEXT|DOCUMENTS?|"
    r"SOURCE MATERIAL|REFERENCE MATERIAL|MATERIAL|DOCUMENT EVIDENCE"
)
_CONTEXT_START_RE = re.compile(
    rf"(?:^|\n){{0,3}}[ \t]*(?:{_CONTEXT_LABELS})\b[^\n]*:[ \t]*\n", re.IGNORECASE
)
#: The next "SECTION LABEL:" line terminates the evidence block.
_CONTEXT_END_RE = re.compile(r"\n#{0,3}\s*[A-Z][A-Z /]{2,}\b[^\n]{0,80}:[ \t]*\n")
#: Provenance lines the agents prepend to each rendered chunk ("[1] Source: ...").
_CITATION_LINE_RE = re.compile(r"^\[\d+\]\s*Source:.*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_COUNT_RE = re.compile(
    r"\b(\d{1,3})\s*(?:\w+[- ]){0,2}?(?:questions?|mcqs?|items?|cards?|flashcards?|problems?)\b"
)
#: Placeholders the planner writes when a constraint is empty.
_EMPTY_CONSTRAINTS = frozenset({"", "not specified", "none", "n/a", "any", "unknown"})


def _constraint_list(prompt: str, label: str) -> list[str]:
    """Read a comma-separated constraint line, ignoring 'not specified' placeholders."""
    match = re.search(rf"\b{label}\s*:\s*(.+)", prompt, re.IGNORECASE)
    if not match:
        return []
    return [
        part.strip()
        for part in match.group(1).split(",")
        if part.strip() and part.strip().lower() not in _EMPTY_CONSTRAINTS
    ]


class OfflineProvider(LLMProvider):
    """Rule-based stand-in for a hosted LLM."""

    name = "offline"

    def __init__(self, model: str = "offline-extractive-v1"):
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _task(messages: Sequence[Message] | LLMRequest) -> str:
        if isinstance(messages, LLMRequest):
            return str(messages.metadata.get("task", "generic"))
        return "generic"

    @staticmethod
    def _prompt(messages: Sequence[Message] | LLMRequest) -> str:
        if isinstance(messages, LLMRequest):
            return messages.prompt_text()
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages)

    @staticmethod
    def _context(prompt: str) -> str:
        """Extract only the evidence block, never the system prompt or instructions.

        Agents label their evidence with varying headers (``CONTEXT:``,
        ``DOCUMENT EVIDENCE (...):``, ``PAPER TEXT:``), so the label is matched
        loosely and the block is terminated by the next ``SECTION LABEL:`` line.
        Falling back to the whole prompt is a last resort - it would let the
        system prompt leak into "extracted" answers.
        """
        match = _CONTEXT_START_RE.search(prompt)
        if match:
            body = prompt[match.end():]
            end = _CONTEXT_END_RE.search(body)
            text = body[: end.start()] if end else body
        else:
            text = prompt
        text = _CITATION_LINE_RE.sub("", text)
        for fence in _CODE_FENCE_RE.findall(text):
            if "|" in fence:  # markdown table row
                text = fence + "\n" + text
        return text.strip()

    #: Prompt labels agents use for the learner's own message, most specific first.
    _QUESTION_LABELS = (
        "RESEARCH QUESTION",
        "QUESTION",
        "LEARNER WANTS TO UNDERSTAND",
        "LEARNER MESSAGE",
        "USER",
        "QUERY",
    )

    @classmethod
    def _question(cls, prompt: str) -> str:
        """Recover the learner's question from a fully-rendered prompt.

        Agents label their prompts inconsistently (``QUESTION:``,
        ``LEARNER MESSAGE:``, ...), so every known label is tried before falling
        back to the last ``user:`` turn.  Without this fallback the router would
        keyword-match the *system* prompt and return nonsense.
        """
        for label in cls._QUESTION_LABELS:
            match = re.search(rf"{label}\s*:\s*(.+)", prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        last_user = ""
        for line in reversed(prompt.splitlines()):
            if line.lower().startswith("user:"):
                last_user = line.split(":", 1)[1].strip()
                break
        if last_user:
            return last_user
        stripped = prompt.strip()
        return stripped.splitlines()[0] if stripped else ""

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        started = time.perf_counter()
        task = self._task(messages)
        prompt = self._prompt(messages)
        handler = getattr(self, f"_task_{task.replace('-', '_')}", None)
        text = handler(prompt) if handler else None
        if text is None:
            text = self._extractive(prompt, question=self._question(prompt))

        if len(text) > (max_tokens or 4000) * 4:
            text = text[: (max_tokens or 4000) * 4]

        return LLMResponse(
            text=text.strip(),
            model=self._model,
            provider=self.name,
            finish_reason="stop",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            raw={"task": task},
        )

    def stream(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        response = self.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)
        words = response.text.split(" ")
        buffer = ""
        for word in words:
            buffer += word + " "
            if len(buffer) > 180:
                yield buffer
                buffer = ""
        if buffer.strip():
            yield buffer

    # ------------------------------------------------------------------
    # Generic extractive answer
    # ------------------------------------------------------------------
    def _extractive(self, prompt: str, *, question: str = "", max_sentences: int = 5) -> str:
        """Answer strictly from the supplied context (never invents facts)."""
        context = self._context(prompt)
        sentences = [s for s in sentences_from_text(context, min_length=40) if s]
        if not sentences:
            return (
                "I could not find this information in the material you have provided.\n\n"
                "_Offline extractive mode: no source passages were retrieved, so no answer can be "
                "grounded in your documents. Add a Groq API key for full generative teaching._"
            )

        terms = set(extract_key_terms(question or prompt, limit=10))
        scored = sorted(
            enumerate(sentences),
            key=lambda item: (-(sum(1 for t in terms if t in item[1].lower()) / (len(terms) or 1)), item[0]),
        )
        selected = sorted(index for index, _ in scored[:max_sentences])
        picked = [sentences[i] for i in selected]

        lines = ["**Based on your material:**", ""]
        lines += [f"- {sentence}" for sentence in picked]
        lines += [
            "",
            "_Offline extractive mode - these sentences are copied from your documents. "
            "Add a Groq API key for a full teaching explanation, analogies and examples._",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------
    def _task_route(self, prompt: str) -> str:
        """Offline intent classifier.

        Mirrors ``workcompanion.agents.nexus_agent._RULES`` (same order, same
        whole-word semantics) so the router produces the same decision offline
        as it would from a real model on an unambiguous message.
        """
        question = self._question(prompt).lower()
        table = [
            (("study plan", "revision schedule", "schedule my", "plan my", "timetable"),
             "plan", "needs_documents"),
            (("exam paper", "mock exam", "exam mode", "full exam", "past paper"),
             "exam", "needs_documents"),
            (("quiz", "test me", "mcq", "practice questions", "viva", "question paper"),
             "quiz", "needs_documents"),
            (("flashcard", "flash card", "revision card", "spaced repetition", "anki"),
             "flashcards", "needs_documents"),
            (("analyse this paper", "analyze this paper", "paper analysis", "methodology of this",
              "research gap", "critical analysis", "summarise this paper", "summarize this paper"),
             "paper_analysis", "needs_documents"),
            (("compare", "versus", " vs ", "vs.", "difference between", "contradict", "conflict"),
             "compare", "needs_documents"),
            (("socratic", "guide me", "hint me", "don't tell me", "dont tell me"),
             "socratic", "needs_documents"),
            (("summarise", "summarize", "tldr", "tl;dr", "in short", "condense"),
             "summarize", "needs_documents"),
            (("how am i", "my progress", "my score", "weak area", "streak", "improve me"),
             "progress", "no_documents"),
            (("latest", "news", "on the internet", "online", "recent paper", "who is working on",
              "research", "find papers", "literature", "state of the art", "survey the"),
             "research", "use_web"),
            (("search for", "search my", "find where", "look up", "locate", "search"),
             "search", "needs_documents"),
            (("explain", "teach me", "how does", "what is", "why does", "walk me through",
              "i don't understand", "help me understand", "meaning of"),
             "explain", "needs_documents"),
            (("hi", "hello", "hey", "thanks", "thank you", "who are you", "what can you do"),
             "small_talk", "no_documents"),
        ]
        intent, need = "unclear", "no_documents"
        for keywords, matched_intent, flag in table:
            if matches_keywords(question, keywords):
                intent, need = matched_intent, flag
                break

        count_match = _COUNT_RE.search(question)
        count = count_match.group(1) if count_match else None
        if count is None:
            count = "15" if intent == "flashcards" else ("6" if intent in ("quiz", "exam") else "null")

        return (
            '{"intent": "%s", "confidence": 0.62, "needs_documents": %s, "use_web": %s, '
            '"rationale": "Rule-based offline routing.", "question_count": %s, "topic": null}'
            % (
                intent,
                str(need == "needs_documents").lower(),
                str(need == "use_web").lower(),
                count,
            )
        )

    def _task_query_transform(self, prompt: str) -> str:
        question = self._question(prompt)
        terms = extract_key_terms(question, limit=6)
        expansions = [f"{term} {question}" for term in terms[:3]] or [question]
        import json

        return json.dumps(
            {
                "standalone": question,
                "queries": expansions,
                "keywords": terms,
                "resolved": False,
            }
        )

    def _task_rerank(self, prompt: str) -> str:
        return '{"scores": [], "rationale": "offline: lexical order preserved"}'

    def _task_quiz(self, prompt: str) -> str:
        import json

        context = self._context(prompt)
        sentences = [s for s in sentences_from_text(context, min_length=45)][:6]
        questions: list[dict[str, Any]] = []
        for index, sentence in enumerate(sentences, start=1):
            snippet = sentence if len(sentence) <= 220 else sentence[:217] + "..."
            options_source = re.split(r"[;.]", sentence)
            distractors = [opt.strip() for opt in options_source[1:4] if opt.strip()] or [
                "None of the above",
                "It depends on the application",
                "Not stated in the material",
            ]
            questions.append(
                {
                    "question": f"Which statement about the material is correct? (#{index})",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "topic": extract_key_terms(sentence, limit=2) or ["General"],
                    "options": [snippet, *distractors[:3]],
                    "correct_index": 0,
                    "explanation": f"Source sentence: {sentence[:220]}",
                    "hint": "Re-read the highlighted passage.",
                }
            )
        return json.dumps({"title": "Offline extractive quiz", "questions": questions})

    def _task_flashcards(self, prompt: str) -> str:
        import json

        context = self._context(prompt)
        sentences = [s for s in sentences_from_text(context, min_length=40)][:10]
        cards = [
            {
                "front": f"Explain: {sentence[:120]}",
                "back": sentence[:400],
                "topic": (extract_key_terms(sentence, limit=1) or ["General"])[0],
                "difficulty": "medium",
            }
            for sentence in sentences
        ]
        return json.dumps({"title": "Offline deck", "cards": cards})

    def _task_evaluate_answer(self, prompt: str) -> str:
        import json

        expected_match = re.search(r"EXPECTED ANSWER\s*:\s*(.+)", prompt)
        expected = expected_match.group(1).strip() if expected_match else ""
        given_match = re.search(r"LEARNER ANSWER\s*:\s*(.+)((?:\n(?!EXPECTED).*)*)", prompt)
        given = (given_match.group(1).strip() if given_match else "").lower()
        overlap = set(extract_key_terms(given)) & set(extract_key_terms(expected))
        correct = bool(given) and (len(overlap) >= max(1, len(extract_key_terms(expected)) // 2))
        return json.dumps(
            {
                "is_correct": correct,
                "score": 1.0 if correct else 0.0,
                "feedback": (
                    "Your answer matches the key points in the material."
                    if correct
                    else "Your answer misses the key idea in the source material."
                ),
                "misconception": None if correct else "Key definition not recalled accurately",
                "severity": "none" if correct else "minor",
                "corrected_explanation": expected[:400],
                "follow_up_question": None if correct else "Can you restate the definition in your own words?",
            }
        )

    def _task_socratic(self, prompt: str) -> str:
        question = self._question(prompt)
        context = self._context(prompt)
        first = sentences_from_text(context, min_length=40)[:1]
        lead = first[0][:180] if first else ""
        return (
            f"Before I answer **{question[:90]}**, let's reason about it together.\n\n"
            "1. What do you already know about the underlying principle here?\n"
            f"2. Which quantity in the passage you read seems to *change*, and in which direction?\n"
            f"3. What relationship would you *predict* between them?\n\n"
            f"_Relevant passage for reference: {lead}_\n\n"
            "Answer question 1 in your own words and I'll take it from there."
        )

    def _task_paper_analysis(self, prompt: str) -> str:
        import json

        context = self._context(prompt)
        # Prefer a markdown heading or the first substantive line as the title.
        headings = [
            line.lstrip("#").strip()
            for line in context.splitlines()
            if line.lstrip().startswith("#") and line.lstrip("#").strip()
        ]
        sentences = sentences_from_text(context, min_length=50)[:6]
        title = headings[0] if headings else (sentences[0][:120] if sentences else "Untitled")
        return json.dumps(
            {
                "title": title,
                "authors": [],
                "abstract": " ".join(sentences[:2])[:800],
                "research_problem": sentences[0] if sentences else "",
                "objectives": sentences[1:3],
                "methodology": sentences[2] if len(sentences) > 2 else "",
                "dataset": "Not explicitly stated in the extracted text.",
                "experimental_setup": "Not explicitly stated in the extracted text.",
                "key_results": sentences[3:5],
                "limitations": [],
                "future_work": [],
                "novelty": "Requires a configured LLM provider for a reliable novelty statement.",
                "research_gaps": [],
                "confidence": 0.25,
            }
        )

    def _task_study_plan(self, prompt: str) -> str:
        import json

        subject_match = re.search(r"SUBJECT\s*:\s*(.+)", prompt)
        subject = subject_match.group(1).strip() if subject_match else "the subject"
        topics = _constraint_list(prompt, "TOPICS")
        weak = _constraint_list(prompt, "WEAK AREAS")
        topics = topics or weak or ["Core concepts", "Practice problems", "Revision"]
        # Rule 2 of the planner contract: weak areas get the earliest slots.
        weak_key = {t.strip().lower() for t in weak}
        topics = [*weak, *[t for t in topics if t.strip().lower() not in weak_key]]
        days = []
        for index in range(7):
            topic = topics[index % len(topics)]
            activity = ["learn", "practice", "flashcards", "revise", "quiz", "mock_exam", "rest"][index]
            days.append(
                {
                    "day_number": index + 1,
                    "focus": topic,
                    "activities": [activity],
                    "topics": [topic],
                    "hours": 2.0 if activity != "rest" else 1.0,
                    "priority": "high" if topic in weak else "medium",
                    "rationale": "Weak-area priority" if topic in weak else "Steady coverage.",
                }
            )
        return json.dumps({"subject": subject, "days": days, "strategy_notes": [
            "Offline heuristic plan: weak areas are scheduled first and interleaved with retrieval practice.",
        ]})

    def _task_research_synthesis(self, prompt: str) -> str:
        return self._extractive(prompt, question=self._question(prompt), max_sentences=6)

    def _task_misconception_check(self, prompt: str) -> str:
        return self._task_evaluate_answer(prompt)

    def _task_compress(self, prompt: str) -> str:
        return self._extractive(prompt, question=self._question(prompt), max_sentences=4)

    def _task_paper_summary(self, prompt: str) -> str:
        return self._extractive(prompt, question=self._question(prompt), max_sentences=6)

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self._model,
            "available": True,
            "mode": "extractive / rule-based (no network, no API key)",
            "fingerprint": hashlib.sha256(self._model.encode()).hexdigest()[:8],
        }