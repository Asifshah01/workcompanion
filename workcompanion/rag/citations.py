"""Citations, grounding verification and confidence scoring.

Hallucination control has three parts:

1. **Only retrieved chunks can be cited.** Markers are generated from the
   chunks that actually supported the answer.
2. **Grounding verification.** A citation is only kept if its marker was
   genuinely available in the context that was sent to the model, so the UI can
   never show a page number the model invented.
3. **Confidence.** Derived from retrieval quality, reranker scores, agreement
   across sources and citation density - not from the model's own certainty.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.common import AgentResult, Citation, ConfidenceLevel, GroundingLabel
from workcompanion.schemas.documents import DocumentChunk
from workcompanion.schemas.retrieval import RetrievedChunk, RetrievalOutcome
from workcompanion.utils.text_utils import extract_key_terms, sentences_from_text

logger = get_logger(__name__)

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")
_SOURCE_CLAIM_RE = re.compile(
    r"(?:according to|source(?:d)? (?:from|in)|as (?:stated|described|shown) in)\s+"
    r"\[?(?P<label>[^\]\n.,;]{3,80})\]?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Citation building
# ---------------------------------------------------------------------------
def build_citations(
    chunks: Sequence[RetrievedChunk | DocumentChunk], *, limit: int = 8
) -> list[Citation]:
    """Create numbered citations for the chunks that will form the context.

    Accepts plain :class:`DocumentChunk` objects too (whole-document rendering,
    e.g. research-paper analysis) - those simply have no relevance score.
    """
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        marker = f"[{index}]"
        if isinstance(chunk, RetrievedChunk):
            citations.append(chunk.citation(marker=marker))
            continue
        metadata = chunk.metadata
        citations.append(
            Citation(
                marker=marker,
                document_id=metadata.document_id,
                document_name=metadata.document_name,
                page=metadata.page,
                section=metadata.section,
                chapter=metadata.chapter,
                chunk_id=chunk.chunk_id,
                quote=chunk.preview(280),
            )
        )
    return citations


def citations_markdown(citations: Sequence[Citation]) -> str:
    """Render the source list shown under an answer."""
    if not citations:
        return ""
    lines = ["", "---", "**Sources**", ""]
    for citation in citations:
        lines.append(f"- `{citation.marker}` **{citation.document_name}**" + _location(citation))
        if citation.url:
            lines.append(f"  <{citation.url}>")
    return "\n".join(lines)


def _location(citation: Citation) -> str:
    bits: list[str] = []
    if citation.page is not None:
        bits.append(f"p. {citation.page}")
    if citation.chapter:
        bits.append(citation.chapter)
    elif citation.section:
        bits.append(citation.section)
    return f" — {', '.join(bits)}" if bits else ""


# ---------------------------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------------------------
def markers_in_text(text: str) -> set[str]:
    """Citation markers used by the model in its answer."""
    return {f"[{int(value)}]" for value in _MARKER_RE.findall(text or "")}


def verify_grounding(
    answer: str, available: Sequence[Citation], *, strict: bool = True
) -> tuple[list[Citation], list[str]]:
    """Filter citations down to those the model was actually given.

    Args:
        answer: the model's answer text.
        available: citations built from the retrieved context.
        strict: when ``True`` a citation is kept only if its marker appears in
            the answer. When ``False`` all available citations are kept (useful
            when the model answered without explicit markers).

    Returns:
        ``(kept_citations, fabricated_markers)``.
    """
    allowed = {citation.marker: citation for citation in available}
    used = markers_in_text(answer)
    fabricated = sorted(used - set(allowed))

    if strict and used:
        kept = [allowed[marker] for marker in used if marker in allowed]
    else:
        kept = list(available)

    if fabricated:
        logger.warning(
            "Model referenced %d citation marker(s) that were not in context: %s",
            len(fabricated), ", ".join(fabricated),
        )
    return kept, fabricated


def annotate_grounding(
    answer: str, chunks: Sequence[RetrievedChunk], available: Sequence[Citation]
) -> list[GroundingLabel]:
    """Classify an answer's content into retrieved fact / reasoning / general knowledge.

    This is a heuristic, transparency-oriented labelling - not a proof of
    grounding.  It tells the learner which parts are quoted evidence and which
    parts are the model's interpretation.
    """
    labels: list[GroundingLabel] = []
    sentences = sentences_from_text(answer, min_length=25)

    quote_tokens: set[str] = set()
    for chunk in chunks:
        quote_tokens.update(extract_key_terms(chunk.chunk.text, limit=14, drop_stopwords=False))

    retrieved_hits = 0
    for sentence in sentences:
        tokens = set(extract_key_terms(sentence, limit=14))
        if not tokens:
            continue
        overlap = len(tokens & quote_tokens) / len(tokens)
        if overlap >= 0.55:
            retrieved_hits += 1

    if retrieved_hits:
        labels.append(GroundingLabel.RETRIEVED_FACT)
    if sentences and retrieved_hits < len(sentences) * 0.6:
        labels.append(GroundingLabel.MODEL_REASONING)
    if not chunks and answer:
        labels.append(GroundingLabel.GENERAL_KNOWLEDGE)

    # Explicit markers in the text are additional evidence of grounding.
    if markers_in_text(answer) & {c.marker for c in available}:
        if GroundingLabel.RETRIEVED_FACT not in labels:
            labels.append(GroundingLabel.RETRIEVED_FACT)
    return labels or [GroundingLabel.MODEL_REASONING]


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def compute_confidence(
    outcome: RetrievalOutcome,
    answer: str = "",
    *,
    answer_present: bool = True,
) -> float:
    """Blend retrieval, reranking and grounding evidence into a 0-1 score.

    Weights: top rerank score 0.30, mean of top-3 0.20, source diversity 0.15,
    term coverage 0.20, citation density 0.15.
    """
    if not outcome.chunks:
        return 0.0

    scores = [chunk.rerank_score or chunk.score for chunk in outcome.chunks]
    top = max(scores) if scores else 0.0
    top_three = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores)) if scores else 0.0

    documents = {chunk.chunk.metadata.document_id for chunk in outcome.chunks}
    diversity = min(1.0, len(documents) / 2.0)

    query_terms = set(outcome.query.keywords) or set(
        extract_key_terms(outcome.query.standalone or outcome.query.original, limit=10)
    )
    if query_terms:
        covered = 0
        for chunk in outcome.chunks[:4]:
            # No frequency cap here. A rare-but-exact hit ("gibbs") is the whole
            # point of the comparison, and a frequency-ranked top-N drops it in
            # favour of words like "the" and "energy" that appear everywhere.
            chunk_terms = set(extract_key_terms(chunk.chunk.text, limit=400))
            covered += len(query_terms & chunk_terms)
        coverage = min(1.0, covered / max(len(query_terms), 1))
    else:
        coverage = 0.5

    if answer_present and answer:
        used_markers = markers_in_text(answer)
        available_markers = {c.marker for c in outcome.citations}
        citation_density = min(1.0, len(used_markers & available_markers) / 2.0) if used_markers else 0.45
    else:
        citation_density = 0.0

    score = (
        0.30 * top
        + 0.20 * top_three
        + 0.15 * diversity
        + 0.20 * coverage
        + 0.15 * citation_density
    )

    # Lexical alignment is a gate, not just another 20% of the blend. A k-NN
    # retriever always returns k neighbours, so an entirely off-topic question
    # still collects high similarity scores purely because of where it sits in
    # the embedding space. Without this, "what is the airspeed velocity of a
    # swallow" scored *above* a genuine question about the indexed material.
    # One matching content term out of five is enough to be let through, which
    # keeps paraphrased-but-related questions working.
    if query_terms:
        score *= min(1.0, coverage / 0.2)

    return round(max(0.0, min(1.0, score)), 4)


def confidence_label(score: float) -> ConfidenceLevel:
    """Map a numeric confidence onto a qualitative level."""
    return ConfidenceLevel.from_score(score)


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_NOTICE = (
    "I couldn't find enough information in your uploaded material to answer this "
    "confidently."
)

INSUFFICIENT_EVIDENCE = (
    "**Not found in your material.**\n\n"
    "I searched every indexed document and could not find passages that answer this "
    "question. Rather than guess, here is what I can offer:\n\n"
    "- Upload the relevant lecture notes, textbook chapter or paper and ask again.\n"
    "- Rephrase the question using terms that appear in your material.\n"
    "- Turn on **Web research** in Settings for an answer from public sources "
    "(clearly marked as web evidence).\n"
    "- Ask me to run a **general knowledge** explanation instead - I will label it "
    "clearly as *not* coming from your documents."
)


def insufficient_evidence_result(agent: str, *, intent: str = "answer", answer: str | None = None) -> AgentResult:
    """The standard 'I refuse to invent' answer.

    ``success`` is False on purpose: the agent handled the request correctly, but
    it did not answer the question, and callers such as the research agent gate
    on this flag. Grounding is empty for the same reason - a refusal cites
    nothing because it retrieved nothing.
    """
    return AgentResult(
        success=False,
        agent=agent,
        intent=intent,
        answer=answer or INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        confidence_level=ConfidenceLevel.LOW,
        grounding=[],
        warnings=["The indexed material does not cover this question."],
        metadata={"evidence": "insufficient", "refused": True},
    )


def append_sources(answer: str, citations: Sequence[Citation], *, verified: bool = True) -> str:
    """Attach a rendered source list to an answer."""
    if not citations:
        return answer
    suffix = citations_markdown(citations)
    if verified:
        return f"{answer}\n{suffix}"
    return f"{answer}\n\n_These passages were retrieved for this question:_" + suffix


def extract_source_claims(answer: str) -> list[str]:
    """Find 'according to X' style claims - checked against real citations."""
    return [match.group("label").strip() for match in _SOURCE_CLAIM_RE.finditer(answer or "")]