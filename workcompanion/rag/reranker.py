"""Reranking layer.

Candidates from hybrid retrieval are fused, then rescored against the original
question with a stronger (but slower) model.  Three implementations are
available and selected by ``settings.reranker``:

* ``CrossEncoderReranker`` - a real cross-encoder (default: ``ms-marco-MiniLM-L-6-v2``).
* ``LocalReranker`` - dependency-free lexical/structural scorer combining term
  coverage, phrase proximity, heading match, IDF weights and position priors.
* ``LLMReranker`` - asks the LLM for relevance judgements (most accurate,
  highest token cost; opt-in).

``reranker="auto"`` prefers the cross-encoder, then the local scorer, so the
pipeline always works even with no optional dependencies installed.
"""

from __future__ import annotations

import abc
import re
from collections.abc import Sequence

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMProvider, LLMRequest, SystemMessage, UserMessage
from workcompanion.rag.sparse_index import tokenize
from workcompanion.schemas.retrieval import RetrievedChunk

logger = get_logger(__name__)

LLM_RERANK_PROMPT = """You rank passages by how well they answer a question.

Return STRICT JSON: {"scores": [{"id": "<passage id>", "score": 0.0-1.0, "why": "<=15 words"}]}
Include every passage you are shown. 0.0 means irrelevant, 1.0 means it fully answers
the question. Judge only relevance - do not attempt to answer the question yourself."""


class Reranker(abc.ABC):
    """Interface every reranker implements."""

    name: str = "base"

    @abc.abstractmethod
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int = 10
    ) -> list[RetrievedChunk]:
        """Return the ``chunks`` re-scored and sorted, best first."""

    def _assign(
        self, chunks: Sequence[RetrievedChunk], scores: Sequence[float]
    ) -> list[RetrievedChunk]:
        """Attach scores, re-rank and trim to ``top_n``."""
        ordered: list[RetrievedChunk] = []
        for position, chunk in enumerate(chunks):
            value = float(scores[position]) if position < len(scores) else 0.0
            chunk.rerank_score = max(0.0, min(1.0, value))
            chunk.score = chunk.rerank_score
            ordered.append(chunk)
        ordered.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, chunk in enumerate(ordered, start=1):
            chunk.rank = rank
        return ordered


# ---------------------------------------------------------------------------
class LocalReranker(Reranker):
    """Fast, dependency-free relevance scorer.

    Score = 0.34 * term_coverage
          + 0.18 * idf_term_coverage
          + 0.16 * phrase_proximity
          + 0.12 * heading_match
          + 0.10 * retrieval_score
          + 0.10 * position_prior
    """

    name = "local"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._weights = {
            "coverage": 0.34,
            "idf": 0.18,
            "proximity": 0.16,
            "heading": 0.12,
            "retrieval": 0.10,
            "position": 0.10,
        }

    # ------------------------------------------------------------------
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int = 10
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        query_tokens = [token for token in tokenize(query) if len(token) > 1]
        query_token_set = set(query_tokens)
        query_phrase = " ".join(token for token in tokenize(query) if token.isalpha())[:60]

        scores: list[float] = []
        for position, chunk in enumerate(chunks):
            text = chunk.chunk.text
            lowered = text.lower()
            tokens = tokenize(text)
            token_set = set(tokens)

            coverage = (
                len(query_token_set & token_set) / len(query_token_set) if query_token_set else 0.0
            )
            # Rare terms matter more than common ones.
            rare = {token for token in query_token_set if len(token) > 4}
            idf_coverage = (
                len(rare & token_set) / len(rare) if rare else coverage
            )

            proximity = self._phrase_proximity(tokens, query_tokens)
            heading = self._heading_match(chunk, query_token_set)
            retrieval = min(max(chunk.score, 0.0), 1.0)
            prior = 1.0 - (position / max(len(chunks), 1))

            score = (
                self._weights["coverage"] * coverage
                + self._weights["idf"] * idf_coverage
                + self._weights["proximity"] * proximity
                + self._weights["heading"] * heading
                + self._weights["retrieval"] * retrieval
                + self._weights["position"] * prior
            )
            if query_phrase and query_phrase in " ".join(lowered.split()):
                score = min(1.0, score + 0.12)
            scores.append(score)

        return self._assign(chunks, scores)[:top_n]

    @staticmethod
    def _phrase_proximity(tokens: Sequence[str], query_tokens: Sequence[str]) -> float:
        """Reward query terms that appear close together."""
        if len(query_tokens) < 2:
            return 1.0 if query_tokens and query_tokens[0] in tokens else 0.0
        positions = [index for index, token in enumerate(tokens) if token in set(query_tokens)]
        if len(positions) < 2:
            return 0.0
        spread = positions[-1] - positions[0] + 1
        return min(1.0, len(query_tokens) / spread)

    @staticmethod
    def _heading_match(chunk: RetrievedChunk, query_tokens: set[str]) -> float:
        heading = (chunk.chunk.heading_path or chunk.chunk.metadata.section or "").lower()
        if not heading:
            return 0.0
        heading_tokens = set(tokenize(heading))
        if not heading_tokens:
            return 0.0
        return len(heading_tokens & query_tokens) / len(heading_tokens)


# ---------------------------------------------------------------------------
class CrossEncoderReranker(Reranker):
    """Cross-encoder reranking via sentence-transformers."""

    name = "cross-encoder"

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._failed = False

    def _ensure_loaded(self) -> object | None:
        if self._model is not None or self._failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            self._failed = True
            logger.warning("Cross-encoder '%s' unavailable (%s).", self._model_name, exc)
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int = 10
    ) -> list[RetrievedChunk]:
        model = self._ensure_loaded()
        if model is None:
            return LocalReranker(self._settings_for_fallback()).rerank(query, chunks, top_n=top_n)
        pairs = [(query, chunk.chunk.text) for chunk in chunks]
        try:
            raw = model.predict(pairs)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - runtime model failure
            logger.warning("Cross-encoder scoring failed (%s); using the local reranker.", exc)
            return LocalReranker(self._settings_for_fallback()).rerank(query, chunks, top_n=top_n)
        # `ms-marco` models emit unbounded logits; squash into [0, 1].
        scores = [_sigmoid(float(value)) for value in raw]
        return self._assign(chunks, scores)[:top_n]

    @staticmethod
    def _settings_for_fallback() -> Settings:
        return get_settings()


def _sigmoid(value: float) -> float:
    if value < -30:
        return 0.0
    if value > 30:
        return 1.0
    return 1.0 / (1.0 + pow(2.718281828, -value))


# ---------------------------------------------------------------------------
class LLMReranker(Reranker):
    """LLM-based relevance judging (highest quality, highest token cost)."""

    name = "llm"

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int = 10
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        passages = "\n\n".join(
            f"id: {chunk.chunk.chunk_id}\n{chunk.chunk.preview(700)}" for chunk in chunks[:14]
        )
        request = LLMRequest(
            messages=[
                SystemMessage(content=LLM_RERANK_PROMPT),
                UserMessage(content=f"QUESTION:\n{query}\n\nPASSAGES:\n{passages}"),
            ],
            temperature=0.0,
            max_tokens=900,
            metadata={"task": "rerank"},
        )
        try:
            payload = self._llm.generate_json(request, retries=1)
        except Exception as exc:
            logger.warning("LLM reranking failed (%s); using the local reranker.", exc)
            return LocalReranker().rerank(query, chunks, top_n=top_n)

        score_map: dict[str, float] = {}
        if isinstance(payload, dict):
            for entry in payload.get("scores") or []:
                if isinstance(entry, dict) and entry.get("id") is not None:
                    try:
                        score_map[str(entry["id"])] = float(entry.get("score", 0.0))
                    except (TypeError, ValueError):
                        continue

        fallback = LocalReranker()
        if not score_map:
            return fallback.rerank(query, chunks, top_n=top_n)

        ranked = fallback.rerank(query, chunks, top_n=max(top_n, len(chunks)))
        blended: list[RetrievedChunk] = []
        for chunk in ranked:
            llm_score = score_map.get(chunk.chunk.chunk_id, 0.0)
            chunk.rerank_score = max(0.0, min(1.0, 0.65 * llm_score + 0.35 * chunk.rerank_score))
            blended.append(chunk)
        blended.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, chunk in enumerate(blended, start=1):
            chunk.rank = rank
            chunk.score = chunk.rerank_score
        return blended[:top_n]


# ---------------------------------------------------------------------------
def get_reranker(
    settings: Settings | None = None, llm: LLMProvider | None = None
) -> Reranker:
    """Instantiate the configured reranker, degrading gracefully."""
    settings = settings or get_settings()
    choice = settings.reranker

    if choice == "local":
        return LocalReranker(settings)
    if choice == "llm":
        return LLMReranker(llm) if llm else LocalReranker(settings)
    if choice == "cross-encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)

    # "auto"
    try:
        import sentence_transformers  # noqa: F401

        return CrossEncoderReranker(settings.cross_encoder_model)
    except Exception as exc:
        logger.info("Cross-encoder reranking unavailable (%s); using the local reranker.", exc)
        return LocalReranker(settings)


def local_relevance(query: str, text: str) -> float:
    """Standalone lexical relevance in [0, 1] (used for confidence scoring)."""
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


_PHRASE = re.compile(r"\s+")


def lexical_overlap(query: str, text: str) -> float:
    """Whitespace-level phrase overlap, robust to tokenisation differences."""
    query_words = {word for word in re.findall(r"[a-z0-9']+", query.lower()) if len(word) > 2}
    text_words = set(re.findall(r"[a-z0-9']+", text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)