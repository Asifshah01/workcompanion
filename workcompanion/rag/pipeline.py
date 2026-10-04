"""The end-to-end RAG pipeline.

    query
      -> query transformation (rewrite / de-contextualise / expand)
      -> multi-query hybrid retrieval (dense + BM25, weighted fusion)
      -> reranking
      -> context compression
      -> citation building
      -> confidence scoring

Every stage is optional/observable; the caller receives a
:class:`~workcompanion.schemas.retrieval.RetrievalOutcome` containing the
compressed context, verifiable citations and a confidence score.
"""

from __future__ import annotations

import time

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMProvider
from workcompanion.rag.citations import build_citations, compute_confidence
from workcompanion.rag.compressor import ContextCompressor
from workcompanion.rag.hybrid_search import HybridRetriever
from workcompanion.rag.query_transform import QueryTransformer
from workcompanion.rag.reranker import Reranker, get_reranker
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import VectorStore
from workcompanion.schemas.retrieval import RetrievalOutcome, RetrievalQuery, RetrievalStats
from workcompanion.utils.telemetry import get_telemetry

logger = get_logger(__name__)

__all__ = ["RAGPipeline", "get_rag_pipeline"]


class RAGPipeline:
    """Advanced retrieval-augmented generation over the personal knowledge base."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_index: BM25Index,
        embedding_provider: object,
        llm: LLMProvider | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._store = vector_store
        self._sparse = sparse_index
        self._embeddings = embedding_provider
        self._llm = llm
        self._transformer = QueryTransformer(llm, self._settings)
        self._retriever = HybridRetriever(
            vector_store, sparse_index, embedding_provider, self._settings  # type: ignore[arg-type]
        )
        self._compressor = ContextCompressor(self._settings, llm)
        self._reranker = reranker or get_reranker(self._settings, llm)

    # ------------------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        """Underlying vector store (used by ingestion and document-scoped agents)."""
        return self._store

    @property
    def sparse_index(self) -> BM25Index:
        return self._sparse

    @property
    def is_empty(self) -> bool:
        return self._store.count() == 0 and self._sparse.size == 0

    @property
    def chunk_count(self) -> int:
        return self._store.count()

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        skip_transform: bool = False,
        skip_rerank: bool = False,
        skip_compression: bool = False,
        query: RetrievalQuery | None = None,
    ) -> RetrievalOutcome:
        """Run the full retrieval pipeline for one question."""
        started = time.perf_counter()
        settings = self._settings
        telemetry = get_telemetry()
        stats = RetrievalStats()

        # 1. Query transformation -------------------------------------
        if query is None:
            if skip_transform:
                query = RetrievalQuery(original=question, standalone=question)
                query.keywords = _keywords(question)
            else:
                query = self._transformer.transform(
                    question, history=history, document_filter=document_filter
                )

        outcome = RetrievalOutcome(query=query, stats=stats)
        if self.is_empty:
            outcome.warnings.append("The knowledge base is empty - upload a document first.")
            stats.total_latency_ms = (time.perf_counter() - started) * 1000
            return outcome

        # 2. Multi-query hybrid retrieval -------------------------------
        queries = query.all_queries()
        stats.used_multi_query = len(queries) > 1
        retrieval_started = time.perf_counter()
        candidates, fusion_stats = self._retriever.retrieve(
            queries,
            top_k=top_k or settings.retrieval_top_k,
            candidate_k=settings.retrieval_candidate_k,
            document_ids=document_ids,
        )
        stats.retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)
        stats.dense_candidates = fusion_stats["dense_candidates"]
        stats.sparse_candidates = fusion_stats["sparse_candidates"]
        stats.fused_candidates = fusion_stats["fused_candidates"]
        stats.used_dense = fusion_stats["dense_candidates"] > 0
        stats.used_sparse = fusion_stats["sparse_candidates"] > 0

        if not candidates:
            outcome.warnings.append("No passages matched this question in the indexed material.")
            stats.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            outcome.confidence = 0.0
            return outcome

        # 3. Reranking --------------------------------------------------
        rerank_started = time.perf_counter()
        if skip_rerank:
            ranked = list(candidates)
            stats.used_reranker = "none"
        else:
            ranked = self._reranker.rerank(
                query.standalone or query.original,
                candidates,
                top_n=max(settings.rerank_top_n, top_k or settings.retrieval_top_k),
            )
            stats.used_reranker = self._reranker.name
        stats.rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 2)
        stats.reranked = len(ranked)

        # 4. Context compression ---------------------------------------
        if settings.context_compression and not skip_compression:
            compressed = self._compressor.compress(query.standalone or query.original, ranked)
        else:
            compressed = [item.model_copy(deep=True) for item in ranked]
        stats.compressed = len(compressed)
        selected = compressed[: (top_k or settings.retrieval_top_k)]

        if not selected:
            outcome.warnings.append("Retrieved passages contained no relevant sentences.")
            stats.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return outcome

        # 5. Context assembly + citations ------------------------------
        outcome.chunks = selected
        outcome.context = self._compressor.build_context(selected)
        outcome.citations = build_citations(selected)
        outcome.confidence = compute_confidence(outcome)
        outcome.confidence_level = _level(outcome.confidence)
        # Retrieving *something* is not the same as retrieving something usable.
        # Without this floor an off-topic question still matched a few loosely
        # related chunks and got a confident-looking answer built on them.
        floor = self._settings.min_retrieval_confidence
        outcome.grounded_context_available = outcome.confidence >= floor
        if not outcome.grounded_context_available:
            outcome.warnings.append(
                f"Best passage scored {outcome.confidence:.2f}, below the "
                f"{floor:.2f} grounding threshold."
            )
            logger.info(
                "RAG: refusing %d chunk(s) at conf=%.2f (floor=%.2f)",
                len(selected), outcome.confidence, floor,
            )
            return outcome

        stats.total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        telemetry.record("rag.total_ms", stats.total_latency_ms)
        telemetry.record("rag.retrieval_ms", stats.retrieval_latency_ms)
        telemetry.record("rag.rerank_ms", stats.rerank_latency_ms)
        telemetry.count("rag.queries")

        logger.info(
            "RAG: %d/%d candidates -> %d chunks (dense=%d sparse=%d rerank=%s) conf=%.2f in %.0fms",
            stats.fused_candidates, stats.reranked, len(selected),
            stats.dense_candidates, stats.sparse_candidates,
            stats.used_reranker, outcome.confidence, stats.total_latency_ms,
        )
        return outcome

    # ------------------------------------------------------------------
    def search_only(
        self,
        question: str,
        *,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> RetrievalOutcome:
        """Retrieve without compression - powers the dedicated search experience."""
        return self.retrieve(
            question,
            top_k=top_k,
            document_ids=document_ids,
            skip_compression=True,
        )

    def refresh_confidence(self, outcome: RetrievalOutcome, answer: str = "") -> RetrievalOutcome:
        """Recompute confidence once an answer exists (citation density matters)."""
        if answer:
            outcome.confidence = compute_confidence(outcome, answer)
            outcome.confidence_level = _level(outcome.confidence)
        return outcome

    def has_documents(self) -> bool:
        return not self.is_empty


def _keywords(text: str) -> list[str]:
    from workcompanion.utils.text_utils import extract_key_terms

    return extract_key_terms(text, limit=10)


def _level(score: float):  # type: ignore[no-untyped-def]
    from workcompanion.schemas.common import ConfidenceLevel

    return ConfidenceLevel.from_score(score)


# ---------------------------------------------------------------------------
_PIPELINES: dict[str, RAGPipeline] = {}


def get_rag_pipeline(
    vector_store: VectorStore,
    sparse_index: BM25Index,
    embedding_provider: object,
    llm: LLMProvider | None = None,
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> RAGPipeline:
    """Build (and cache) a pipeline for the given components."""
    settings = settings or get_settings()
    key = f"{id(vector_store)}|{id(sparse_index)}|{id(embedding_provider)}|{settings.dense_weight}"
    if force:
        _PIPELINES.pop(key, None)
    if key not in _PIPELINES:
        _PIPELINES[key] = RAGPipeline(
            vector_store, sparse_index, embedding_provider, llm, None, settings
        )
    return _PIPELINES[key]


def reset_pipeline_cache() -> None:
    """Drop cached pipelines (used after re-indexing or in tests)."""
    _PIPELINES.clear()