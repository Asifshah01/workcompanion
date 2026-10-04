"""Hybrid retrieval: dense (embeddings) + sparse (BM25) with weighted fusion.

    Hybrid Score = w_dense * normalised_dense + w_sparse * normalised_bm25

Scores are min-max normalised per query before fusion so the two very different
score scales are comparable.  Multi-query results are merged with
reciprocal-rank fusion (RRF), then de-duplicated by chunk id.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.rag.embeddings import EmbeddingProvider
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import VectorStore
from workcompanion.schemas.retrieval import RetrievedChunk

logger = get_logger(__name__)

__all__ = ["HybridRetriever", "reciprocal_rank_fusion", "min_max_normalise"]


def min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Normalise scores into [0, 1]; a flat distribution maps to 1.0."""
    if not scores:
        return {}
    values = list(scores.values())
    lowest, highest = min(values), max(values)
    if highest - lowest < 1e-9:
        return {key: 1.0 for key in scores}
    span = highest - lowest
    return {key: (value - lowest) / span for key, value in scores.items()}


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int = 60
) -> dict[str, float]:
    """Classic RRF: merge several ranked id lists into one score map."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, identifier in enumerate(ranked):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (k + position + 1)
    return fused


class HybridRetriever:
    """Combines dense and sparse retrieval into one ranked candidate list."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        settings: Settings | None = None,
    ):
        self._store = vector_store
        self._sparse = sparse_index
        self._embeddings = embedding_provider
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    @property
    def dense_weight(self) -> float:
        return self._settings.dense_weight

    @property
    def sparse_weight(self) -> float:
        return self._settings.sparse_weight

    def _normalised_weights(self) -> tuple[float, float]:
        total = self.dense_weight + self.sparse_weight
        if total <= 0:
            return 0.7, 0.3
        return self.dense_weight / total, self.sparse_weight / total

    # ------------------------------------------------------------------
    def retrieve(
        self,
        queries: Sequence[str],
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        document_ids: Sequence[str] | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, int]]:
        """Retrieve candidates for one or more queries.

        Args:
            queries: one or more query variants (multi-query retrieval).
            top_k: final number of chunks to return.
            candidate_k: per-query candidate depth before fusion.
            document_ids: restrict retrieval to these documents.
            query_embedding: pre-computed embedding of ``queries[0]`` to avoid
                recomputation for multi-query retrieval.

        Returns:
            ``(chunks, stats)`` where ``chunks`` are ranked
            :class:`RetrievedChunk` objects carrying per-stage scores.
        """
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k
        candidate_k = candidate_k or max(settings.retrieval_candidate_k, top_k)
        dense_weight, sparse_weight = self._normalised_weights()
        doc_filter = set(document_ids) if document_ids else None
        query_list = [q for q in queries if q and q.strip()] or [""]
        stats = {"dense_candidates": 0, "sparse_candidates": 0, "fused_candidates": 0}

        dense_ranked: list[list[tuple[str, float]]] = []
        sparse_ranked: list[list[tuple[str, float]]] = []

        # --- dense ----------------------------------------------------
        embeddings: list[np.ndarray] = []
        if query_list[0]:
            if query_embedding is not None and len(query_list) == 1:
                embeddings.append(np.asarray(query_embedding, dtype=np.float32))
            else:
                embeddings = list(self._embeddings.embed_documents(query_list))

        for index, query in enumerate(query_list):
            vector = embeddings[index] if index < len(embeddings) else None
            if vector is None:
                continue
            hits = self._store.query(vector, candidate_k, filters=self._store_filters(doc_filter))
            if hits:
                dense_ranked.append(hits)
                stats["dense_candidates"] += len(hits)

        # --- sparse ---------------------------------------------------
        for query in query_list:
            hits = self._sparse.search(
                query, candidate_k, document_ids=doc_filter if doc_filter else None
            )
            if hits:
                sparse_ranked.append(hits)
                stats["sparse_candidates"] += len(hits)

        if not dense_ranked and not sparse_ranked:
            return [], stats

        # --- weighted fusion -------------------------------------------
        fused = self._fuse(dense_ranked, sparse_ranked, dense_weight, sparse_weight)
        stats["fused_candidates"] = len(fused)
        if not fused:
            return [], stats

        # --- hydrate with chunk objects -------------------------------
        ordered = sorted(fused.items(), key=lambda item: item[1][0], reverse=True)[:top_k]
        chunk_ids = [chunk_id for chunk_id, _ in ordered]
        chunks = {chunk.chunk_id: chunk for chunk in self._store.get_chunks(chunk_ids)}
        # Chunks may be missing if the vector store and BM25 index drifted apart.
        results: list[RetrievedChunk] = []
        for rank, (chunk_id, scores) in enumerate(ordered, start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=scores[0],
                    dense_score=scores[1],
                    sparse_score=scores[2],
                    rank=rank,
                    matched_queries=[q for q in query_list if q][:3],
                )
            )
        return results, stats

    # ------------------------------------------------------------------
    def _fuse(
        self,
        dense_ranked: list[list[tuple[str, float]]],
        sparse_ranked: list[list[tuple[str, float]]],
        dense_weight: float,
        sparse_weight: float,
    ) -> dict[str, tuple[float, float, float]]:
        """Weighted score fusion with per-stage normalisation."""
        fused: dict[str, tuple[float, float, float]] = {}

        for hits in dense_ranked:
            normalised = min_max_normalise(dict(hits))
            for chunk_id, _score in hits:
                previous = fused.get(chunk_id, (0.0, 0.0, 0.0))
                fused[chunk_id] = (
                    previous[0],
                    max(previous[1], normalised[chunk_id]),
                    previous[2],
                )

        for hits in sparse_ranked:
            normalised = min_max_normalise(dict(hits))
            for chunk_id, _score in hits:
                previous = fused.get(chunk_id, (0.0, 0.0, 0.0))
                fused[chunk_id] = (
                    previous[0],
                    previous[1],
                    max(previous[2], normalised[chunk_id]),
                )

        for chunk_id, (_dense, dense_score, sparse_score) in fused.items():
            combined = dense_weight * dense_score + sparse_weight * sparse_score
            # Items found by only one retriever still compete fairly.
            if dense_score <= 0 and sparse_score <= 0:
                combined = 0.0
            fused[chunk_id] = (combined, dense_score, sparse_score)
        return fused

    @staticmethod
    def _store_filters(document_ids: set[str] | None) -> dict[str, object] | None:
        """Build a vector-store filter restricting results to given documents."""
        if not document_ids:
            return None
        if len(document_ids) == 1:
            return {"document_id": next(iter(document_ids))}
        return {"document_id": {"$in": sorted(document_ids)}}

    # ------------------------------------------------------------------
    def fuse_multi_query(
        self, ranked_lists: Sequence[Sequence[RetrievedChunk]]
    ) -> list[RetrievedChunk]:
        """RRF-merge the ranked lists produced by several queries."""
        id_lists = [[item.chunk.chunk_id for item in ranked] for ranked in ranked_lists]
        scores = reciprocal_rank_fusion(id_lists)
        merged: dict[str, RetrievedChunk] = {}
        for ranked in ranked_lists:
            for item in ranked:
                merged.setdefault(item.chunk.chunk_id, item)
        output = []
        for rank, (chunk_id, score) in enumerate(
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1
        ):
            item = merged[chunk_id]
            item.score = max(item.score, score)
            item.rank = rank
            output.append(item)
        return output