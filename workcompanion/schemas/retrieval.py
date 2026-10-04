"""Retrieval-layer schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from workcompanion.schemas.common import Citation, ConfidenceLevel
from workcompanion.schemas.documents import DocumentChunk


class RetrievalQuery(BaseModel):
    """A user question after query transformation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original: str
    queries: list[str] = Field(default_factory=list)
    standalone: str = ""
    keywords: list[str] = Field(default_factory=list)
    conversation_resolved: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)

    def all_queries(self) -> list[str]:
        seen: list[str] = []
        for candidate in [self.original, self.standalone, *self.queries]:
            value = (candidate or "").strip()
            if value and value not in seen:
                seen.append(value)
        return seen or [self.original]


class RetrievalStats(BaseModel):
    """Observability payload for the RAG pipeline."""

    dense_candidates: int = 0
    sparse_candidates: int = 0
    fused_candidates: int = 0
    reranked: int = 0
    compressed: int = 0
    used_dense: bool = False
    used_sparse: bool = False
    used_multi_query: bool = False
    used_reranker: str = "none"
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class RetrievedChunk(BaseModel):
    """A chunk with its retrieval scores and citation payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: DocumentChunk
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0
    compression_ratio: float = 1.0
    matched_queries: list[str] = Field(default_factory=list)

    @property
    def metadata(self) -> Any:
        return self.chunk.metadata

    def citation(self, marker: str) -> Citation:
        meta = self.chunk.metadata
        return Citation(
            marker=marker,
            document_id=meta.document_id,
            document_name=meta.document_name,
            page=meta.page,
            section=meta.section,
            chapter=meta.chapter,
            chunk_id=self.chunk.chunk_id,
            quote=self.chunk.preview(280),
            relevance=round(min(max(self.score, 0.0), 1.0), 4),
        )


class RetrievalOutcome(BaseModel):
    """Everything downstream agents need to answer grounded."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: RetrievalQuery
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    context: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    stats: RetrievalStats = Field(default_factory=RetrievalStats)
    warnings: list[str] = Field(default_factory=list)
    grounded_context_available: bool = False

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    def by_document(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.chunks:
            counts[item.chunk.metadata.document_name] = counts.get(item.chunk.metadata.document_name, 0) + 1
        return counts