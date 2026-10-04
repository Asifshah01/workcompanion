"""Advanced RAG layer: ingestion, semantic chunking, hybrid retrieval, reranking.

The pipeline is deliberately modular - each stage can be swapped or disabled
independently via settings, which is what makes the behaviour configurable and
testable.
"""

from workcompanion.rag.chunking import SemanticChunker, chunk_document
from workcompanion.rag.citations import (
    append_sources,
    build_citations,
    citations_markdown,
    compute_confidence,
    insufficient_evidence_result,
    verify_grounding,
)
from workcompanion.rag.compressor import ContextCompressor
from workcompanion.rag.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
)
from workcompanion.rag.hybrid_search import HybridRetriever
from workcompanion.rag.pipeline import RAGPipeline, get_rag_pipeline
from workcompanion.rag.query_transform import QueryTransformer
from workcompanion.rag.reranker import LocalReranker, Reranker, get_reranker
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorStore,
    get_vector_store,
)

__all__ = [
    "SemanticChunker",
    "chunk_document",
    "append_sources",
    "build_citations",
    "citations_markdown",
    "compute_confidence",
    "insufficient_evidence_result",
    "verify_grounding",
    "ContextCompressor",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "cosine_similarity",
    "get_embedding_provider",
    "HybridRetriever",
    "RAGPipeline",
    "get_rag_pipeline",
    "QueryTransformer",
    "LocalReranker",
    "Reranker",
    "get_reranker",
    "BM25Index",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "VectorStore",
    "get_vector_store",
]