"""Vector storage abstraction.

Retrieval code depends on :class:`VectorStore` only, so swapping Chroma for
Qdrant / Weaviate / Pinecone later means implementing this one interface.
Embeddings are always passed explicitly, so no backend-side embedding model is
ever required (which also means no silent network downloads).
"""

from __future__ import annotations

import abc
import threading
from collections.abc import Sequence

import numpy as np

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.rag.embeddings import l2_normalize
from workcompanion.schemas.documents import ChunkKind, ChunkMetadata, DocumentChunk

logger = get_logger(__name__)

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    "reset_vector_store_cache",
]

#: Chunk metadata fields persisted as Chroma scalars.
_SCALAR_FIELDS = (
    "document_id",
    "document_name",
    "document_type",
    "page",
    "section",
    "chapter",
    "chunk_index",
    "source",
    "kind",
    "heading_path",
)


class VectorStore(abc.ABC):
    """Interface every vector backend implements."""

    name: str = "base"

    # -- writes -------------------------------------------------------
    @abc.abstractmethod
    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> int:
        """Insert or replace chunks with their embeddings; returns the count."""

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Delete the entire collection."""

    # -- reads --------------------------------------------------------
    @abc.abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, similarity)`` pairs ordered by similarity."""

    @abc.abstractmethod
    def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        """Fetch full chunks by id (used for citations and context assembly)."""

    @abc.abstractmethod
    def count(self, *, filters: dict[str, object] | None = None) -> int:
        """Number of stored chunks (optionally filtered)."""

    @abc.abstractmethod
    def document_ids(self) -> list[str]:
        """Distinct document ids present in the store."""

    @abc.abstractmethod
    def counts_by_document(self) -> dict[str, int]:
        """Chunk count per document id."""

    @abc.abstractmethod
    def all_chunks(self, document_id: str | None = None) -> list[DocumentChunk]:
        """Every chunk in the store (optionally one document), in reading order."""

    # -- helpers ------------------------------------------------------
    def exists(self, chunk_id: str) -> bool:
        return bool(self.get_chunks([chunk_id]))

    def describe(self) -> dict[str, object]:
        return {"provider": self.name, "chunks": self.count(), "documents": len(self.document_ids())}


# ---------------------------------------------------------------------------
class InMemoryVectorStore(VectorStore):
    """Dict-backed store - fast, dependency-free, ideal for tests."""

    name = "memory"

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        with self._lock:
            for chunk, vector in zip(chunks, embeddings, strict=False):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = np.asarray(vector, dtype=np.float32)
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            targets = [
                chunk_id for chunk_id, chunk in self._chunks.items()
                if chunk.metadata.document_id == document_id
            ]
            for chunk_id in targets:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)
        return len(targets)

    def reset(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._vectors.clear()

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        with self._lock:
            items = list(self._chunks.items())
        scored: list[tuple[str, float]] = []
        for chunk_id, chunk in items:
            if filters and not _matches(chunk.metadata.model_dump(), filters):
                continue
            other = self._vectors.get(chunk_id)
            if other is None:
                continue
            scored.append((chunk_id, float(np.dot(vector, other))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        with self._lock:
            return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]

    def count(self, *, filters: dict[str, object] | None = None) -> int:
        with self._lock:
            chunks = list(self._chunks.values())
        if not filters:
            return len(chunks)
        return sum(1 for chunk in chunks if _matches(chunk.metadata.model_dump(), filters))

    def document_ids(self) -> list[str]:
        with self._lock:
            return list({chunk.metadata.document_id for chunk in self._chunks.values()})

    def counts_by_document(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for chunk in self._chunks.values():
                counts[chunk.metadata.document_id] = counts.get(chunk.metadata.document_id, 0) + 1
            return counts

    def all_chunks(self, document_id: str | None = None) -> list[DocumentChunk]:
        with self._lock:
            chunks = [
                chunk
                for chunk in self._chunks.values()
                if document_id is None or chunk.metadata.document_id == document_id
            ]
        chunks.sort(key=lambda item: (item.metadata.chunk_index, item.chunk_id))
        return chunks


# ---------------------------------------------------------------------------
class ChromaVectorStore(VectorStore):
    """ChromaDB-backed store (persistent, cosine distance)."""

    name = "chroma"

    def __init__(self, settings: Settings | None = None, collection_name: str | None = None):
        self._settings = settings or get_settings()
        self._collection_name = collection_name or self._settings.vector_store_collection
        self._client = None
        self._collection = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _ensure_collection(self) -> object:
        if self._collection is not None:
            return self._collection
        with self._lock:
            if self._collection is not None:
                return self._collection
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            path = self._settings.vector_db_path
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Chroma collection '%s' ready at %s", self._collection_name, path)
        return self._collection

    # ------------------------------------------------------------------
    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        collection = self._ensure_collection()
        matrix = l2_normalize(np.asarray(embeddings, dtype=np.float32))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, object]] = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(_metadata_payload(chunk))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=matrix.tolist())
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        collection = self._ensure_collection()
        existing = collection.get(where={"document_id": document_id}, include=[]) or {}
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=list(ids))
        return len(ids)

    def reset(self) -> None:
        collection = self._ensure_collection()
        try:
            collection.delete(where={})
        except Exception:  # pragma: no cover - older chroma builds
            collection.delete(where_document={"$contains": ""})

    # ------------------------------------------------------------------
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        collection = self._ensure_collection()
        if collection.count() == 0:
            return []
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        result = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=max(1, min(top_k, collection.count())),
            where=_chroma_where(filters),
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        pairs: list[tuple[str, float]] = []
        for chunk_id, distance in zip(ids, distances, strict=False):
            # Cosine distance -> similarity in [0, 1]
            pairs.append((chunk_id, float(max(0.0, 1.0 - float(distance)))))
        return pairs

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return []
        collection = self._ensure_collection()
        found: dict[str, DocumentChunk] = {}
        # Chroma's `get` has a practical batch limit; 500 keeps us safe.
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            payload = collection.get(ids=batch, include=["documents", "metadatas"]) or {}
            for chunk_id, document, metadata in zip(
                payload.get("ids") or [],
                payload.get("documents") or [],
                payload.get("metadatas") or [],
                strict=False,
            ):
                found[chunk_id] = _chunk_from_payload(chunk_id, document, metadata)
        return [found[chunk_id] for chunk_id in ids if chunk_id in found]

    def count(self, *, filters: dict[str, object] | None = None) -> int:
        collection = self._ensure_collection()
        try:
            return int(collection.count(where=_chroma_where(filters)))
        except Exception:  # pragma: no cover - defensive
            return int(collection.count())

    def document_ids(self) -> list[str]:
        return list(self.counts_by_document())

    def counts_by_document(self) -> dict[str, int]:
        collection = self._ensure_collection()
        payload = collection.get(include=["metadatas"]) or {}
        metadatas = payload.get("metadatas") or []
        counts: dict[str, int] = {}
        for metadata in metadatas:
            document_id = str((metadata or {}).get("document_id", "unknown"))
            counts[document_id] = counts.get(document_id, 0) + 1
        return counts

    def all_chunks(self, document_id: str | None = None) -> list[DocumentChunk]:
        """Fetch every chunk (optionally for one document) - used for re-indexing."""
        collection = self._ensure_collection()
        payload = collection.get(
            where=_chroma_where({"document_id": document_id} if document_id else None),
            include=["documents", "metadatas"],
        ) or {}
        chunks: list[DocumentChunk] = []
        for chunk_id, document, metadata in zip(
            payload.get("ids") or [],
            payload.get("documents") or [],
            payload.get("metadatas") or [],
            strict=False,
        ):
            chunks.append(_chunk_from_payload(chunk_id, document, metadata))
        chunks.sort(key=lambda item: (item.metadata.chunk_index, item.chunk_id))
        return chunks


# ---------------------------------------------------------------------------
def _metadata_payload(chunk: DocumentChunk) -> dict[str, object]:
    """Flatten a chunk into Chroma-compatible scalar metadata."""
    metadata = chunk.metadata
    payload: dict[str, object] = {
        "document_id": metadata.document_id,
        "document_name": metadata.document_name,
        "document_type": str(metadata.document_type),
        "chunk_index": int(metadata.chunk_index),
        "kind": chunk.kind.value if isinstance(chunk.kind, ChunkKind) else str(chunk.kind),
    }
    if metadata.page is not None:
        payload["page"] = int(metadata.page)
    for key in ("section", "chapter", "source"):
        value = getattr(metadata, key)
        if value:
            payload[key] = str(value)[:480]
    if chunk.heading_path:
        payload["heading_path"] = chunk.heading_path[:480]
    return payload


def _chunk_from_payload(chunk_id: str, document: str | None, metadata: dict | None) -> DocumentChunk:
    metadata = metadata or {}
    meta = ChunkMetadata(
        document_id=str(metadata.get("document_id", "unknown")),
        document_name=str(metadata.get("document_name", "Unknown document")),
        document_type=str(metadata.get("document_type", "unknown")),  # type: ignore[arg-type]
        chunk_index=int(metadata.get("chunk_index", 0)),
        chunk_id=chunk_id,
        page=int(metadata["page"]) if metadata.get("page") is not None else None,
        section=metadata.get("section"),
        chapter=metadata.get("chapter"),
        source=metadata.get("source"),
    )
    kind_value = str(metadata.get("kind", "text"))
    try:
        kind = ChunkKind(kind_value)
    except ValueError:
        kind = ChunkKind.TEXT
    return DocumentChunk(
        chunk_id=chunk_id,
        text=document or "",
        metadata=meta,
        kind=kind,
        heading_path=metadata.get("heading_path"),
    )


def _matches(meta: dict[str, object], filters: dict[str, object]) -> bool:
    """Equality (or ``$in``) filter evaluation shared by the in-memory store."""
    for key, expected in filters.items():
        actual = meta.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual is None or str(actual) not in {str(v) for v in expected["$in"]}:
                return False
            continue
        if actual is None or str(actual) != str(expected):
            return False
    return True


def _chroma_where(filters: dict[str, object] | None) -> dict | None:
    """Build a Chroma ``where`` clause (implicit AND)."""
    if not filters:
        return None
    if len(filters) == 1:
        key, value = next(iter(filters.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in filters.items()]}


# ---------------------------------------------------------------------------
_STORES: dict[str, VectorStore] = {}


def get_vector_store(
    settings: Settings | None = None, *, force: bool = False
) -> VectorStore:
    """Return the configured vector store (cached)."""
    settings = settings or get_settings()
    key = f"{settings.vector_store_provider}|{settings.vector_db_path}|{settings.vector_store_collection}"
    if force:
        _STORES.pop(key, None)
    if key not in _STORES:
        if settings.vector_store_provider == "memory":
            _STORES[key] = InMemoryVectorStore()
        else:
            _STORES[key] = ChromaVectorStore(settings)
    return _STORES[key]


def reset_vector_store_cache() -> None:
    """Drop cached store instances (tests, re-index, settings changes)."""
    _STORES.clear()