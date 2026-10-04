"""Ingestion orchestration.

    file -> validate -> load/parse -> clean -> semantic chunk
         -> embed -> vector store + BM25 index -> database record

The whole operation is idempotent: re-indexing a document removes its previous
chunks first, and a content hash short-circuits exact duplicates.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.loaders.base import LoadError
from workcompanion.loaders.registry import get_loader, supported_extensions
from workcompanion.rag.chunking import SemanticChunker
from workcompanion.rag.embeddings import EmbeddingProvider
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import VectorStore
from workcompanion.schemas.documents import DocumentChunk, ParsedDocument
from workcompanion.utils.cache import DiskCache
from workcompanion.utils.file_utils import (
    FileValidationError,
    describe_file,
    sanitize_filename,
    unique_path,
    validate_upload,
)
from workcompanion.utils.telemetry import get_telemetry

logger = get_logger(__name__)


def _document_id_for(sha256: str) -> str:
    """A stable document id derived from content.

    Loaders default to a random UUID, which makes re-ingesting the same file
    create a second document. Deriving the id from the content hash means the
    same bytes always resolve to the same document, so ingestion is idempotent
    and duplicate detection is a cheap existence check.
    """
    return hashlib.sha256(sha256.encode()).hexdigest()[:32]


@dataclass(slots=True)
class IngestionResult:
    """Outcome of ingesting one file."""

    document_id: str
    document_name: str
    status: str = "pending"  # indexed | failed | skipped
    chunk_count: int = 0
    page_count: int = 0
    char_count: int = 0
    stored_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    used_ocr: bool = False
    latency_ms: float = 0.0
    duplicate_of: str | None = None
    parsed: ParsedDocument | None = None

    @property
    def ok(self) -> bool:
        return self.status == "indexed"

    def summary(self) -> str:
        if self.status == "indexed":
            return (
                f"{self.document_name}: {self.chunk_count} chunks from {self.page_count} pages "
                f"({self.char_count:,} chars)"
            )
        if self.status == "skipped":
            return f"{self.document_name}: skipped ({self.error})"
        return f"{self.document_name}: failed - {self.error}"


class IngestionService:
    """Parses, chunks and indexes documents into the knowledge base."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
    ):
        self._settings = settings or get_settings()
        self._store = vector_store
        self._sparse = sparse_index
        self._embeddings = embedding_provider
        self._chunker = SemanticChunker(self._settings)
        self._cache = cache
        self._telemetry = get_telemetry()

    # ------------------------------------------------------------------
    @property
    def embedding_cache(self) -> DiskCache | None:
        if self._cache is None and self._settings.embedding_cache_enabled:
            self._cache = DiskCache(
                self._settings.cache_dir / "embeddings",
                ttl_seconds=self._settings.cache_ttl_seconds,
                enabled=self._settings.cache_enabled,
            )
        return self._cache

    @property
    def index_path(self) -> Path:
        return Path(self._settings.vector_db_path) / "bm25_index.json"

    # ------------------------------------------------------------------
    def parse_file(self, path: str | Path) -> ParsedDocument:
        """Parse a file into a :class:`ParsedDocument` without indexing it."""
        target = Path(path)
        if not target.is_file():
            raise LoadError(f"File not found: {target}")
        loader = get_loader(target)
        return loader.load(target)

    def ingest_text(
        self,
        text: str,
        *,
        document_name: str,
        source_type: str = "txt",
        subject: str | None = None,
    ) -> tuple[ParsedDocument, list[DocumentChunk]]:
        """Chunk raw text directly (used by the demo dataset and tests)."""
        from workcompanion.schemas.documents import RawPage
        from workcompanion.utils.text_utils import clean_text

        cleaned = clean_text(text)
        document = ParsedDocument(
            document_name=sanitize_filename(document_name),
            source_type=source_type,  # type: ignore[arg-type]
            pages=[RawPage(page_number=1, text=cleaned)],
            full_text=cleaned,
            subject=subject,
        )
        document.page_count = 1
        document.char_count = len(cleaned)
        chunks = self._chunker.chunk_document(document)
        return document, chunks

    # ------------------------------------------------------------------
    def ingest_file(
        self,
        path: str | Path,
        *,
        subject: str | None = None,
        persist: bool = True,
        allow_duplicate: bool = False,
        tags: list[str] | None = None,
    ) -> IngestionResult:
        """Full ingestion pipeline for one file."""
        started = time.perf_counter()
        telemetry = self._telemetry
        source = Path(path)

        try:
            info = describe_file(source, with_hash=True)
        except FileValidationError as exc:
            return IngestionResult(
                document_id="", document_name=source.name, status="failed", error=str(exc)
            )

        try:
            safe_name, _extension = validate_upload(
                info.filename,
                info.size_bytes,
                allowed_extensions=self._settings.allowed_upload_extensions,
                max_mb=self._settings.max_upload_mb,
            )
        except FileValidationError as exc:
            return IngestionResult(
                document_id="", document_name=info.filename, status="failed", error=str(exc)
            )

        if info.sha256 and not allow_duplicate and self._is_duplicate(info.sha256):
            return IngestionResult(
                document_id="",
                document_name=safe_name,
                status="skipped",
                error="this exact file is already indexed",
            )

        # 1. Parse ------------------------------------------------------
        try:
            document = self.parse_file(source)
        except LoadError as exc:
            logger.warning("Failed to parse '%s': %s", info.filename, exc)
            return IngestionResult(
                document_id="",
                document_name=safe_name,
                status="failed",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # unexpected parser failure
            logger.exception("Unexpected error parsing '%s'", info.filename)
            return IngestionResult(
                document_id="",
                document_name=safe_name,
                status="failed",
                error=f"Unexpected parser error: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # Keep the sanitised filename when the loader echoed the raw name back.
        if document.document_name == info.filename:
            document.document_name = safe_name
        if subject:
            document.subject = subject
        if tags:
            document.tags = list(dict.fromkeys([*document.tags, *tags]))

        # Content-address the document. Loaders mint a random UUID by default,
        # which means re-ingesting the same file creates a *second* document and
        # doubles every chunk in the index. Deriving the id from the content
        # makes ingestion idempotent: the same bytes always land on the same
        # document, so the store upserts over the previous copy.
        if info.sha256:
            # Content only, no filename: the same bytes re-uploaded under a
            # different name must still resolve to the same document.
            document.document_id = _document_id_for(info.sha256)

        # 2. Store the file (optional) ----------------------------------
        stored_path: str | None = str(source)
        if persist and self._settings.documents_dir is not None:
            try:
                destination = unique_path(Path(self._settings.documents_dir), safe_name)
                destination.write_bytes(source.read_bytes())
                stored_path = str(destination)
                document.source_path = stored_path
            except OSError as exc:  # pragma: no cover - disk issues
                document.add_warning(f"Could not copy the file into the library: {exc}")

        # 3. Chunk ------------------------------------------------------
        try:
            chunks = self._chunker.chunk_document(document)
        except Exception as exc:
            logger.exception("Chunking failed for '%s'", safe_name)
            return IngestionResult(
                document_id=document.document_id,
                document_name=safe_name,
                status="failed",
                error=f"Chunking failed: {exc}",
                stored_path=stored_path,
                parsed=document,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if not chunks:
            return IngestionResult(
                document_id=document.document_id,
                document_name=safe_name,
                status="failed",
                error="No indexable content was found (the document may be empty or scanned).",
                stored_path=stored_path,
                parsed=document,
                warnings=list(document.warnings),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 4. Embed + store ---------------------------------------------
        try:
            # Re-indexing replaces the document instead of adding to it. Chunk
            # ids are minted fresh on every run, so an upsert alone would leave
            # the previous copy behind and silently double the corpus.
            try:
                stale = self._store.delete_document(document.document_id)
                if stale:
                    self._sparse.remove_document(document.document_id)
                    logger.info(
                        "Replacing %d existing chunk(s) for '%s'", stale, safe_name
                    )
            except Exception:  # pragma: no cover - store without document deletes
                logger.debug("Store could not clear the previous copy first.")
            self.index_chunks(chunks)
        except Exception as exc:
            logger.exception("Indexing failed for '%s'", safe_name)
            return IngestionResult(
                document_id=document.document_id,
                document_name=safe_name,
                status="failed",
                error=f"Indexing failed: {exc}",
                stored_path=stored_path,
                parsed=document,
                warnings=list(document.warnings),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        telemetry.record("ingest.total_ms", latency_ms)
        telemetry.count("ingest.documents")

        return IngestionResult(
            document_id=document.document_id,
            document_name=safe_name,
            status="indexed",
            chunk_count=len(chunks),
            page_count=document.page_count,
            char_count=document.char_count,
            stored_path=stored_path,
            warnings=list(document.warnings),
            used_ocr=document.used_ocr,
            latency_ms=latency_ms,
            parsed=document,
        )

    # ------------------------------------------------------------------
    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed chunks and write them to the vector store + BM25 index."""
        texts = [_embedding_text(chunk) for chunk in chunks]
        vectors = self._embeddings.embed_with_cache(texts, cache=self.embedding_cache)
        stored = self._store.upsert(chunks, vectors)
        self._sparse.add_chunks(chunks)
        return stored

    def index_parsed_document(self, document: ParsedDocument) -> tuple[ParsedDocument, list[DocumentChunk]]:
        """Chunk + index an already-parsed document (demo data, re-indexing)."""
        chunks = self._chunker.chunk_document(document)
        if chunks:
            self.index_chunks(chunks)
        return document, chunks

    # ------------------------------------------------------------------
    def remove_document(self, document_id: str) -> int:
        """Remove a document from both indexes (and the saved file)."""
        removed_chunks = self._store.delete_document(document_id)
        self._sparse.remove_document(document_id)
        self.persist_sparse_index()
        return removed_chunks

    def _is_duplicate(self, sha256: str) -> bool:
        """Whether this exact content is already indexed.

        Backed by the vector store rather than the database, because ingestion
        deliberately has no session of its own. The document id is derived from
        the content hash, so a plain existence check is enough.
        """
        try:
            # ``VectorStore.exists`` takes a *chunk* id, so check the document
            # roster instead.
            return _document_id_for(sha256) in set(self._store.document_ids())
        except Exception:  # pragma: no cover - a store backend that cannot list
            logger.debug("Store cannot answer duplicate checks; allowing ingest.")
            return False

    def persist_sparse_index(self) -> None:
        """Write the BM25 index to disk so it survives a restart."""
        try:
            self._sparse.save(self.index_path)
        except OSError as exc:  # pragma: no cover - disk issues
            logger.warning("Could not persist the BM25 index: %s", exc)

    def load_sparse_index(self) -> bool:
        """Load a previously persisted BM25 index."""
        return self._sparse.load(self.index_path)

    def rebuild_sparse_index(self) -> int:
        """Rebuild the BM25 index from the vector store (self-healing)."""
        from workcompanion.rag.vector_store import ChromaVectorStore

        self._sparse.clear()
        if isinstance(self._store, ChromaVectorStore):
            chunks = self._store.all_chunks()
        else:  # pragma: no cover - memory store cannot enumerate cheaply
            return 0
        count = self._sparse.add_chunks(chunks)
        self.persist_sparse_index()
        logger.info("Rebuilt the BM25 index from %d stored chunks", count)
        return count

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Index statistics for the Knowledge page."""
        return {
            "chunks": self._store.count(),
            "documents": len(self._store.document_ids()),
            "sparse_documents": self._sparse.size,
            "embedding_provider": self._embeddings.describe(),
            "chunk_target_chars": self._settings.chunk_target_chars,
            "index_path": str(self.index_path),
        }


def _embedding_text(chunk: DocumentChunk) -> str:
    """What actually gets embedded: heading context + body.

    Prefixing the heading path measurably improves retrieval for questions that
    name a section rather than its content.
    """
    if chunk.heading_path:
        return f"{chunk.heading_path}\n{chunk.text}"
    return chunk.text


def supported_types_help() -> str:
    """A human-readable list of accepted uploads."""
    return ", ".join(supported_extensions())