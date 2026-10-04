"""Document / page / chunk schemas shared by loaders, chunking and storage."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["pdf", "docx", "pptx", "txt", "markdown", "csv", "html", "image", "demo", "unknown"]


class ChunkKind(str, Enum):
    """Semantic role of a chunk - drives retrieval and prompting."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    DEFINITION = "definition"
    EQUATION = "equation"
    SUMMARY = "summary"
    QUESTION_ANSWER = "qa"


class RawPage(BaseModel):
    """A single page (or sheet / slide) of extracted content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_number: int = Field(..., ge=1, description="1-based page / slide number.")
    text: str = ""
    tables: list[str] = Field(default_factory=list, description="Markdown tables.")
    heading_hint: str | None = None
    used_ocr: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text) + sum(len(t) for t in self.tables)


class ParsedDocument(BaseModel):
    """Normalised output of every loader - the ingestion contract."""

    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    document_name: str
    source_path: str | None = None
    source_type: SourceType = "unknown"
    pages: list[RawPage] = Field(default_factory=list)
    full_text: str = ""
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    tags: list[str] = Field(default_factory=list)
    page_count: int = 0
    char_count: int = 0
    used_ocr: bool = False
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def tables(self) -> list[str]:
        return [table for page in self.pages for table in page.tables]

    def content_hash(self) -> str:
        """Stable hash of the text content, used for duplicate detection."""
        basis = self.full_text or "\n".join(page.text for page in self.pages)
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


class ChunkMetadata(BaseModel):
    """Provenance attached to every chunk (citation-ready)."""

    document_id: str
    document_name: str
    document_type: SourceType = "unknown"
    chunk_index: int = 0
    chunk_id: str = ""
    page: int | None = None
    section: str | None = None
    chapter: str | None = None
    source: str | None = None
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        bits = [self.document_name]
        if self.page is not None:
            bits.append(f"p. {self.page}")
        if self.chapter:
            bits.append(self.chapter)
        elif self.section:
            bits.append(self.section)
        return ", ".join(bits)


class DocumentChunk(BaseModel):
    """A semantically bounded, retrievable unit of text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    text: str
    metadata: ChunkMetadata
    kind: ChunkKind = ChunkKind.TEXT
    heading_path: str | None = None
    key_terms: list[str] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def preview(self, limit: int = 240) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[:limit] + "\u2026"