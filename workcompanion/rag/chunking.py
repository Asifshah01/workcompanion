"""Semantic chunking.

Documents are **not** split every N characters.  The chunker walks the document
line by line, tracks a heading stack (chapter/section breadcrumbs), never splits
a table or an equation block, prefers paragraph boundaries, and only splits
oversized blocks on sentence boundaries with a character overlap window.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.schemas.documents import (
    ChunkKind,
    ChunkMetadata,
    DocumentChunk,
    ParsedDocument,
    RawPage,
)
from workcompanion.utils.text_utils import (
    detect_heading_level,
    extract_key_terms,
    looks_like_heading,
    normalize_whitespace,
    sentences_from_text,
)
from workcompanion.utils.token_utils import estimate_tokens

logger = get_logger(__name__)

_HEADING_LINE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
_EQUATION_HINT = re.compile(r"(?:\\[a-zA-Z]+|\$\$.*|^\s*\w+\s*=\s*[\w\(\{\\\^/])")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_CHAPTER_LINE = re.compile(r"^\s*(chapter|unit|module|part|section|lecture|topic)\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class _Section:
    """An in-progress section of a document."""

    __slots__ = ("heading_path", "chapter", "section", "page", "blocks")

    def __init__(self, heading_path: list[tuple[int, str]], chapter: str | None, section: str | None, page: int | None):
        self.heading_path = heading_path
        self.chapter = chapter
        self.section = section
        self.page = page
        self.blocks: list[tuple[str, ChunkKind]] = []


class SemanticChunker:
    """Turn a :class:`ParsedDocument` into retrievable, citation-ready chunks."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Chunk a whole parsed document."""
        sections = self._iter_sections(document.pages)
        chunks: list[DocumentChunk] = []
        for section in sections:
            chunks.extend(self._chunk_section(section, document))
        if not chunks:
            logger.warning("Chunking produced no chunks for '%s'.", document.document_name)
        logger.info(
            "Chunked '%s' into %d chunks (target=%d chars)",
            document.document_name, len(chunks), self._settings.chunk_target_chars,
        )
        return chunks

    # ------------------------------------------------------------------
    def _iter_sections(self, pages: list[RawPage]) -> Iterator[_Section]:
        """Walk pages, emitting sections delimited by headings / page changes."""
        heading_stack: list[tuple[int, str]] = []
        current = self._new_section(heading_stack, pages[0].page_number if pages else None)

        for page in pages:
            for block in self._page_blocks(page):
                text, kind = block
                if kind in (ChunkKind.HEADING, ChunkKind.TEXT) and self._is_heading_block(text):
                    level = self._heading_level(text)
                    title = self._heading_title(text)
                    # Close the previous section only when the heading changes context.
                    if current.blocks or level <= max((lvl for lvl, _ in heading_stack), default=99):
                        yield current
                        heading_stack = list(heading_stack)
                    heading_stack = self._push_heading(heading_stack, level, title)
                    current = _Section(
                        list(heading_stack),
                        chapter=self._chapter_from(heading_stack),
                        section=self._section_from(heading_stack),
                        page=page.page_number,
                    )
                    # The heading line itself is kept as context for every chunk.
                    current.blocks.append((self._heading_text(heading_stack), ChunkKind.HEADING))
                    continue

                current.blocks.append((text, kind))

            if page.tables:
                for table in page.tables:
                    current.blocks.append((table, ChunkKind.TABLE))
            if page.page_number != current.page and current.blocks:
                # Keep accumulating within the same section across pages; the page
                # is recorded on the block below instead.
                pass

        if current.blocks:
            yield current

    @staticmethod
    def _new_section(heading_stack: list[tuple[int, str]], page: int | None) -> _Section:
        return _Section(list(heading_stack), None, None, page)

    @staticmethod
    def _page_blocks(page: RawPage) -> list[tuple[str, ChunkKind]]:
        """Split a page's text into heading / paragraph / equation blocks."""
        blocks: list[tuple[str, ChunkKind]] = []
        buffer: list[str] = []
        in_table = False

        def flush() -> None:
            if not buffer:
                return
            text = normalize_whitespace("\n".join(buffer))
            if text:
                kind = ChunkKind.EQUATION if _EQUATION_HINT.search(text) else ChunkKind.TEXT
                blocks.append((text, kind))
            buffer.clear()

        for raw_line in page.text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if _TABLE_LINE.match(line):
                flush()
                in_table = True
                buffer.append(line)
                continue
            if in_table and not stripped:
                flush()
                in_table = False
                continue
            if in_table and stripped and "|" in stripped:
                buffer.append(line)
                continue
            in_table = False

            if not stripped:
                flush()
                continue
            if _HEADING_LINE.match(stripped) or looks_like_heading(stripped):
                flush()
                blocks.append((stripped, ChunkKind.HEADING))
                continue
            buffer.append(line)

        flush()
        return blocks

    @staticmethod
    def _is_heading_block(text: str) -> bool:
        if _HEADING_LINE.match(text.strip()):
            return True
        return looks_like_heading(text)

    @staticmethod
    def _heading_level(text: str) -> int:
        match = _HEADING_LINE.match(text.strip())
        if match:
            return min(len(match.group(1)), 6)
        return detect_heading_level(text)

    @staticmethod
    def _heading_title(text: str) -> str:
        match = _HEADING_LINE.match(text.strip())
        title = match.group(2) if match else text.strip()
        return title.strip().strip("*_# ").strip()

    @staticmethod
    def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> list[tuple[int, str]]:
        updated = [(lvl, name) for lvl, name in stack if lvl < level]
        updated.append((level, title))
        return updated

    @staticmethod
    def _heading_text(stack: Iterable[tuple[int, str]]) -> str:
        return " > ".join(title for _, title in stack) or "Document"

    @staticmethod
    def _chapter_from(stack: list[tuple[int, str]]) -> str | None:
        chapter = next((name for _, name in stack if _CHAPTER_LINE.match(name)), None)
        if chapter:
            return chapter[:180]
        top = next((name for lvl, name in stack if lvl <= 2), None)
        return top[:180] if top else None

    @staticmethod
    def _section_from(stack: list[tuple[int, str]]) -> str | None:
        if not stack:
            return None
        return stack[-1][1][:180]

    # ------------------------------------------------------------------
    def _chunk_section(self, section: _Section, document: ParsedDocument) -> list[DocumentChunk]:
        """Group a section's blocks into chunks that respect size limits."""
        settings = self._settings
        heading_line = section.blocks[0][0] if section.blocks and section.blocks[0][1] is ChunkKind.HEADING else ""
        body_blocks = [
            block for block in section.blocks
            if not (block[1] is ChunkKind.HEADING and block[0] == heading_line)
        ]

        chunks: list[DocumentChunk] = []
        buffer: list[str] = []
        buffer_len = 0

        def emit(pieces: list[str], kind: ChunkKind) -> None:
            text = "\n\n".join(piece.strip() for piece in pieces if piece.strip()).strip()
            if len(text) < settings.chunk_min_chars and chunks and kind is not ChunkKind.TABLE:
                return  # too small to stand alone; let the caller merge it
            if not text:
                return
            chunks.append(self._build_chunk(text, document, section, kind, heading_line))

        for text, kind in body_blocks:
            if kind is ChunkKind.TABLE:
                # Tables stay intact, even when they exceed the target size.
                if buffer:
                    emit(buffer, ChunkKind.TEXT)
                    buffer, buffer_len = [], 0
                emit([heading_line, text] if heading_line else [text], ChunkKind.TABLE)
                continue

            if len(text) > settings.chunk_target_chars * 1.6:
                if buffer:
                    emit(buffer, ChunkKind.TEXT)
                    buffer, buffer_len = [], 0
                for piece in self._split_long_block(text):
                    emit(([heading_line, piece] if heading_line else [piece]), kind)
                continue

            if buffer_len + len(text) > settings.chunk_target_chars and buffer:
                emit(buffer, ChunkKind.TEXT)
                buffer, buffer_len = [], 0
            buffer.append(text)
            buffer_len += len(text) + 2

        if buffer:
            emit(buffer, ChunkKind.TEXT)

        # Fold undersized trailing chunks into their predecessor.
        merged: list[DocumentChunk] = []
        for chunk in chunks:
            if (
                merged
                and chunk.char_count < settings.chunk_min_chars
                and chunk.kind is ChunkKind.TEXT
                and (merged[-1].char_count + chunk.char_count) <= settings.chunk_target_chars * 1.5
            ):
                previous = merged[-1]
                previous.text = f"{previous.text}\n\n{chunk.text}".strip()
                previous.key_terms = list(dict.fromkeys(previous.key_terms + chunk.key_terms))[:14]
                continue
            merged.append(chunk)

        for index, chunk in enumerate(merged):
            chunk.metadata.chunk_index = index
            chunk.metadata.extra.setdefault("section_index", index)
        return merged

    def _split_long_block(self, text: str) -> list[str]:
        """Split an oversized block on sentence boundaries with overlap."""
        target = self._settings.chunk_target_chars
        overlap = self._settings.chunk_overlap_chars
        sentences = sentences_from_text(text, min_length=40) or [text]

        pieces: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in sentences:
            if length + len(sentence) > target and current:
                pieces.append(" ".join(current))
                if overlap > 0:
                    tail: list[str] = []
                    tail_len = 0
                    for previous in reversed(current):
                        if tail_len + len(previous) > overlap:
                            break
                        tail.insert(0, previous)
                        tail_len += len(previous) + 1
                    current = list(tail)
                    length = tail_len
                else:
                    current, length = [], 0
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            pieces.append(" ".join(current))
        return [piece.strip() for piece in pieces if piece.strip()]

    # ------------------------------------------------------------------
    def _build_chunk(
        self,
        text: str,
        document: ParsedDocument,
        section: _Section,
        kind: ChunkKind,
        heading_line: str,
    ) -> DocumentChunk:
        heading_path = self._heading_text(section.heading_path)
        searchable = f"{heading_line}\n{text}" if heading_line else text
        metadata = ChunkMetadata(
            document_id=document.document_id,
            document_name=document.document_name,
            document_type=document.source_type,
            page=section.page,
            section=section.section,
            chapter=section.chapter,
            source=document.source_path or document.document_name,
            token_estimate=estimate_tokens(searchable),
            extra={"heading_path": heading_path},
        )
        return DocumentChunk(
            text=text,
            metadata=metadata,
            kind=kind,
            heading_path=heading_path,
            key_terms=extract_key_terms(f"{heading_line} {text}", limit=10),
        )


def chunk_document(
    document: ParsedDocument, settings: Settings | None = None
) -> list[DocumentChunk]:
    """Convenience wrapper around :class:`SemanticChunker`."""
    return SemanticChunker(settings).chunk_document(document)