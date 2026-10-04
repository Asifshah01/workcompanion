"""Plain-text and Markdown loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.loaders.base import BaseLoader, LoadError, paginate_text
from workcompanion.loaders.tables import detect_pipe_table
from workcompanion.schemas.documents import ParsedDocument, RawPage
from workcompanion.utils.text_utils import clean_text, looks_like_heading, strip_markdown_noise

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FIELD_RE = re.compile(r"^([A-Za-z_][\w\-]*)\s*:\s*(.+)$")


class TextLoader(BaseLoader):
    """Parse ``.txt`` and ``.md``/``.markdown`` files."""

    extensions = (".txt", ".text", ".log")
    source_type = "txt"

    def load(self, path: str | Path, **kwargs: Any) -> ParsedDocument:
        target = Path(path)
        if not target.is_file():
            raise LoadError(f"Text file not found: {target}")
        try:
            raw = self.load_text(target)
        except OSError as exc:
            raise LoadError(f"Could not read '{target.name}': {exc}") from exc

        document = ParsedDocument(
            document_name=target.name,
            source_path=str(target.resolve()),
            source_type="txt",
        )
        raw = self._apply_frontmatter(raw, document)

        cleaned = strip_markdown_noise(clean_text(raw))
        tables = [table for table in (detect_pipe_table(page) for page in [cleaned]) if table]
        document.pages = paginate_text(cleaned)
        if tables:
            document.pages[0].tables.append(tables[0])
        document = self.finalize(document)
        document.title = document.title or self.detect_first_heading(document.pages) or target.stem
        logger.info("Parsed text '%s': %d pages, %d chars", target.name, document.page_count, document.char_count)
        return document

    @staticmethod
    def _apply_frontmatter(raw: str, document: ParsedDocument) -> str:
        """Read YAML-ish frontmatter (title/author/tags) and strip it from the body."""
        match = _FRONTMATTER_RE.match(raw)
        if not match:
            return raw
        for line in match.group(1).splitlines():
            field = _FIELD_RE.match(line.strip())
            if not field:
                continue
            key, value = field.group(1).lower(), field.group(2).strip().strip("\"'")
            if key == "title":
                document.title = value
            elif key == "author":
                document.author = value
            elif key in {"subject", "course"}:
                document.subject = value
            elif key in {"tags", "keywords"}:
                document.tags = [tag.strip() for tag in re.split(r"[,;]", value) if tag.strip()][:12]
        return raw[match.end() :]


class MarkdownLoader(TextLoader):
    """Markdown-aware variant that promotes ``#`` headings to page titles."""

    extensions = (".md", ".markdown", ".mdx", ".rst")
    source_type = "markdown"

    def load(self, path: str | Path, **kwargs: Any) -> ParsedDocument:
        document = super().load(path, **kwargs)
        document.source_type = "markdown"
        # Give every top-level section its own synthetic page so that
        # chunk-level chapter metadata is accurate.
        sectioned = self._split_by_heading(document)
        if len(sectioned) > 1:
            document.pages = sectioned
            document = self.finalize(document)
        return document

    @staticmethod
    def _split_by_heading(document: ParsedDocument) -> list[RawPage]:
        pages: list[RawPage] = []
        current_heading: str | None = None
        buffer: list[str] = []
        tables: list[str] = []

        def flush() -> None:
            if buffer or current_heading:
                pages.append(
                    RawPage(
                        page_number=len(pages) + 1,
                        text="\n\n".join(buffer).strip(),
                        heading_hint=current_heading,
                        tables=list(tables),
                    )
                )
            buffer.clear()
            tables.clear()

        for page in document.pages:
            for block in page.text.split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                lines = block.splitlines()
                first_line = lines[0].strip()
                if re.match(r"^#{1,3}\s+\S", first_line) or looks_like_heading(first_line):
                    flush()
                    current_heading = first_line.lstrip("#").strip()
                    buffer.append(f"## {current_heading}")
                    # A heading and its body usually share one block, because
                    # hard-wrapped prose contains no blank lines. The remainder
                    # must be carried over or the section body is lost.
                    remainder = "\n".join(lines[1:]).strip()
                    if remainder:
                        buffer.append(remainder)
                else:
                    buffer.append(block)
            tables.extend(page.tables)
        flush()
        return pages