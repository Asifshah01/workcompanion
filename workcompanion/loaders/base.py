"""Loader base class and shared helpers."""

from __future__ import annotations

import abc
from pathlib import Path

from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.documents import ParsedDocument, RawPage
from workcompanion.utils.text_utils import clean_text, looks_like_heading

logger = get_logger(__name__)

#: Roughly how many characters of text to place on a synthetic "page".
SYNTHETIC_PAGE_CHARS = 3000


class LoadError(RuntimeError):
    """Raised when a document cannot be parsed."""


class BaseLoader(abc.ABC):
    """Common behaviour for every document loader."""

    #: File extensions handled by this loader (lowercase, with dot).
    extensions: tuple[str, ...] = ()
    #: Value stored in ``ParsedDocument.source_type``.
    source_type: str = "unknown"

    def can_load(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.extensions

    @abc.abstractmethod
    def load(self, path: str | Path, **kwargs: object) -> ParsedDocument:
        """Parse ``path`` into a :class:`ParsedDocument`."""

    # ------------------------------------------------------------------
    def load_text(self, path: str | Path) -> str:
        """Read a file as UTF-8 text with a safe fallback chain."""
        raw = Path(path).read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def finalize(self, document: ParsedDocument) -> ParsedDocument:
        """Apply common clean-up and derive counts."""
        for page in document.pages:
            page.text = clean_text(page.text)
            page.tables = [clean_text(table) for table in page.tables if table and table.strip()]
        document.pages = [page for page in document.pages if page.text.strip() or page.tables]
        document.full_text = "\n\n".join(
            f"{page.text}\n{' '.join(page.tables)}".strip() for page in document.pages if page.text or page.tables
        )
        document.page_count = len(document.pages)
        document.char_count = sum(page.char_count for page in document.pages)
        document.used_ocr = any(page.used_ocr for page in document.pages)
        if document.char_count == 0:
            document.add_warning("No extractable text was found in this document.")
        return document

    # ------------------------------------------------------------------
    @staticmethod
    def detect_first_heading(pages: list[RawPage]) -> str | None:
        for page in pages:
            for line in page.text.splitlines():
                if looks_like_heading(line) and len(line.strip()) > 3:
                    return line.strip().strip("#* ")
        return None


def paginate_text(
    text: str,
    *,
    chars_per_page: int = SYNTHETIC_PAGE_CHARS,
    start_page: int = 1,
) -> list[RawPage]:
    """Split plain text into synthetic pages on blank-line / form-feed boundaries.

    Used by the loaders for formats without real pages (TXT, MD, DOCX, HTML).
    """
    if not text.strip():
        return []

    segments: list[str] = []
    current: list[str] = []
    length = 0
    for block in text.replace("\f", "\n\n").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if length + len(block) > chars_per_page and current:
            segments.append("\n\n".join(current))
            current, length = [], 0
        current.append(block)
        length += len(block) + 2
    if current:
        segments.append("\n\n".join(current))

    return [
        RawPage(page_number=start_page + index, text=segment)
        for index, segment in enumerate(segments)
    ]