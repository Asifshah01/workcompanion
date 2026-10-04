"""PDF loader.

Strategy:

1. Extract text + metadata with ``pypdf`` (always available, fast).
2. Extract real table structures with ``pdfplumber`` when installed.
3. Detect whitespace-aligned pseudo tables in the text layer.
4. If a page yields too little text (scanned page), fall back to OCR - only
   when OCR is enabled *and* an engine is actually installed.

PDF outlines (bookmarks) are mapped onto chapter/section metadata when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.loaders.base import BaseLoader, LoadError
from workcompanion.loaders.tables import detect_pipe_table, table_to_markdown
from workcompanion.schemas.documents import ParsedDocument, RawPage
from workcompanion.utils.text_utils import clean_text, collapse_whitespace

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """Parse ``.pdf`` files into pages of text plus markdown tables."""

    extensions = (".pdf",)
    source_type = "pdf"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    def load(self, path: str | Path, **kwargs: Any) -> ParsedDocument:
        target = Path(path)
        if not target.is_file():
            raise LoadError(f"PDF not found: {target}")
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LoadError("pypdf is not installed. Run: pip install -r requirements.txt") from exc

        document = ParsedDocument(
            document_name=target.name,
            source_path=str(target.resolve()),
            source_type="pdf",
        )

        try:
            reader = PdfReader(str(target))
        except Exception as exc:
            raise LoadError(f"Could not open '{target.name}' as a PDF: {exc}") from exc

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:  # pragma: no cover - depends on the file
                document.add_warning("This PDF is encrypted; text extraction may be incomplete.")

        self._read_metadata(reader, document)
        outline = self._read_outline(reader)

        pdfplumber = self._try_pdfplumber(target)
        plumber_pages = self._extract_plumber_tables(pdfplumber) if pdfplumber else {}

        pages: list[RawPage] = []
        ocr_pages = 0
        for index, pdf_page in enumerate(reader.pages):
            try:
                raw_text = pdf_page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - malformed page
                logger.debug("Text extraction failed on page %d: %s", index + 1, exc)
                raw_text = ""

            tables = plumber_pages.get(index + 1, [])
            if not tables:
                pseudo = detect_pipe_table(raw_text)
                tables = [pseudo] if pseudo else []

            page = RawPage(page_number=index + 1, text=raw_text, tables=tables)

            if self._needs_ocr(page) and ocr_pages < self._settings.ocr_max_pages:
                ocr_pages += 1
                page = self._apply_ocr(target, index, page, document)

            heading = outline.get(index + 1)
            if heading:
                page.heading_hint = heading
            pages.append(page)

        if not pages:
            raise LoadError(f"'{target.name}' contains no readable pages.")

        document.pages = pages
        document = self.finalize(document)
        if outline:
            document.metadata["outline_entries"] = len(outline)
        logger.info(
            "Parsed PDF '%s': %d pages, %d chars, tables=%d, ocr_pages=%d",
            target.name, document.page_count, document.char_count,
            len(document.tables), sum(1 for p in document.pages if p.used_ocr),
        )
        return document

    # ------------------------------------------------------------------
    def _read_metadata(self, reader: Any, document: ParsedDocument) -> None:
        try:
            info = reader.metadata or {}
        except Exception:  # pragma: no cover
            info = {}
        title = info.get("/Title") if hasattr(info, "get") else None
        author = info.get("/Author") if hasattr(info, "get") else None
        subject = info.get("/Subject") if hasattr(info, "get") else None
        if isinstance(title, str) and clean_text(title) and clean_text(title).lower() != "untitled":
            document.title = clean_text(title)
        if isinstance(author, str) and clean_text(author):
            document.author = clean_text(author)
        if isinstance(subject, str) and clean_text(subject):
            document.subject = clean_text(subject)
        if not document.title:
            document.title = self._guess_title(reader)

    @staticmethod
    def _guess_title(reader: Any) -> str | None:
        """Use the first non-empty line of page 1 as a title candidate."""
        try:
            first = reader.pages[0].extract_text() or ""
        except Exception:  # pragma: no cover
            return None
        for line in first.splitlines():
            candidate = collapse_whitespace(line)
            if len(candidate) > 4:
                return candidate[:180]
        return None

    @staticmethod
    def _read_outline(reader: Any) -> dict[int, str]:
        """Map page numbers to their nearest outline (bookmark) title."""
        mapping: dict[int, str] = {}
        try:
            outline = reader.outline
        except Exception:
            return mapping
        if not outline:
            return mapping

        flat: list[Any] = []

        def walk(items: Any) -> None:
            for item in items:
                if isinstance(item, list):
                    walk(item)
                else:
                    flat.append(item)

        try:
            walk(outline)
        except Exception:  # pragma: no cover
            return mapping

        for item in flat:
            try:
                page_number = reader.get_destination_page_number(item) + 1
                title = clean_text(getattr(item, "title", "")) or None
                if title:
                    mapping.setdefault(page_number, title[:180])
            except Exception:  # pragma: no cover
                continue
        return mapping

    # ------------------------------------------------------------------
    def _try_pdfplumber(self, path: Path) -> Any | None:
        try:
            import pdfplumber
        except ImportError:
            logger.info("pdfplumber unavailable - falling back to text-based table detection.")
            return None
        try:
            return pdfplumber.open(str(path))
        except Exception as exc:  # pragma: no cover - malformed pdf
            logger.warning("pdfplumber could not open '%s': %s", path.name, exc)
            return None

    @staticmethod
    def _extract_plumber_tables(plumber: Any) -> dict[int, list[str]]:
        tables: dict[int, list[str]] = {}
        try:
            for page_number, page in enumerate(plumber.pages, start=1):
                try:
                    extracted = page.extract_tables() or []
                except Exception:  # pragma: no cover
                    extracted = []
                rendered = [
                    table_to_markdown(rows, caption=None)
                    for rows in extracted
                    if rows and len(rows) >= 2
                ]
                rendered = [table for table in rendered if table]
                if rendered:
                    tables[page_number] = rendered
        except Exception as exc:  # pragma: no cover
            logger.debug("Table extraction failed: %s", exc)
        finally:
            try:
                plumber.close()
            except Exception:
                pass
        return tables

    # ------------------------------------------------------------------
    def _needs_ocr(self, page: RawPage) -> bool:
        if not self._settings.enable_ocr_fallback or self._settings.ocr_engine == "none":
            return False
        text_chars = len(page.text.strip())
        if page.tables:
            text_chars += sum(len(t) for t in page.tables)
        return text_chars < self._settings.ocr_min_chars_per_page

    def _apply_ocr(
        self, path: Path, index: int, page: RawPage, document: ParsedDocument
    ) -> RawPage:
        from workcompanion.loaders.ocr import ocr_available, ocr_pdf_page

        if not ocr_available():
            document.add_warning(
                f"Page {index + 1} looks scanned but no OCR engine (Tesseract) is installed; "
                "its text is missing. Install Tesseract + pdf2image to enable the OCR fallback."
            )
            return page

        result = ocr_pdf_page(path, index)
        if result.succeeded:
            logger.info("OCR recovered %d chars from page %d", len(result.text), index + 1)
            page.text = f"{page.text}\n{result.text}".strip()
            page.used_ocr = True
        else:
            document.add_warning(f"OCR failed on page {index + 1}: {result.message}")
        return page