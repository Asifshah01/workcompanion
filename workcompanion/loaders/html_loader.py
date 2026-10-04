"""HTML loader - strips markup and keeps a readable text flow."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.loaders.base import BaseLoader, LoadError, paginate_text
from workcompanion.loaders.tables import detect_pipe_table, table_to_markdown
from workcompanion.schemas.documents import ParsedDocument
from workcompanion.utils.text_utils import clean_text

logger = get_logger(__name__)

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg)\b.*?</\1>", re.DOTALL | re.I)
_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.I)
_BLOCK_END = re.compile(
    r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre|br)\s*>|<\br\s*/?>", re.I
)
_HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.DOTALL | re.I)
_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.DOTALL | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS_RUN = re.compile(r"[ \t]{2,}")


class HtmlLoader(BaseLoader):
    """Parse ``.html``/``.htm`` files into text pages."""

    extensions = (".html", ".htm", ".xhtml")
    source_type = "html"

    def load(self, path: str | Path, **kwargs: Any) -> ParsedDocument:
        target = Path(path)
        if not target.is_file():
            raise LoadError(f"HTML file not found: {target}")
        raw = self.load_text(target)

        document = ParsedDocument(
            document_name=target.name,
            source_path=str(target.resolve()),
            source_type="html",
        )
        title_match = _TITLE.search(raw)
        if title_match:
            document.title = clean_text(_TAG.sub("", title_match.group(1)))[:200]

        tables = [
            table_to_markdown(self._table_to_rows(table))
            for table in _TABLE.findall(raw)
        ]
        tables = [table for table in tables if table]

        body = _SCRIPT_STYLE.sub(" ", raw)
        body = _HEADING.sub(lambda m: "\n\n" + "#" * int(m.group(1)) + " " + m.group(2) + "\n\n", body)
        body = _TABLE.sub("\n\n[table removed here]\n\n", body)
        body = _BLOCK_END.sub("\n\n", body)
        body = _TAG.sub(" ", body)
        body = html.unescape(body)
        body = _WS_RUN.sub(" ", body)
        body = re.sub(r"\n{3,}", "\n\n", body)

        pseudo = detect_pipe_table(body)
        if pseudo:
            tables.append(pseudo)

        document.pages = paginate_text(clean_text(body))
        if tables and document.pages:
            document.pages[0].tables.extend(tables[:3])
        document = self.finalize(document)
        logger.info("Parsed HTML '%s': %d pages, %d tables", target.name, document.page_count, len(tables))
        return document

    @staticmethod
    def _table_to_rows(table_html: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_html in re.findall(r"<tr\b.*?</tr>", table_html, re.DOTALL | re.I):
            cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.I)
            if cells:
                rows.append([html.unescape(_TAG.sub(" ", cell)).strip() for cell in cells])
        return rows