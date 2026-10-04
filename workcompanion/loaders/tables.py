"""Table detection and structure-preserving serialisation.

Tables are **not** flattened into meaningless running text.  They are rendered
as Markdown pipe tables, which keeps the row/column relationships readable to
the LLM while remaining fully searchable and embeddable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["table_to_markdown", "detect_pipe_table", "sanitize_cell", "tables_to_chunks"]

_CELL_SPLIT = re.compile(r"\s{2,}|\s*\|\s*|\t+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def sanitize_cell(value: object, *, max_length: int = 220) -> str:
    """Normalise a single table cell."""
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    text = _MULTI_SPACE.sub(" ", text).strip()
    if len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "\u2026"
    return text


def _infer_header(row: Sequence[object]) -> list[str]:
    cells = [sanitize_cell(value) for value in row]
    numeric = sum(1 for value in cells if value and re.fullmatch(r"[-+]?[\d.,%$ ]+", value))
    has_header_keyword = any(
        re.search(r"\b(name|value|unit|total|item|type|description|quantity|result|method|year|rank|amount)\b", value, re.I)
        for value in cells
    )
    if len(cells) >= 2 and numeric >= max(1, len(cells) - 2) and not has_header_keyword:
        return [f"Col {index + 1}" for index in range(len(cells))]
    return cells or [f"Col {index + 1}" for index in range(len(row))]


def table_to_markdown(rows: Sequence[Sequence[object]], *, caption: str | None = None) -> str:
    """Serialise a table (list of rows) into a Markdown pipe table."""
    cleaned_rows = [list(row) for row in rows if row is not None and len([c for c in row if str(c or '').strip()])]
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalised: list[list[str]] = []
    for row in cleaned_rows:
        cells = [sanitize_cell(value) for value in row][:width]
        cells += [""] * (width - len(cells))
        normalised.append(cells)

    header = _infer_header(normalised[0])
    body = normalised[1:]

    lines = []
    if caption:
        lines.append(f"**{caption.strip()}**")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def detect_pipe_table(text: str, *, min_columns: int = 3, min_rows: int = 2) -> str | None:
    """Detect a whitespace/pipe-aligned pseudo table inside plain text.

    Many lecture PDFs render tables as aligned text columns, which the PDF text
    extractor returns without pipes.  This converts such blocks back into
    Markdown tables so their structure survives.
    """
    lines = [line.rstrip() for line in (text or "").splitlines()]
    candidate: list[tuple[int, list[str]]] = []

    def flush() -> str | None:
        nonlocal candidate
        if len(candidate) < min_rows:
            candidate = []
            return None
        widths = {len(cells) for _, cells in candidate}
        if len(widths) != 1 or next(iter(widths)) < min_columns:
            candidate = []
            return None
        table = table_to_markdown([cells for _, cells in candidate])
        candidate = []
        return table or None

    for index, line in enumerate(lines):
        stripped = line.strip()
        is_separator = bool(re.fullmatch(r"[\s|:\-]+", stripped)) and "-" in stripped
        has_delimiter = "|" in stripped or bool(_MULTI_SPACE.search(stripped))
        if stripped and not is_separator and has_delimiter:
            cells = [cell for cell in _CELL_SPLIT.split(stripped) if cell.strip()]
            if len(cells) >= min_columns:
                candidate.append((index, cells))
                continue
        result = flush()
        if result:
            return result
        if not stripped:
            continue
        candidate = []
    return flush()


def tables_to_chunks(tables: Sequence[str]) -> list[str]:
    """Normalise a collection of markdown tables for storage."""
    return [table.strip() for table in tables if table and table.strip()]