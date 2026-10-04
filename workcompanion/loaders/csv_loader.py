"""CSV / TSV loader - tabular data kept as real tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.loaders.base import BaseLoader, LoadError, paginate_text
from workcompanion.loaders.tables import table_to_markdown
from workcompanion.schemas.documents import ParsedDocument

logger = get_logger(__name__)


class CsvLoader(BaseLoader):
    """Parse ``.csv`` / ``.tsv`` files into markdown tables plus row prose."""

    extensions = (".csv", ".tsv", ".txt.tsv")
    source_type = "csv"

    def load(self, path: str | Path, **kwargs: Any) -> ParsedDocument:
        target = Path(path)
        if not target.is_file():
            raise LoadError(f"CSV not found: {target}")

        raw = self.load_text(target)
        delimiter = "\t" if target.suffix.lower() == ".tsv" else _sniff_delimiter(raw)
        try:
            rows = list(csv.reader(raw.splitlines(), delimiter=delimiter))
        except csv.Error as exc:
            raise LoadError(f"Could not parse '{target.name}' as CSV: {exc}") from exc

        rows = [row for row in rows if any(str(cell).strip() for cell in row)]
        if not rows:
            raise LoadError(f"'{target.name}' contains no rows.")

        document = ParsedDocument(
            document_name=target.name,
            source_path=str(target.resolve()),
            source_type="csv",
        )
        table = table_to_markdown(rows[:200], caption=f"{target.name} (first {min(len(rows), 200)} rows)")
        prose_lines = [f"# Dataset: {target.stem}", "", table, "", "## Sample rows"]
        for row in rows[1:21]:
            if any(str(cell).strip() for cell in row):
                prose_lines.append("- " + ", ".join(str(cell).strip() for cell in row if str(cell).strip()))

        document.pages = paginate_text("\n\n".join(prose_lines))
        document.pages[0].tables.append(table)
        document.metadata["row_count"] = len(rows) - 1
        document.metadata["column_count"] = len(rows[0])
        document = self.finalize(document)
        document.title = target.stem
        logger.info("Parsed CSV '%s': %d rows", target.name, len(rows) - 1)
        return document


def _sniff_delimiter(sample: str) -> str:
    """Guess the delimiter from the header line."""
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    try:
        return csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        return "," if first_line.count(",") >= first_line.count(";") else ";"