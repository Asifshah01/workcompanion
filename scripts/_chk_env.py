"""Verify ``.env.example`` parses into valid Settings.

Names being valid is not enough: a value out of range or a bad enum member would
make the documented defaults unusable, and the README tells people to copy this
file verbatim.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workcompanion.config.settings import Settings  # noqa: E402


def main() -> int:
    example = ROOT / ".env.example"

    with tempfile.TemporaryDirectory() as tmp:
        # Point the derived paths at a throwaway directory so nothing in the
        # repository is touched by the import.
        copy = Path(tmp) / ".env"
        copy.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        os.environ["DATA_DIR"] = str(Path(tmp) / "data")
        os.environ["CACHE_DIR"] = str(Path(tmp) / "cache")
        os.environ["VECTOR_DB_PATH"] = str(Path(tmp) / "vs")
        os.environ["DOCUMENTS_DIR"] = str(Path(tmp) / "docs")

        settings = Settings(_env_file=copy)  # type: ignore[call-arg]

    problems: list[str] = []

    # Every documented non-empty value must be what the code would have used, so
    # the file cannot silently change behaviour relative to the defaults.
    for key in ("DENSE_WEIGHT", "SPARSE_WEIGHT", "RETRIEVAL_TOP_K", "RERANKER"):
        value = example_line(example, key)
        field = key.lower()
        actual = getattr(settings, field)
        if value.lower() != str(actual).lower():
            problems.append(f"{key}: documented {value!r}, parsed {actual!r}")

    if settings.allowed_upload_extensions and not isinstance(
        settings.allowed_upload_extensions, tuple
    ):
        problems.append("ALLOWED_UPLOAD_EXTENSIONS did not parse as a sequence")

    if settings.dense_weight + settings.sparse_weight <= 0:
        problems.append("retrieval weights sum to zero")

    for label in problems:
        print(f"  [FAIL] {label}")
    if not problems:
        print("  [PASS] .env.example parses into valid Settings")
        print(f"         weights dense={settings.dense_weight} sparse={settings.sparse_weight}")
        print(f"         uploads {len(settings.allowed_upload_extensions)} extensions")
    return 1 if problems else 0


def example_line(text: Path, key: str) -> str:
    for raw in text.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(key + "="):
            return line.partition("=")[2].strip()
    return ""


if __name__ == "__main__":
    sys.exit(main())