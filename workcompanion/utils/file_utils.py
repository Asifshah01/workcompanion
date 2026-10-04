"""Upload validation, filename sanitisation and file metadata helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FileValidationError",
    "SafeNameError",
    "sanitize_filename",
    "validate_upload",
    "validate_extension",
    "human_size",
    "describe_file",
    "file_sha256",
    "unique_path",
]

_UNSAFE_CHARS = re.compile(r"[^\w.\- ]+", re.UNICODE)
_MULTI_DOT = re.compile(r"\.{2,}")
_RESERVED_WINDOWS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


class SafeNameError(FileValidationError):
    """Raised when a filename cannot be made safe."""


def sanitize_filename(name: str, *, max_length: int = 120, fallback: str = "document") -> str:
    """Return a filesystem-safe filename preserving the extension.

    Handles path traversal, reserved Windows device names, control characters
    and length limits.
    """
    if not name:
        return fallback
    raw = unicodedata.normalize("NFKD", str(name)).replace("\u0000", "")
    raw = raw.replace("\\", "/").split("/")[-1]  # strip any directory component
    raw = raw.strip().strip(".") or fallback

    stem, dot, extension = raw.rpartition(".")
    if not dot:
        stem, extension = raw, ""
    stem = _UNSAFE_CHARS.sub("_", stem).strip(" ._-") or fallback
    extension = _UNSAFE_CHARS.sub("", extension).strip(".")

    if stem.upper() in _RESERVED_WINDOWS:
        stem = f"{stem}_file"
    if extension:
        stem = f"{stem}.{extension}"
    if len(stem) > max_length:
        keep = max_length - (len(extension) + 1 if extension else 0)
        stem = stem[: max(keep, 8)] + (f".{extension}" if extension else "")
    return stem


def validate_extension(filename: str, allowed: tuple[str, ...]) -> str:
    """Validate the extension against an allow-list and return it lowercased."""
    extension = Path(filename).suffix.lower()
    if not extension:
        raise FileValidationError(f"'{filename}' has no file extension.")
    if extension not in allowed:
        raise FileValidationError(
            f"Unsupported file type '{extension}'. Allowed types: {', '.join(sorted(allowed))}"
        )
    return extension


def validate_upload(
    filename: str,
    size_bytes: int,
    *,
    allowed_extensions: tuple[str, ...],
    max_mb: int,
) -> tuple[str, str]:
    """Validate an upload and return ``(safe_filename, extension)``."""
    safe_name = sanitize_filename(filename)
    extension = validate_extension(safe_name, allowed_extensions)
    if size_bytes <= 0:
        raise FileValidationError(f"'{safe_name}' is empty.")
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileValidationError(
            f"'{safe_name}' is {human_size(size_bytes)} which exceeds the {max_mb} MB limit."
        )
    return safe_name, extension


def human_size(num_bytes: float) -> str:
    """Format a byte count for humans."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file (used for duplicate detection)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_path(directory: str | Path, filename: str) -> Path:
    """Resolve ``filename`` inside ``directory``, adding ``_1``, ``_2`` ... on clash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileValidationError(f"Could not find a free filename for '{filename}'.")


@dataclass(slots=True, frozen=True)
class FileInfo:
    """Static metadata about a stored file."""

    filename: str
    size_bytes: int
    extension: str
    path: str
    sha256: str | None = None

    @property
    def size_display(self) -> str:
        return human_size(self.size_bytes)


def describe_file(path: str | Path, *, with_hash: bool = True) -> FileInfo:
    """Build a :class:`FileInfo` for a file on disk."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileValidationError(f"File not found: {resolved}")
    return FileInfo(
        filename=resolved.name,
        size_bytes=resolved.stat().st_size,
        extension=resolved.suffix.lower(),
        path=str(resolved),
        sha256=file_sha256(resolved) if with_hash else None,
    )