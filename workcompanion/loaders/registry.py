"""Loader registry - maps a file extension to the right loader."""

from __future__ import annotations

from pathlib import Path

from workcompanion.config.logging_config import get_logger
from workcompanion.loaders.base import BaseLoader, LoadError
from workcompanion.loaders.csv_loader import CsvLoader
from workcompanion.loaders.docx_loader import DocxLoader
from workcompanion.loaders.html_loader import HtmlLoader
from workcompanion.loaders.pdf_loader import PDFLoader
from workcompanion.loaders.pptx_loader import PptxLoader
from workcompanion.loaders.txt_loader import MarkdownLoader, TextLoader

logger = get_logger(__name__)

_LOADER_CLASSES: tuple[type[BaseLoader], ...] = (
    PDFLoader,
    DocxLoader,
    PptxLoader,
    MarkdownLoader,
    TextLoader,
    CsvLoader,
    HtmlLoader,
)

_INSTANCES: dict[str, BaseLoader] = {}


def supported_extensions() -> tuple[str, ...]:
    """All extensions the ingestion pipeline can handle."""
    extensions: list[str] = []
    for loader_class in _LOADER_CLASSES:
        extensions.extend(loader_class.extensions)
    return tuple(sorted(set(extensions)))


def is_supported(path: str | Path) -> bool:
    """Whether a path can be parsed by any registered loader."""
    return Path(path).suffix.lower() in supported_extensions()


def get_loader(path: str | Path) -> BaseLoader:
    """Return a loader instance able to parse ``path``.

    Raises:
        LoadError: when no loader handles the extension.
    """
    extension = Path(path).suffix.lower()
    if extension in _INSTANCES:
        return _INSTANCES[extension]
    for loader_class in _LOADER_CLASSES:
        if extension in loader_class.extensions:
            instance = loader_class()
            _INSTANCES[extension] = instance
            return instance
    raise LoadError(
        f"No loader for '{extension}'. Supported types: {', '.join(supported_extensions())}"
    )


def reset_loaders() -> None:
    """Drop cached loader instances (tests / settings changes)."""
    _INSTANCES.clear()