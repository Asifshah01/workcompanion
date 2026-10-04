"""Optional OCR support for scanned PDFs.

Tesseract (``pytesseract`` + ``pdf2image``/Poppler) is entirely optional: the
pipeline only reaches for it when a page yields too little text.  Every import
is lazy and failures degrade to a clear warning rather than an exception.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from workcompanion.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class OCRResult:
    """Outcome of an OCR attempt on a single page."""

    text: str = ""
    succeeded: bool = False
    engine: str = "none"
    message: str = ""


def ocr_available() -> bool:
    """Whether an OCR engine can actually be used on this machine."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_pdf_page(pdf_path: str | Path, page_index: int, *, dpi: int = 200) -> OCRResult:
    """OCR a single zero-based page of a PDF.

    Returns a failed :class:`OCRResult` (rather than raising) when OCR is not
    installed or the page cannot be processed.
    """
    path = Path(pdf_path)
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return OCRResult(message="pytesseract/Pillow not installed - skipping OCR")

    try:
        from pdf2image import convert_from_path
    except ImportError:
        return OCRResult(message="pdf2image not installed - skipping OCR")

    try:
        images = convert_from_path(str(path), dpi=dpi, first_page=page_index + 1, last_page=page_index + 1)
    except Exception as exc:  # pragma: no cover - depends on Poppler
        return OCRResult(message=f"PDF rasterisation failed: {exc}")

    if not images:
        return OCRResult(message="No image produced for the page")

    text_parts: list[str] = []
    try:
        for image in images:
            text_parts.append(pytesseract.image_to_string(Image.open(image), lang="eng"))
    except Exception as exc:  # pragma: no cover - depends on Tesseract
        return OCRResult(message=f"Tesseract failed: {exc}")

    text = "\n".join(part.strip() for part in text_parts if part.strip())
    return OCRResult(
        text=text,
        succeeded=bool(text.strip()),
        engine="tesseract",
        message="" if text.strip() else "OCR produced no text for this page",
    )


def ocr_image_file(path: str | Path) -> OCRResult:
    """OCR a standalone image file."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return OCRResult(message="pytesseract/Pillow not installed - skipping OCR")
    try:
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image, lang="eng")
    except Exception as exc:  # pragma: no cover
        return OCRResult(message=f"Tesseract failed: {exc}")
    cleaned = text.strip()
    return OCRResult(text=cleaned, succeeded=bool(cleaned), engine="tesseract")