"""Document loaders.

Every loader normalises its input into a
:class:`~workcompanion.schemas.documents.ParsedDocument` so downstream
chunking/embedding code never needs to know the source format.
"""

from workcompanion.loaders.base import BaseLoader, LoadError, paginate_text
from workcompanion.loaders.registry import (
    get_loader,
    is_supported,
    supported_extensions,
)

__all__ = [
    "BaseLoader",
    "LoadError",
    "paginate_text",
    "get_loader",
    "is_supported",
    "supported_extensions",
]