"""Shared helpers used across ingestion, RAG, agents and the UI."""

from workcompanion.utils.text_utils import (
    clean_text,
    collapse_whitespace,
    detect_heading_level,
    extract_key_terms,
    looks_like_heading,
    normalize_query,
    normalize_whitespace,
    sentences_from_text,
    split_into_paragraphs,
    truncate_words,
)
from workcompanion.utils.file_utils import (
    FileValidationError,
    describe_file,
    human_size,
    sanitize_filename,
    validate_upload,
)
from workcompanion.utils.token_utils import estimate_tokens, trim_to_tokens
from workcompanion.utils.cache import DiskCache

__all__ = [
    "clean_text",
    "collapse_whitespace",
    "detect_heading_level",
    "extract_key_terms",
    "looks_like_heading",
    "normalize_query",
    "normalize_whitespace",
    "sentences_from_text",
    "split_into_paragraphs",
    "truncate_words",
    "FileValidationError",
    "describe_file",
    "human_size",
    "sanitize_filename",
    "validate_upload",
    "estimate_tokens",
    "trim_to_tokens",
    "DiskCache",
]