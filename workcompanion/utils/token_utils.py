"""Token accounting helpers.

``tiktoken`` is used when installed; otherwise a language-model-agnostic
heuristic (``~4 characters per token``) keeps the budget arithmetic working.
"""

from __future__ import annotations

import functools
import re

from workcompanion.utils.text_utils import normalize_whitespace

__all__ = ["estimate_tokens", "count_tokens", "trim_to_tokens", "token_budget_report"]

_ENCODING: object | None = None
_ENCODING_RESOLVED = False


@functools.lru_cache(maxsize=8)
def _encoding_for(model: str):  # pragma: no cover - depends on optional dependency
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _get_encoding(model: str | None) -> object | None:
    global _ENCODING, _ENCODING_RESOLVED
    if not _ENCODING_RESOLVED:
        _ENCODING_RESOLVED = True
        try:
            _ENCODING = _encoding_for(model or "gpt-4")
        except Exception:
            _ENCODING = None
    return _ENCODING


def count_tokens(text: str, model: str | None = None) -> int:
    """Exact token count when possible, heuristic estimate otherwise."""
    if not text:
        return 0
    encoding = _get_encoding(model)
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]
        except Exception:
            pass
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~4 characters/token)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(words, int(chars / 4) + 1)


def trim_to_tokens(text: str, max_tokens: int, model: str | None = None) -> str:
    """Truncate ``text`` so it fits inside ``max_tokens`` tokens."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text, model) <= max_tokens:
        return text
    encoding = _get_encoding(model)
    if encoding is not None:
        try:
            tokens = encoding.encode(text, disallowed_special=())[:max_tokens]  # type: ignore[attr-defined]
            return encoding.decode(tokens)  # type: ignore[attr-defined]
        except Exception:
            pass
    ratio = max_tokens / max(estimate_tokens(text), 1)
    return text[: int(len(text) * ratio)].rstrip()


def token_budget_report(prompt: str, completion: str = "", model: str | None = None) -> dict[str, int]:
    """Small helper used for cost/observability logging."""
    return {
        "prompt_tokens": count_tokens(prompt, model),
        "completion_tokens": count_tokens(completion, model),
        "total_tokens": count_tokens(prompt, model) + count_tokens(completion, model),
    }