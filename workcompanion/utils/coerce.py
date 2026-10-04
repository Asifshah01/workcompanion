"""Tolerant coercion of loosely-typed model output.

LLMs return numbers in shapes Python refuses: ``"3"``, ``"3.0"``, ``"1-5"``,
``"4 out of 5"``, ``"level 2"``, ``"three"``. A single ``int(value)`` in the
middle of an agent is enough to take down a whole page of the app, so every
model-supplied number goes through here instead.

The rule throughout: never raise. A value we cannot understand is ``None`` and
the caller falls back to its own default.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["coerce_int", "coerce_float", "coerce_bool"]

#: The first number anywhere in a string, with thousands separators tolerated.
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

#: Number words, because "three" is a perfectly reasonable answer.
_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_TRUE = {"true", "yes", "y", "1", "on", "correct", "always"}
_FALSE = {"false", "no", "n", "0", "off", "never", "none", "null"}


def coerce_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float from model output. Never raises."""
    if value is None or isinstance(value, bool):
        return default

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return default

    if text in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[text])

    match = _NUMBER_RE.search(text)
    if match is None:
        return default

    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - the regex already guarantees this
        return default


def coerce_int(
    value: Any,
    default: int | None = None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Best-effort int from model output. Never raises.

    ``"1-5"`` yields ``1``; ``"4 out of 5"`` yields ``4``. Values outside
    ``minimum``/``maximum`` return the default rather than being clamped,
    since an out-of-range number usually means the field was not understood.
    """
    number = coerce_float(value)
    if number is None:
        return default

    try:
        number = int(round(number))
    except (ValueError, OverflowError):  # pragma: no cover - inf/nan
        return default

    if minimum is not None and number < minimum:
        return default
    if maximum is not None and number > maximum:
        return default
    return number


def coerce_bool(value: Any, default: bool | None = None) -> bool | None:
    """Best-effort bool from model output. Never raises."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default