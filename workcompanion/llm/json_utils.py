"""Tolerant JSON extraction from LLM output.

Models frequently wrap JSON in prose or code fences, emit trailing commas, or use
smart quotes.  Rather than failing an agent call, we normalise the text and
attempt a series of increasingly forgiving parses.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_json", "strip_code_fences", "balance_json", "coerce_to_schema"]

_FENCE_RE = re.compile(r"```(?:json|JSON|python)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", " ": " "})

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}", "]"}


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the original text."""
    match = _FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def balance_json(text: str) -> str:
    """Close unbalanced brackets/braces in a JSON fragment.

    Models frequently truncate long documents; balancing lets us recover the
    complete portion instead of discarding the whole response.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in _CLOSERS:
            if stack and stack[-1] == char:
                stack.pop()
    repaired = text
    if in_string:
        repaired += '"'
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    for closer in reversed(stack):
        repaired += closer
    return repaired


def _candidate_spans(text: str) -> list[str]:
    """All plausible JSON substrings, most likely first."""
    stripped = strip_code_fences(text)
    spans = [stripped] if stripped else []
    for match in _FENCE_RE.finditer(text or ""):
        spans.append(match.group(1).strip())

    # Largest balanced object/array starting anywhere in the text.
    for opener in ("{", "["):
        start = stripped.find(opener)
        while start != -1:
            for end in range(len(stripped), start, -1):
                candidate = stripped[start:end]
                if candidate and candidate[-1] in _CLOSERS:
                    spans.append(candidate)
                    break
            start = stripped.find(opener, start + 1)

    # Balanced-scan extraction (handles trailing prose after the document).
    for start_char, closer in _OPENERS.items():
        index = 0
        while True:
            index = stripped.find(start_char, index)
            if index == -1:
                break
            depth = 0
            in_string = False
            escaped = False
            for position in range(index, len(stripped)):
                char = stripped[position]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == start_char:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        spans.append(stripped[index : position + 1])
                        break
            index += 1

    seen: set[str] = set()
    unique: list[str] = []
    for span in spans:
        cleaned = span.translate(_SMART_QUOTES).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of the first JSON document inside ``text``."""
    if not text:
        return None

    for candidate in _candidate_spans(text):
        for attempt in (candidate, balance_json(candidate)):
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, (dict, list)):
                return parsed

    try:  # optional dependency, handles trailing commas / unquoted keys
        import json_repair  # type: ignore

        repaired = json_repair.loads(strip_code_fences(text))
        if isinstance(repaired, (dict, list)) and repaired:
            return repaired
    except Exception:
        pass

    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        return None
    return None


def coerce_to_schema(payload: Any, model: type) -> Any:
    """Validate ``payload`` against ``model``, returning ``None`` on failure."""
    if payload is None:
        return None
    try:
        return model.model_validate(payload)
    except Exception:
        return None