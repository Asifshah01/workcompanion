"""Text normalisation, segmentation and light linguistic utilities.

These functions run on every ingestion / retrieval path, so they are written to
be dependency-free, fast and deterministic.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "clean_text",
    "collapse_whitespace",
    "normalize_whitespace",
    "normalize_query",
    "dehyphenate",
    "split_into_paragraphs",
    "sentences_from_text",
    "looks_like_heading",
    "detect_heading_level",
    "extract_heading_path",
    "extract_key_terms",
    "truncate_words",
    "strip_markdown_noise",
    "similarity_ratio",
    "matches_keywords",
    "compile_keyword_pattern",
]

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_SOFT_HYPHEN = re.compile(r"(\w)-\n(\w)")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*\u2022\u25cf\u25aa\u00b7]|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'\u201c])")
_WORD_RE = re.compile(r"(?P<word>[A-Za-z][A-Za-z'\-]*)|(?P<weight>\d+(?:\.\d+)?)")
_MD_NOISE = re.compile(r"(!\[[^\]]*\]\([^)]*\))|(?<!\w)!\[|\[([^\]]+)\]\([^)]*\)")

_STOPWORDS = frozenset(
    """
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he her here hers herself him himself his how i if in into is isn't it
its itself let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she should shouldn't so some such than
that the their theirs them themselves then there these they this those through to too
under until up very was wasn't we were weren't what when where which while who whom why
with won't would wouldn't you your yours yourself yourselves explain tell me about
please give describe what does mean do does is are the this that of and in to how why
""".split()
)

_ACRONYMS = frozenset(
    "pdf csv pptx docx html md json api llm rag sql http https url cpu gpu ram ml ai "
    "id ok eg ie etc".split()
)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and limit blank lines to at most two."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    """Collapse *all* whitespace (including newlines) into single spaces."""
    return _MULTI_SPACE.sub(" ", normalize_whitespace(text).replace("\n", " "))


def dehyphenate(text: str) -> str:
    """Re-join words split across a line break by a hyphen (``thermo-\\ndynamic``)."""
    return _SOFT_HYPHEN.sub(r"\1\2", text)


def clean_text(text: str) -> str:
    """Full cleaning pass used by the ingestion pipeline."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = dehyphenate(text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_markdown_noise(text: str) -> str:
    """Drop image syntax and inline link targets while keeping link text."""
    return _MD_NOISE.sub(lambda m: m.group(2) or "", text)


def normalize_query(query: str) -> str:
    """Normalise a natural-language question for retrieval."""
    q = clean_text(query or "")
    q = strip_markdown_noise(q)
    q = re.sub(r"\s+", " ", q)
    return q.strip().strip("?").strip()


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def split_into_paragraphs(text: str) -> list[str]:
    """Split normalised text into non-empty paragraphs."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n|(?<!\n)\n(?=\s*[A-Z0-9])", normalize_whitespace(text))
    return [p.strip() for p in parts if p and p.strip()]


def sentences_from_text(text: str, *, min_length: int = 25) -> list[str]:
    """Split text into sentences, keeping a conservative minimum length.

    Abbreviation-aware enough for lecture notes: splits on sentence-ending
    punctuation followed by a capitalised token.
    """
    if not text:
        return []
    text = normalize_whitespace(text).replace("\n", " ")
    raw = _SENTENCE_SPLIT.split(text)
    sentences: list[str] = []
    buffer = ""
    for part in raw:
        candidate = f"{buffer} {part}".strip() if buffer else part.strip()
        if not candidate:
            continue
        # Merge fragments that are clearly too small to be standalone sentences.
        if len(candidate) < min_length and not re.search(r"[.!?]$", candidate):
            buffer = candidate
            continue
        buffer = ""
        if len(candidate) >= min_length or sentences:
            sentences.append(candidate.strip())
        else:
            buffer = candidate
    if buffer:
        sentences.append(buffer.strip())
    return [s for s in sentences if s]


def truncate_words(text: str, limit: int, suffix: str = "\u2026") -> str:
    """Truncate to ``limit`` words, appending an ellipsis when shortened."""
    words = (text or "").split()
    if len(words) <= limit:
        return text or ""
    return " ".join(words[:limit]) + suffix


# ---------------------------------------------------------------------------
# Structural detection
# ---------------------------------------------------------------------------
def looks_like_heading(line: str) -> bool:
    """Heuristically decide whether a single line is a structural heading."""
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", ",", ";")) and not re.match(r"^\d+(\.\d+)*\s", stripped):
        return False

    # Markdown / numbered / "Chapter 3" / "ALL CAPS" / Title Case short lines.
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    if re.match(r"^(?:chapter|section|unit|module|part|lecture|topic|appendix)\b", stripped, re.I):
        return True
    if re.match(r"^\d+(?:\.\d+){0,3}[\.\)]?\s+\S", stripped):
        return True
    if re.match(r"^(?:[IVXLC]+)[\.\)]\s+\S", stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(stripped.split()) <= 12:
        return True
    if stripped.startswith(("**", "__")) and stripped.endswith(("**", "__")):
        return True
    # A short Title Case line followed by body text reads as a heading.
    if len(stripped.split()) <= 9 and stripped[0].isupper() and not stripped.endswith(":"):
        return True
    return False


def detect_heading_level(line: str) -> int:
    """Return an approximate heading depth (1 = most significant)."""
    stripped = (line or "").strip()
    if match := re.match(r"^(#{1,6})\s", stripped):
        return len(match.group(1))
    if match := re.match(r"^(\d+(?:\.\d+){0,3})[\.\)]?\s", stripped):
        return min(len(match.group(1).split(".")) + 1, 6)
    if re.match(r"^(?:chapter|part|unit)\b", stripped, re.I):
        return 1
    if re.match(r"^(?:section|module|lecture|topic)\b", stripped, re.I):
        return 2
    return 2


def extract_heading_path(heading_stack: Iterable[tuple[int, str]]) -> str:
    """Join a heading stack into a breadcrumb such as ``Chapter 3 > Entropy``."""
    return " > ".join(title for _, title in heading_stack if title)


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------
def extract_key_terms(text: str, *, limit: int = 12, drop_stopwords: bool = True) -> list[str]:
    """Frequency-ranked content words, acronyms and numbers."""
    counts: dict[str, float] = {}
    order: dict[str, int] = {}
    for position, match in enumerate(_WORD_RE.finditer(text or "")):
        word = match.group("word")
        # The weight group is a decimal string ("1", "1.00", "2.5"), never an int.
        try:
            weight = float(match.group("weight") or 1)
        except ValueError:  # pragma: no cover - regex guarantees a number
            weight = 1.0
        if not word or len(word) < 2:
            continue
        lowered = word.lower()
        if drop_stopwords and lowered in _STOPWORDS and lowered not in _WORD_NUMBERS:
            continue
        if lowered.isdigit() and int(lowered) in range(0, 11):
            continue
        key = lowered
        counts[key] = counts.get(key, 0) + weight
        order.setdefault(key, position)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], order[item[0]]))
    return [word for word, _ in ranked[:limit]]


# ---------------------------------------------------------------------------
# Keyword matching (intent routing)
# ---------------------------------------------------------------------------
_KEYWORD_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def compile_keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    """Compile a case-insensitive alternation of keywords with word boundaries.

    Plain ``in`` checks are unsafe for short keywords: ``"hi" in "this"`` and
    ``"vs" in "versus"`` are both true.  Each keyword is therefore fenced with
    ``(?<![a-z0-9])`` / ``(?![a-z0-9])`` so it only matches as a whole word
    (leading/trailing spaces and punctuation are still allowed).
    """
    key = tuple(keywords)
    cached = _KEYWORD_CACHE.get(key)
    if cached is not None:
        return cached
    parts = []
    for keyword in key:
        bare = keyword.strip().lower()
        escaped = re.escape(bare).replace(r"\ ", r"\s+")
        # Long keywords also match their plural ("flashcard" -> "flashcards").
        # Short ones must not ("hi" would otherwise swallow "his"/"him").
        plural = r"(?:e?s)?" if len(bare) >= 4 else ""
        parts.append(rf"(?<![a-z0-9]){escaped}{plural}(?![a-z0-9])")
    pattern = re.compile("|".join(parts) if parts else r"(?!)")
    _KEYWORD_CACHE[key] = pattern
    return pattern


def matches_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Whole-word keyword test used by the deterministic router."""
    return bool(keywords) and compile_keyword_pattern(keywords).search((text or "").lower()) is not None


def similarity_ratio(left: str, right: str) -> float:
    """Token-set Jaccard similarity - cheap lexical overlap measure."""
    left_tokens = {t for t in _WORD_RE.findall((left or "").lower()) if t not in _STOPWORDS}
    right_tokens = {t for t in _WORD_RE.findall((right or "").lower()) if t not in _STOPWORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)