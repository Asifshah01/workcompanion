"""Sparse (BM25) retrieval index.

Complements dense vector search: BM25 reliably finds exact terminology,
acronyms, equations, numbers and proper nouns that dense embeddings blur -
exactly the things students search for in lecture notes.

The index is persisted as JSON (token lists only) and rebuilt incrementally, so
it always stays in step with the vector store.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.documents import DocumentChunk

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
_MIN_TOKEN_LEN = 2


def tokenize(text: str, *, keep_symbols: bool = True) -> list[str]:
    """Tokenise text for sparse retrieval.

    Short mathematical symbols (``=``, ``<``, ``Δ``) are kept because in STEM
    material "ΔT = 40 K" must remain findable.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        if raw.isalpha():
            lowered = raw.lower()
            if len(lowered) < _MIN_TOKEN_LEN:
                tokens.append(lowered)
                continue
            tokens.append(lowered)
            # Light suffix folding: cheap stemming that helps recall without
            # the risk of a full stemmer mangling technical terms.
            if lowered.endswith("ies") and len(lowered) > 5:
                tokens.append(lowered[:-3] + "y")
            elif lowered.endswith("sses"):
                tokens.append(lowered[:-2])
            elif lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
                tokens.append(lowered[:-1])
            elif lowered.endswith("ing") and len(lowered) > 6:
                tokens.append(lowered[:-3])
        elif raw[0].isdigit():
            tokens.append(raw)
        elif keep_symbols and not raw.isspace():
            tokens.append(raw)
    return tokens


class BM25Index:
    """Self-contained Okapi BM25 index with persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_documents: dict[str, str] = {}
        self._doc_lengths: dict[str, int] = {}
        self._token_frequencies: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._path: Path | None = None

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._doc_tokens)

    @property
    def average_length(self) -> float:
        with self._lock:
            if not self._doc_lengths:
                return 0.0
            return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def contains(self, chunk_id: str) -> bool:
        with self._lock:
            return chunk_id in self._doc_tokens

    # ------------------------------------------------------------------
    def add(
        self,
        chunk_id: str,
        text: str,
        *,
        document_id: str = "",
        boost_terms: Sequence[str] | None = None,
    ) -> None:
        """Index one chunk; re-indexing the same id replaces the previous entry."""
        tokens = tokenize(text)
        if boost_terms:
            # Repeat key terms so headings and phrases carry extra sparse weight.
            tokens = tokens + [str(term).lower() for term in boost_terms for _ in range(2)]
        with self._lock:
            self._remove_locked(chunk_id)
            self._doc_tokens[chunk_id] = tokens
            self._doc_documents[chunk_id] = document_id
            self._doc_lengths[chunk_id] = len(tokens)
            frequencies = Counter(tokens)
            self._token_frequencies[chunk_id] = frequencies
            for token in frequencies:
                self._document_frequency[token] += 1

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        """Index a batch of chunks, boosting headings and extracted key terms."""
        count = 0
        for chunk in chunks:
            heading_terms = chunk.heading_path.split(" > ") if chunk.heading_path else []
            self.add(
                chunk.chunk_id,
                chunk.text,
                document_id=chunk.metadata.document_id,
                boost_terms=[*chunk.key_terms, *heading_terms],
            )
            count += 1
        return count

    def remove(self, chunk_id: str) -> bool:
        with self._lock:
            return self._remove_locked(chunk_id)

    def remove_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document."""
        with self._lock:
            targets = [
                chunk_id for chunk_id, owner in self._doc_documents.items() if owner == document_id
            ]
            for chunk_id in targets:
                self._remove_locked(chunk_id)
            return len(targets)

    def _remove_locked(self, chunk_id: str) -> bool:
        if chunk_id not in self._doc_tokens:
            return False
        for token in self._token_frequencies.pop(chunk_id, Counter()):
            if self._document_frequency[token] <= 1:
                del self._document_frequency[token]
            else:
                self._document_frequency[token] -= 1
        self._doc_tokens.pop(chunk_id, None)
        self._doc_documents.pop(chunk_id, None)
        self._doc_lengths.pop(chunk_id, None)
        return True

    def clear(self) -> None:
        with self._lock:
            self._doc_tokens.clear()
            self._doc_documents.clear()
            self._doc_lengths.clear()
            self._token_frequencies.clear()
            self._document_frequency.clear()

    # ------------------------------------------------------------------
    def search(
        self, query: str, top_k: int = 20, *, document_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """BM25 search returning ``(chunk_id, score)`` sorted by score."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            if not self._doc_tokens:
                return []
            total_documents = len(self._doc_tokens)
            average_length = self.average_length or 1.0
            idf: dict[str, float] = {}
            for token in set(query_tokens):
                frequency = self._document_frequency.get(token, 0)
                idf[token] = math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))

            query_frequency = Counter(query_tokens)
            scored: list[tuple[str, float]] = []
            for chunk_id, frequencies in self._token_frequencies.items():
                if document_ids is not None and self._doc_documents.get(chunk_id) not in document_ids:
                    continue
                length = self._doc_lengths.get(chunk_id, 0) or 1
                score = 0.0
                for token, query_count in query_frequency.items():
                    term_frequency = frequencies.get(token, 0)
                    if not term_frequency:
                        continue
                    denominator = term_frequency + self.k1 * (
                        1 - self.b + self.b * length / average_length
                    )
                    token_weight = idf.get(token, 0.0) * (1 + math.log(query_count))
                    score += token_weight * term_frequency * (self.k1 + 1) / denominator
                if score > 0:
                    scored.append((chunk_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def idf_weights(self, query: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Query terms ordered by IDF weight (diagnostics helper)."""
        with self._lock:
            total = len(self._doc_tokens) or 1
            weights = [
                (
                    token,
                    math.log(
                        1 + (total - self._document_frequency.get(token, 0) + 0.5)
                        / (self._document_frequency.get(token, 0) + 0.5)
                    ),
                )
                for token in set(tokenize(query))
            ]
        weights.sort(key=lambda item: item[1], reverse=True)
        return weights[:top_k]

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the index to a JSON file."""
        target = Path(path)
        with self._lock:
            payload = {
                "k1": self.k1,
                "b": self.b,
                "documents": [
                    {"id": chunk_id, "document_id": self._doc_documents.get(chunk_id, ""), "tokens": tokens}
                    for chunk_id, tokens in self._doc_tokens.items()
                ],
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        self._path = target

    def load(self, path: str | Path) -> bool:
        """Load a persisted index; returns ``False`` when absent or corrupt."""
        source = Path(path)
        if not source.is_file():
            return False
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            entries: list[dict[str, object]] = payload["documents"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not load BM25 index '%s': %s", source, exc)
            return False

        self.clear()
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        with self._lock:
            for entry in entries:
                chunk_id = str(entry.get("id") or "")
                tokens = [str(token) for token in (entry.get("tokens") or [])]
                if not chunk_id:
                    continue
                self._doc_tokens[chunk_id] = tokens
                self._doc_documents[chunk_id] = str(entry.get("document_id") or "")
                self._doc_lengths[chunk_id] = len(tokens)
                frequencies = Counter(tokens)
                self._token_frequencies[chunk_id] = frequencies
                for token in frequencies:
                    self._document_frequency[token] += 1
        self._path = Path(path)
        logger.info("Loaded BM25 index with %d documents from %s", self.size, source)
        return True