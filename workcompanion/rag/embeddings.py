"""Embedding provider abstraction.

The RAG pipeline only talks to :class:`EmbeddingProvider`.  Two implementations
ship:

* :class:`SentenceTransformerEmbeddingProvider` - local open-source model
  (``all-MiniLM-L6-v2`` by default).  High quality, downloads once.
* :class:`HashingEmbeddingProvider` - dependency-free hashed character n-gram +
  word vector with sublinear term weighting.  Instant, deterministic, no
  download; good enough for lexical/semantic-ish matching and used for tests
  and as the automatic fallback.

Embeddings are always L2-normalised so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import abc
import hashlib
import math
import threading
from collections.abc import Sequence

import numpy as np

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.utils.cache import DiskCache, cache_key
from workcompanion.utils.text_utils import extract_key_terms

logger = get_logger(__name__)

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "HashingEmbeddingProvider",
    "get_embedding_provider",
    "cosine_similarity",
    "l2_normalize",
]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with zero-row protection."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array if norm == 0 else array / norm
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two batches of (already normalised) vectors."""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    return left @ right.T


class EmbeddingProvider(abc.ABC):
    """Interface every embedding backend implements."""

    name: str = "base"

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Embedding width."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @abc.abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents into a ``(n, dim)`` float32 array."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, returned as a 1-D float32 array."""
        return self.embed_documents([text])[0]

    def embed_with_cache(self, texts: Sequence[str], *, cache: DiskCache | None = None) -> np.ndarray:
        """Embed documents, consulting a disk cache per item.

        Only the cache misses are sent to the model, which keeps re-indexing a
        single new file cheap.
        """
        if cache is None or not cache.enabled:
            return self.embed_documents(texts)

        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        keys = [cache_key("emb", self.model_name, text) for text in texts]

        for index, key in enumerate(keys):
            cached = cache.get(key)
            if cached is not None:
                try:
                    vectors[index] = np.asarray(cached, dtype=np.float32)
                    continue
                except (TypeError, ValueError):
                    vectors[index] = None
            missing.append(index)

        if missing:
            fresh = self.embed_documents([texts[i] for i in missing])
            for position, index in enumerate(missing):
                vectors[index] = fresh[position]
                cache.set(keys[index], fresh[position].tolist())

        return l2_normalize(np.vstack([vector for vector in vectors if vector is not None]))

    def describe(self) -> dict[str, object]:
        return {"provider": self.name, "model": self.model_name, "dimension": self.dimension}


# ---------------------------------------------------------------------------
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers backend (lazy import)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, *, batch_size: int = 64):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dimension = 0
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s'...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
class HashingEmbeddingProvider(EmbeddingProvider):
    """Dependency-free hashed embedding (word + character n-gram hashing).

    Terms are hashed into a fixed-width vector with sublinear term frequency
    weighting and IDF-style down-weighting of stopwords.  Not a transformer, but
    genuinely more than bag-of-words: character trigrams give it robustness to
    morphology and typos, which matters for STEM terminology.
    """

    name = "hashing"

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._word_regex = __import__("re").compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}d"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> dict[int, float]:
        lowered = (text or "").lower()
        features: dict[int, float] = {}
        counts: dict[str, int] = {}

        for token in self._word_regex.findall(lowered):
            counts[token] = counts.get(token, 0) + 1
            padded = f"<{token}>"
            for size in (3, 4, 5):
                for start in range(max(1, len(padded) - size + 1)):
                    gram = padded[start : start + size]
                    index = _stable_hash(gram) % self._dimension
                    features[index] = features.get(index, 0.0) + 0.35

        key_terms = set(extract_key_terms(lowered, limit=8))
        for token, count in counts.items():
            weight = 1.0 + math.log(count)
            if token in key_terms:
                weight *= 1.6
            index = _stable_hash(token) % self._dimension
            features[index] = features.get(index, 0.0) + weight
            bigram_index = _stable_hash(f"{token}~stem") % self._dimension
            features[bigram_index] = features.get(bigram_index, 0.0) + weight * 0.4
        return features

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for index, value in self._features(text).items():
                matrix[row, index] = value
        return l2_normalize(matrix)


def _stable_hash(token: str) -> int:
    """Deterministic hash (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
_PROVIDERS: dict[str, EmbeddingProvider] = {}


def _build(settings: Settings) -> EmbeddingProvider:
    choice = settings.embedding_provider
    if choice == "hashing":
        return HashingEmbeddingProvider()
    if choice == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )

    # "auto": prefer the real model, fall back if unavailable (e.g. offline).
    try:
        import sentence_transformers  # noqa: F401

        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )
        provider.dimension  # force the load so failures surface here
        return provider
    except Exception as exc:
        logger.warning(
            "Embedding model '%s' unavailable (%s); using the built-in hashing embedder.",
            settings.embedding_model, exc,
        )
        return HashingEmbeddingProvider()


def get_embedding_provider(settings: Settings | None = None, *, force: bool = False) -> EmbeddingProvider:
    """Return the configured embedding provider (cached)."""
    settings = settings or get_settings()
    key = f"{settings.embedding_provider}|{settings.embedding_model}"
    if force:
        _PROVIDERS.pop(key, None)
    if key not in _PROVIDERS:
        _PROVIDERS[key] = _build(settings)
    return _PROVIDERS[key]


def reset_embedding_cache() -> None:
    """Drop cached provider instances."""
    _PROVIDERS.clear()