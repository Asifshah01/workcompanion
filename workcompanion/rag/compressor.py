"""Context compression.

Sending twenty full chunks to the LLM wastes tokens and dilutes the answer.
The compressor keeps only the sentences within each surviving chunk that
actually bear on the question, always retaining the chunk's heading, page and
source line so citations stay verifiable.

Extractive (not abstractive) by default: it is free, fast and cannot invent
content - which matters for hallucination control.  An optional LLM mode exists
for hard cases.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMProvider, LLMRequest, SystemMessage, UserMessage
from workcompanion.rag.reranker import lexical_overlap
from workcompanion.rag.sparse_index import tokenize
from workcompanion.schemas.documents import DocumentChunk
from workcompanion.schemas.retrieval import RetrievedChunk
from workcompanion.utils.text_utils import sentences_from_text

logger = get_logger(__name__)

COMPRESS_PROMPT = """Extract only the sentences from the passage that are needed to \
answer the question. Copy sentences VERBATIM - never paraphrase, summarise or add
information. If nothing in the passage is relevant reply with exactly: NO_RELEVANT_CONTENT

Return STRICT JSON: {"sentences": ["verbatim sentence", ...]}"""


class ContextCompressor:
    """Reduce retrieved chunks to the minimum evidence needed."""

    def __init__(self, settings: Settings | None = None, llm: LLMProvider | None = None):
        self._settings = settings or get_settings()
        self._llm = llm

    # ------------------------------------------------------------------
    def compress(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        max_sentences: int | None = None,
        keep_tables_whole: bool = True,
    ) -> list[RetrievedChunk]:
        """Return copies of ``chunks`` trimmed to query-relevant sentences."""
        if not chunks:
            return []
        limit = max_sentences or self._settings.compressor_max_sentences
        output: list[RetrievedChunk] = []

        for source in chunks:
            # Deep-copy: the store may hand back shared chunk objects and the
            # compressed text must never overwrite what is persisted.
            chunk = source.model_copy(deep=True)
            text = source.chunk.text
            original_length = len(text)

            is_table = "|" in text and text.count("|") >= 4
            if is_table and keep_tables_whole:
                # Tables are structurally meaningful; trimming them destroys meaning.
                chunk.compression_ratio = 1.0
                output.append(chunk)
                continue

            trimmed = self._extractive(query, text, limit)
            if not trimmed:
                # Nothing matched - keep a short lead so the chunk is not empty.
                trimmed = self._lead(text, 2)
            if not trimmed.strip():
                continue

            chunk.chunk.text = trimmed
            chunk.compression_ratio = round(len(trimmed) / max(original_length, 1), 4)
            chunk.rerank_score = chunk.rerank_score or chunk.score
            output.append(chunk)

        return output

    # ------------------------------------------------------------------
    def _extractive(self, query: str, text: str, limit: int) -> str:
        """Keep the highest-scoring sentences, preserving document order."""
        sentences = sentences_from_text(text, min_length=25)
        if not sentences:
            return text.strip()[:1200]
        if len(sentences) <= limit:
            return "\n".join(sentences).strip()

        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(query_tokens & sentence_tokens) / (len(query_tokens) or 1)
            # Slight preference for earlier sentences (definitions first).
            position_bonus = 1.0 - (index / (len(sentences) * 2))
            scored.append((overlap + position_bonus * 0.12, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        keep = sorted(index for score, index in scored[:limit] if score > 0.05)
        if not keep:
            keep = [0]
        return "\n".join(sentences[index] for index in keep).strip()

    @staticmethod
    def _lead(text: str, sentences: int) -> str:
        parts = sentences_from_text(text, min_length=20)
        if not parts:
            return text.strip()[:400]
        return "\n".join(parts[:sentences])

    # ------------------------------------------------------------------
    def compress_with_llm(
        self, query: str, chunks: Sequence[RetrievedChunk], *, max_chunks: int = 4
    ) -> list[RetrievedChunk]:
        """LLM-assisted extraction (opt-in; costs tokens)."""
        if not self._llm or not chunks:
            return self.compress(query, chunks)
        output: list[RetrievedChunk] = []
        for source in chunks[:max_chunks]:
            chunk = source.model_copy(deep=True)
            request = LLMRequest(
                messages=[
                    SystemMessage(content=COMPRESS_PROMPT),
                    UserMessage(
                        content=f"QUESTION:\n{query}\n\nPASSAGE:\n{source.chunk.text[:2500]}"
                    ),
                ],
                temperature=0.0,
                max_tokens=700,
                metadata={"task": "compress"},
            )
            try:
                payload = self._llm.generate_json(request, retries=1)
            except Exception as exc:
                logger.warning("LLM compression failed (%s); keeping the extractive result.", exc)
                output.extend(self.compress(query, [source]))
                continue

            sentences: list[str] = []
            if isinstance(payload, dict):
                sentences = [str(s) for s in (payload.get("sentences") or []) if str(s).strip()]
            if not sentences or "NO_RELEVANT_CONTENT" in " ".join(sentences):
                continue
            kept = "\n".join(sentences).strip()
            if not kept:
                continue
            ratio = len(kept) / max(len(source.chunk.text), 1)
            chunk.chunk.text = kept
            chunk.compression_ratio = round(ratio, 4)
            output.append(chunk)
        return output

    # ------------------------------------------------------------------
    @staticmethod
    def build_context(
        chunks: Sequence[RetrievedChunk | DocumentChunk], *, max_chars: int = 12_000
    ) -> str:
        """Render chunks into a labelled, citable context block for the LLM."""
        blocks: list[str] = []
        total = 0
        for index, chunk in enumerate(chunks, start=1):
            header = ContextCompressor.citation_header(chunk, marker=index)
            body = (chunk.chunk if isinstance(chunk, RetrievedChunk) else chunk).text.strip()
            block = f"[{index}] {header}\n{body}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 400:
                    blocks.append(block[:remaining].rstrip() + "\n[...truncated]")
                break
            blocks.append(block)
            total += len(block) + 2
        return "\n\n".join(blocks)

    @staticmethod
    def citation_header(chunk: RetrievedChunk | DocumentChunk, *, marker: int = 0) -> str:
        """A compact provenance line, e.g. ``Source: Physics.pdf | p. 42 | Chapter 3``.

        Accepts either a scored :class:`RetrievedChunk` or a plain
        :class:`DocumentChunk` (whole-document rendering, e.g. paper analysis).
        """
        inner = chunk.chunk if isinstance(chunk, RetrievedChunk) else chunk
        metadata = inner.metadata
        bits = [f"Source: {metadata.document_name}"]
        if metadata.page is not None:
            bits.append(f"p. {metadata.page}")
        if metadata.chapter:
            bits.append(metadata.chapter)
        elif metadata.section:
            bits.append(metadata.section)
        elif inner.heading_path:
            bits.append(inner.heading_path)
        score = (
            chunk.rerank_score or chunk.score
            if isinstance(chunk, RetrievedChunk)
            else None
        )
        if score is not None:
            bits.append(f"relevance={score:.2f}")
        return " | ".join(bits)

    @staticmethod
    def relevance_score(query: str, text: str) -> float:
        """Lexical relevance used by the confidence calculator."""
        return lexical_overlap(query, text)


_WHITESPACE = re.compile(r"\s+")


def clean_context(text: str) -> str:
    """Normalise a context block for display."""
    return _WHITESPACE.sub(" ", text or "").strip()