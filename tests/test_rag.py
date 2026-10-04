"""RAG pipeline: chunking, hybrid retrieval, grounding and citations.

These are the guarantees that make the system trustworthy, so each one is pinned
by a test rather than assumed.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
class TestChunking:
    def test_splits_on_headings_and_keeps_metadata(self, settings, notes_file):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker

        chunks = SemanticChunker(settings).chunk_document(get_loader(notes_file).load(notes_file))

        assert chunks, "a multi-section note must produce chunks"
        assert all(chunk.text.strip() for chunk in chunks)
        # Each chunk records where it came from, or the citations are useless.
        assert all(chunk.metadata.document_name == notes_file.name for chunk in chunks)

    def test_chunk_ids_are_unique(self, settings, notes_file):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker

        chunks = SemanticChunker(settings).chunk_document(get_loader(notes_file).load(notes_file))
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_tiny_document_still_yields_one_chunk(self, settings, tmp_path):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker

        path = tmp_path / "documents" / "tiny.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("One short sentence.", encoding="utf-8")

        chunks = SemanticChunker(settings).chunk_document(get_loader(path).load(path))
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class TestIngestion:
    def test_ingests_markdown_and_indexes_it(self, bundle, notes_file):
        result = bundle.ingestion.ingest_file(notes_file, subject="Physics")

        assert result.ok, result.error
        assert result.chunk_count > 0
        assert not bundle.pipeline.is_empty

    def test_ingestion_is_idempotent(self, bundle, notes_file):
        """Re-ingesting identical bytes must not double the corpus."""
        first = bundle.ingestion.ingest_file(notes_file, subject="Physics")
        assert first.ok, first.error

        second = bundle.ingestion.ingest_file(
            notes_file, subject="Physics", allow_duplicate=False
        )

        assert second.status == "skipped"
        assert bundle.pipeline.chunk_count == first.chunk_count
        assert len(bundle.store.document_ids()) == 1

    def test_forced_reindex_replaces_rather_than_appends(self, bundle, notes_file):
        """allow_duplicate=True replaces the document instead of doubling it."""
        first = bundle.ingestion.ingest_file(notes_file, subject="Physics")
        again = bundle.ingestion.ingest_file(
            notes_file, subject="Physics", allow_duplicate=True
        )

        assert again.ok, again.error
        assert bundle.pipeline.chunk_count == first.chunk_count
        assert len(bundle.store.document_ids()) == 1

    def test_same_content_under_a_new_name_is_still_a_duplicate(
        self, bundle, notes_file
    ):
        """The document id is derived from content, not the filename."""
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        renamed = notes_file.parent / "lecture3_copy.md"
        renamed.write_text(notes_file.read_text(encoding="utf-8"), encoding="utf-8")

        result = bundle.ingestion.ingest_file(renamed, subject="Physics")
        assert result.status == "skipped"

    def test_rejects_an_unsupported_extension(self, bundle, tmp_path):
        path = tmp_path / "documents" / "notes.exe"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZ\x90\x00")

        result = bundle.ingestion.ingest_file(path, subject="Physics")
        assert not result.ok
        assert result.error

    @pytest.mark.parametrize(
        ("name", "body"),
        [
            ("table.csv", "organelle,size_um\nMitochondrion,0.5-1\nNucleus,1\n"),
            ("page.html", "<h1>Photosynthesis</h1><p>Oxygen comes from water.</p>"),
            ("plain.txt", "Quantum mechanics notes. Superposition precedes measurement."),
        ],
    )
    def test_every_text_loader_produces_chunks(
        self, bundle, tmp_path, name: str, body: str
    ):
        path = tmp_path / "documents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

        result = bundle.ingestion.ingest_file(path, subject="Mixed")
        assert result.ok, f"{name} failed to ingest: {result.error}"
        assert result.chunk_count > 0

    def test_csv_tables_are_preserved(self, bundle, tmp_path):
        path = tmp_path / "documents" / "table.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "organelle,function\nMitochondrion,oxidative phosphorylation\n",
            encoding="utf-8",
        )

        result = bundle.ingestion.ingest_file(path, subject="Biology")
        assert result.ok

        retrieved = bundle.pipeline.retrieve("What does the mitochondrion do?")
        text = " ".join(chunk.chunk.text for chunk in retrieved.chunks)
        assert "oxidative phosphorylation" in text.lower()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestRetrieval:
    def test_finds_the_relevant_section(self, bundle, notes_file):
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        outcome = bundle.pipeline.retrieve("What is the Gibbs free energy criterion?")

        assert outcome.chunks
        text = " ".join(chunk.chunk.text for chunk in outcome.chunks).lower()
        assert "gibbs" in text or "g = h - t s" in text

    def test_confidence_is_bounded(self, bundle, notes_file):
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        outcome = bundle.pipeline.retrieve("Explain Gibbs free energy")
        assert 0.0 <= outcome.confidence <= 1.0

    def test_off_topic_question_is_not_marked_grounded(self, bundle, notes_file):
        """The confidence floor is the whole hallucination-control mechanism."""
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        outcome = bundle.pipeline.retrieve(
            "What is the airspeed velocity of an unladen swallow?"
        )
        assert not outcome.grounded_context_available

    def test_document_filter_scopes_results(self, bundle, notes_file):
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        outcome = bundle.pipeline.retrieve(
            "Entropy", document_filter="thermodynamics.md"
        )
        if outcome.chunks:
            assert all(
                chunk.chunk.metadata.document_name == "thermodynamics.md"
                for chunk in outcome.chunks
            )

    def test_empty_index_returns_nothing(self, bundle):
        outcome = bundle.pipeline.retrieve("anything")
        assert not outcome.chunks
        assert not outcome.grounded_context_available


# ---------------------------------------------------------------------------
# Sparse index
# ---------------------------------------------------------------------------
class TestBM25:
    def test_finds_exact_terms_dense_search_would_miss(self, settings, notes_file):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker
        from workcompanion.rag.sparse_index import BM25Index

        chunks = SemanticChunker(settings).chunk_document(get_loader(notes_file).load(notes_file))
        index = BM25Index()
        index.add_chunks(chunks)

        # ``search`` returns (chunk_id, score) pairs.
        hits = index.search("latent heat fusion", top_k=3)
        assert hits, "a distinctive phrase must match"

        matching = [c for c in chunks if "latent" in c.text.lower() and c.chunk_id in dict(hits)]
        assert matching, "the passage mentioning latent heat should rank"

    def test_round_trips_through_disk(self, settings, tmp_path, notes_file):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker
        from workcompanion.rag.sparse_index import BM25Index

        chunks = SemanticChunker(settings).chunk_document(get_loader(notes_file).load(notes_file))
        original = BM25Index()
        original.add_chunks(chunks)
        path = tmp_path / "bm25.json"
        original.save(path)

        restored = BM25Index()
        restored.load(path)
        assert restored.size == original.size

    def test_document_scoping_removes_only_that_document(self, settings, notes_file):
        from workcompanion.loaders.registry import get_loader
        from workcompanion.rag.chunking import SemanticChunker
        from workcompanion.rag.sparse_index import BM25Index

        chunks = SemanticChunker(settings).chunk_document(get_loader(notes_file).load(notes_file))
        index = BM25Index()
        index.add_chunks(chunks)
        assert index.size > 0

        # ``remove_document`` takes a document id, not a filename.
        index.remove_document(chunks[0].metadata.document_id)
        assert index.size == 0


# ---------------------------------------------------------------------------
# Citations and grounding
# ---------------------------------------------------------------------------
class TestCitations:
    def test_citations_reference_real_chunks(self, bundle, notes_file):
        bundle.ingestion.ingest_file(notes_file, subject="Physics")

        outcome = bundle.pipeline.retrieve("What is the second law?")
        assert outcome.citations
        for citation in outcome.citations:
            assert citation.document_name == notes_file.name
            assert citation.quote

    def test_verify_grounding_drops_fabricated_markers(self, bundle, notes_file):
        from workcompanion.rag.citations import verify_grounding

        bundle.ingestion.ingest_file(notes_file, subject="Physics")
        outcome = bundle.pipeline.retrieve("What is the second law?")
        assert outcome.citations

        real = outcome.citations[0].marker
        answer = f"The second law holds {real}. Nothing supports this [99]."
        kept, fabricated = verify_grounding(answer, outcome.citations, strict=True)

        assert fabricated == ["[99]"]
        assert [c.marker for c in kept] == [real]

    def test_refusal_reports_failure_not_success(self, bundle, notes_file):
        """A refusal is a handled request, but it is not an answer."""
        from workcompanion.schemas.common import GroundingLabel

        bundle.ingestion.ingest_file(notes_file, subject="Physics")
        result = bundle.rag.answer(
            "What is the airspeed velocity of an unladen swallow?",
            allow_general_knowledge=False,
        )

        assert not result.success
        assert result.confidence == 0.0
        assert GroundingLabel.RETRIEVED_FACT not in result.grounding
        assert result.warnings
