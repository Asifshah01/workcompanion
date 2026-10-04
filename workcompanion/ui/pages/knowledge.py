"""My Knowledge: upload, index, inspect and search the document library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)


def render_knowledge(context: state.AppContext) -> None:
    """Document management plus a retrieval sandbox."""
    theme.hero("My Knowledge", "Upload material, inspect how it was chunked, and test retrieval.")

    tabs = st.tabs(["Upload", "Library", "Retrieval sandbox", "Index health"])

    with tabs[0]:
        _render_upload(context)
    with tabs[1]:
        _render_library(context)
    with tabs[2]:
        _render_sandbox(context)
    with tabs[3]:
        _render_health(context)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def _render_upload(context: state.AppContext) -> None:
    settings = context.settings
    st.markdown(
        f"**Supported:** {' · '.join(settings.allowed_upload_extensions)} "
        f"(up to {settings.max_upload_mb} MB each). PDFs are OCR'd when they have no text layer."
    )
    subject = st.text_input("Subject (optional)", key="kb_subject", placeholder="e.g. Thermodynamics")
    uploaded = st.file_uploader(
        "Choose files",
        type=[ext.lstrip(".") for ext in settings.allowed_upload_extensions],
        accept_multiple_files=True,
        key="kb_upload",
    )
    if not uploaded:
        _render_ingest_sample(context)
        return

    if st.button(f"Ingest {len(uploaded)} file(s)", type="primary", key="kb_ingest"):
        _ingest_uploaded(context, uploaded, subject=subject or None)
    _render_ingest_sample(context)


def _ingest_uploaded(context: state.AppContext, files: list[Any], *, subject: str | None) -> None:
    settings = context.settings
    documents_dir = Path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for upload in files:
        name = getattr(upload, "name", "upload")
        target = documents_dir / Path(name).name
        target.write_bytes(upload.getbuffer())
        saved.append(target)

    _ingest_paths(context, saved, subject=subject)
    st.session_state["kb_upload"] = None
    st.rerun()


def _render_ingest_sample(context: state.AppContext) -> None:
    """One-click demo material so a first run has something to retrieve."""
    if context.bundle.pipeline.is_empty and context.settings.demo_mode_autoload:
        st.divider()
        st.markdown("**No material yet?**")
        st.caption(
            "Load a small thermodynamics set (2 documents) to try every feature: "
            "grounded answers, quizzes, flashcards and a study plan."
        )
        if st.button("Load demo material", key="kb_demo", type="primary"):
            for path in demo_files(context):
                _ingest_paths(context, [path], subject="Thermodynamics", quiet=True)
            st.rerun()


def demo_files(context: state.AppContext) -> list[Path]:
    """Materialise the bundled demo documents into the documents directory."""
    source = Path(context.settings.documents_dir).parent / "demo"
    target = Path(context.settings.documents_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not source.exists():
        return written
    for item in sorted(source.glob("*")):
        if item.suffix.lower() in {".md", ".txt"} and item.is_file():
            destination = target / item.name
            if not destination.exists():
                destination.write_bytes(item.read_bytes())
            written.append(destination)
    return written


def _ingest_paths(
    context: state.AppContext,
    paths: list[Path],
    *,
    subject: str | None = None,
    quiet: bool = False,
) -> list[Any]:
    results = []
    for path in paths:
        with st.spinner(f"Indexing {path.name}…"):
            try:
                outcome = context.bundle.ingestion.ingest_file(path, subject=subject)
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                logger.exception("ingestion failed for %s", path)
                st.error(f"{path.name}: {exc}")
                continue
        results.append(outcome)
        if outcome.status == "failed":
            st.error(f"{outcome.document_name}: {outcome.error}")
            continue
        if outcome.duplicate_of:
            st.warning(
                f"{outcome.document_name} is a duplicate of an already indexed document "
                f"({outcome.duplicate_of[:8]}…). Nothing was duplicated."
            )
            continue
        message = outcome.summary()
        if outcome.warnings:
            message += f"  · {len(outcome.warnings)} notice(s)"
        if quiet:
            logger.info("[ui] ingested %s", message)
        else:
            st.success(message)
            for warning in outcome.warnings[:3]:
                theme.hint(f"⚠️ {warning}")

    context.bundle.nexus.set_document_state(not context.bundle.pipeline.is_empty)
    return results


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
def _render_library(context: state.AppContext) -> None:
    with context.session() as repositories:
        records = repositories.documents.list_all(repositories.session)
        total, indexed = repositories.documents.counts(repositories.session)

    theme.stat_row(
        [("Indexed", indexed), ("Total", total), ("Chunks", context.bundle.pipeline.chunk_count)]
    )
    if not records:
        st.info("Nothing indexed yet. Upload a file on the **Upload** tab.")
        return

    for record in records:
        with st.container(border=True):
            title = record.title or record.name
            theme.badge_row(
                [
                    (record.status, "high" if record.status == "indexed" else "low", "●"),
                    (record.source_type, "neutral", "📄"),
                    (f"{record.chunk_count or 0} chunks", "neutral", "🧩"),
                ]
            )
            st.markdown(f"**{title}**")
            bits = [f"{record.char_count or 0:,} chars"]
            if record.page_count:
                bits.append(f"{record.page_count} pages")
            if record.subject:
                bits.append(record.subject)
            if record.used_ocr:
                bits.append("OCR")
            theme.hint(" · ".join(bits))
            if record.error:
                theme.warning(record.error)

            columns = st.columns([1, 1, 1, 4], gap="small")
            if columns[0].button("Analyse", key=f"kb_an_{record.id[:8]}"):
                state.navigate("research")
                st.session_state["wc_active_document"] = record.id
                st.session_state["auto_analyse"] = record.id
                st.rerun()
            if columns[1].button("Quiz me", key=f"kb_qz_{record.id[:8]}"):
                state.navigate("quiz")
                st.session_state["quiz_seed"] = record.title or record.name
                st.rerun()
            if columns[2].button("Delete", key=f"kb_del_{record.id[:8]}"):
                _delete_document(context, record.id, record.name or "document")
                st.rerun()
            with columns[3].popover("Chunk map"):
                _render_chunk_map(context, record)


def _delete_document(context: state.AppContext, document_id: str, name: str) -> None:
    context.bundle.ingestion.remove_document(document_id)
    context.bundle.nexus.set_document_state(not context.bundle.pipeline.is_empty)
    logger.info("[ui] removed document %s", document_id)


def _render_chunk_map(context: state.AppContext, record: Any) -> None:
    """Show the chunks produced for a document, with their metadata."""
    chunks = []
    try:
        chunks = list(context.bundle.pipeline.store.all_chunks(document_id=record.id))
    except Exception as exc:  # pragma: no cover
        st.caption(f"Could not read chunks: {exc}")
        return
    if not chunks:
        st.caption("No chunks stored for this document.")
        return
    st.caption(f"{len(chunks)} chunk(s) · semantic splitting with overlap")
    for index, chunk in enumerate(chunks[:25], start=1):
        meta = chunk.metadata
        heading = (
            meta.section
            or meta.chapter
            or meta.document_name
            or f"chunk {index}"
        )
        location = f" · p. {meta.page}" if meta.page else ""
        st.markdown(f"**[{index}]** {heading} · {len(chunk.text)} chars{location}")
        st.caption(chunk.text[:220].replace("\n", " ") + ("…" if len(chunk.text) > 220 else ""))
    if len(chunks) > 25:
        st.caption(f"…and {len(chunks) - 25} more.")


# ---------------------------------------------------------------------------
# Retrieval sandbox
# ---------------------------------------------------------------------------
def _render_sandbox(context: state.AppContext) -> None:
    """Run the pipeline directly and inspect every stage."""
    st.caption(
        "This runs the full RAG pipeline without calling a language model, so you can see "
        "exactly which passages a question retrieves and why."
    )
    left, right = st.columns([3, 1], gap="medium")
    question = left.text_input(
        "Question", key="kb_probe", placeholder="What determines whether a process is spontaneous?"
    )
    top_k = right.slider("Passages", 1, 15, context.settings.retrieval_top_k, key="kb_topk")

    if not st.button("Run retrieval", key="kb_run", type="primary"):
        return
    if not question.strip():
        st.warning("Type a question first.")
        return

    with st.spinner("Retrieving…"):
        outcome = context.bundle.rag.retrieve(question.strip(), top_k=top_k)
    components.render_retrieval(outcome)


# ---------------------------------------------------------------------------
# Index health
# ---------------------------------------------------------------------------
def _render_health(context: state.AppContext) -> None:
    store = context.bundle.store
    sparse = context.bundle.sparse
    describe = store.describe() if hasattr(store, "describe") else {}
    theme.kv(
        [
            ("Vector store", type(store).__name__),
            ("Documents", len(store.document_ids())),
            ("Chunks", store.count()),
            ("Sparse postings", sparse.size),
            ("Average chunk length", f"{sparse.average_length:.0f} tokens"),
            ("Embeddings", context.settings.embedding_model),
            ("Hybrid weights", f"{context.settings.dense_weight:g} dense / {context.settings.sparse_weight:g} sparse"),
            ("Reranker", context.settings.reranker),
            ("Store details", str(describe.get("path") or describe or "-")),
        ]
    )
    if hasattr(store, "counts_by_document"):
        with st.expander("Chunks per document", expanded=True):
            st.json({k[:8]: v for k, v in store.counts_by_document().items()})

    st.divider()
    st.markdown("**Rebuild the sparse index**")
    st.caption(
        "Only needed if the BM25 index is out of step with the vector store (for example "
        "after restoring a database without its data directory)."
    )
    if st.button("Rebuild", key="kb_rebuild"):
        with st.spinner("Rebuilding…"):
            context.bundle.ingestion.rebuild_sparse_index()
        st.success("Sparse index rebuilt.")