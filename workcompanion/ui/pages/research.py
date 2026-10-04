"""Research: structured paper analysis and multi-source synthesis."""

from __future__ import annotations

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)


def render_research(context: state.AppContext) -> None:
    """Two modes: analyse one indexed paper, or synthesise across sources."""
    theme.hero(
        "Research",
        "Structured analysis of an indexed paper, or a synthesis across your documents "
        "and (optionally) the web — with documents and web sources cited separately.",
    )
    tabs = st.tabs(["Analyse a paper", "Synthesise a question", "Web search"])
    with tabs[0]:
        _render_paper(context)
    with tabs[1]:
        _render_synthesis(context)
    with tabs[2]:
        _render_web(context)


# ---------------------------------------------------------------------------
# Paper analysis
# ---------------------------------------------------------------------------
def _render_paper(context: state.AppContext) -> None:
    with context.session() as repositories:
        records = repositories.documents.list_all(repositories.session, status="indexed")
    if not records:
        st.info("Index a document in **My Knowledge** first.")
        return

    by_id = {record.id: record for record in records}
    labels = {f"{record.title or record.name}  ·  {record.chunk_count or 0} chunks": record.id for record in records}
    preset = st.session_state.get("wc_active_document")
    options = list(labels)
    index = 0
    if preset in by_id:
        for position, label in enumerate(options):
            if labels[label] == preset:
                index = position
                break

    left, right = st.columns([3, 1], gap="medium")
    label = left.selectbox("Document", options, index=index, key="rs_doc")
    level = right.selectbox(
        "Level", ["beginner", "undergraduate", "graduate", "expert"], key="rs_level"
    )
    document_id = labels[label]

    st.session_state["wc_active_document"] = document_id
    if not st.button("Analyse", key="rs_run", type="primary"):
        return

    record = by_id.get(document_id)
    with st.spinner(f"Analysing {record.name if record else 'document'}…"):
        analysis = context.bundle.research.analyze_paper(
            document_id,
            document_name=record.name if record else None,
            level=level,
        )

    theme.stat_row(
        [
            ("Completeness", f"{analysis.completeness:.0%}"),
            ("Confidence", f"{analysis.confidence:.0%}"),
            ("Citations", len(analysis.citations)),
            ("Sections found", sum(
                1 for value in (
                    analysis.research_problem,
                    analysis.methodology,
                    analysis.key_results,
                    analysis.limitations,
                ) if value
            )),
        ]
    )
    st.markdown(context.bundle.research.render_analysis(analysis))
    components.render_warnings(analysis.warnings)
    components.render_sources(analysis.citations, title="Analysed passages")

    if analysis.contribution_summary:
        with st.expander("Contribution & novelty", expanded=False):
            st.markdown(f"**Novelty:** {analysis.novelty or 'not stated'}")
            st.markdown(analysis.contribution_summary)
    if analysis.research_gaps or analysis.future_work:
        with st.expander("Research gaps & future work", expanded=False):
            for gap in (*analysis.research_gaps, *analysis.future_work):
                st.markdown(f"- {gap}")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def _render_synthesis(context: state.AppContext) -> None:
    settings = context.settings
    left, right = st.columns([3, 1], gap="medium")
    question = left.text_input(
        "Research question",
        key="rs_question",
        placeholder="What determines whether a process is spontaneous?",
    )
    web = right.toggle(
        "Include web sources",
        value=settings.enable_web_research,
        key="rs_web",
        disabled=not context.bundle.web_search.enabled,
        help=(
            "Disabled automatically when web research is off in Settings."
            if not context.bundle.web_search.enabled
            else "Web results are cited as [W1], [W2]…"
        ),
    )
    document_filter = st.text_input(
        "Restrict to documents matching…", key="rs_filter", placeholder="(optional)"
    )

    if not st.button("Research", key="rs_go", type="primary", disabled=not question.strip()):
        return

    with st.spinner("Retrieving evidence and searching sources…"):
        brief = context.bundle.research.research(
            question.strip(),
            use_web=web,
            document_filter=document_filter.strip() or None,
        )

    theme.badge_row(
        [
            (f"confidence {brief.confidence:.0%}", theme.level_key(brief.confidence), "🎯"),
            (f"{len(brief.citations)} citations", "neutral", "📎"),
            ("web used" if brief.used_web else "documents only", "neutral", "🌐"),
        ]
    )
    st.markdown(brief.answer)

    if brief.document_findings:
        with st.expander("From your documents", expanded=False):
            for finding in brief.document_findings:
                st.markdown(f"- {finding}")
    if brief.web_findings:
        with st.expander("From the web", expanded=False):
            for finding in brief.web_findings:
                st.markdown(f"- {finding}")
    if brief.conflicting_points:
        with st.expander("Where sources disagree", expanded=True):
            for point in brief.conflicting_points:
                theme.warning(point)
    if brief.research_gaps:
        with st.expander("Still unknown", expanded=True):
            for gap in brief.research_gaps:
                st.markdown(f"- {gap}")
    if brief.suggested_readings:
        with st.expander("Suggested next reading", expanded=False):
            for reading in brief.suggested_readings:
                st.markdown(f"- {reading}")

    components.render_sources(brief.citations)
    components.render_warnings(brief.warnings)


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def _render_web(context: state.AppContext) -> None:
    tool = context.bundle.web_search
    if not tool.enabled:
        st.warning(
            f"Web research is unavailable ({tool.last_error or 'disabled in Settings'}). "
            "Everything else in the app still works from your own documents."
        )
        return
    query = st.text_input("Search the web", key="web_query", placeholder="lattice gauge theory")
    limit = st.slider("Results", 1, 10, context.settings.web_max_results, key="web_limit")
    if not st.button("Search", key="web_go", type="primary", disabled=not query.strip()):
        return
    with st.spinner("Searching…"):
        sources = tool.search(query.strip(), max_results=limit)
    if not sources:
        st.info("No results (the search endpoint may be blocked; this never breaks the app).")
        return
    for index, source in enumerate(sources, start=1):
        with st.container(border=True):
            st.markdown(f"**[W{index}] {source.title}**")
            st.caption(source.url)
            st.markdown(source.snippet)