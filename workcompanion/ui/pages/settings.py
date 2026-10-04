"""Settings: live configuration, the agent roster and a health report.

Changes are written to ``.env.local`` (not ``.env``) so a committed ``.env`` with
real keys is never overwritten by a click in the UI.
"""

from __future__ import annotations

import streamlit as st

from workcompanion.agents import agent_catalogue
from workcompanion.config.logging_config import get_logger
from workcompanion.ui import state, theme

logger = get_logger(__name__)


def render_settings(context: state.AppContext) -> None:
    """Configuration forms plus a read-only report of what is actually loaded."""
    theme.hero("Settings", "Tune retrieval and agents, then see exactly what is running.")
    tabs = st.tabs(["Environment", "Retrieval", "Agents", "Learner", "Maintenance"])
    with tabs[0]:
        _render_environment(context)
    with tabs[1]:
        _render_retrieval(context)
    with tabs[2]:
        _render_agents(context)
    with tabs[3]:
        _render_learner(context)
    with tabs[4]:
        _render_maintenance(context)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _render_environment(context: state.AppContext) -> None:
    settings = context.settings
    st.markdown(
        "Edits are stored in `.env.local` and take effect on the next run. "
        "Your `.env` (which may hold real keys) is never modified."
    )

    key = st.text_input(
        "Groq API key",
        value="",
        type="password",
        key="set_key",
        placeholder="configured" if settings.has_groq_key else "gsk_…",
        help="Leave blank to keep the current value. Stored in `.env.local` only.",
    )
    model = st.text_input(
        "Model", value=settings.groq_model, key="set_model",
        help="Any Groq chat model, e.g. llama-3.3-70b-versatile.",
    )
    temperature = st.slider("Temperature", 0.0, 1.5, settings.groq_temperature, 0.05, key="set_temp")
    max_tokens = st.number_input(
        "Max output tokens", 256, 16384, settings.groq_max_tokens, step=64, key="set_tokens"
    )
    provider = st.selectbox(
        "Provider",
        ["groq", "offline"],
        index=["groq", "offline"].index(settings.llm_provider),
        key="set_provider",
        help="'offline' forces the deterministic extractive provider - useful for "
        "demos and for verifying the pipeline without a network call.",
    )
    web = st.toggle("Enable web research", value=settings.enable_web_research, key="set_web")
    ocr = st.toggle(
        "Enable OCR fallback for scanned PDFs",
        value=settings.enable_ocr_fallback,
        key="set_ocr",
    )

    if st.button("Save environment", type="primary", key="set_save_env"):
        overrides: dict[str, object] = {
            "LLM_PROVIDER": provider,
            "GROQ_MODEL": model,
            "GROQ_TEMPERATURE": temperature,
            "GROQ_MAX_TOKENS": int(max_tokens),
            "ENABLE_WEB_RESEARCH": str(web).lower(),
            "ENABLE_OCR_FALLBACK": str(ocr).lower(),
        }
        if key.strip():
            overrides["GROQ_API_KEY"] = key.strip()
        state.apply_settings(overrides)
        st.success("Saved. The next request will use the new configuration.")
        st.rerun()

    st.divider()
    st.markdown("**What is loaded right now**")
    theme.kv(state.environment_report(context))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _render_retrieval(context: state.AppContext) -> None:
    settings = context.settings
    st.caption(
        "Hybrid retrieval blends dense embeddings with BM25. Raising the dense weight "
        "favours paraphrase; lowering it favours exact terminology."
    )

    dense = st.slider(
        "Dense weight", 0.0, 1.0, settings.dense_weight, 0.05, key="set_dense",
        help="Complement of the sparse weight; the two always sum to 1.",
    )
    top_k = st.slider("Passages per query", 1, 20, settings.retrieval_top_k, key="set_topk")
    candidates = st.slider(
        "Candidates before reranking",
        5,
        100,
        settings.retrieval_candidate_k,
        step=5,
        key="set_cand",
    )
    reranker_options = ["auto", "local", "cross-encoder", "llm"]
    reranker = st.selectbox(
        "Reranker",
        reranker_options,
        index=reranker_options.index(settings.reranker)
        if settings.reranker in reranker_options
        else 0,
        key="set_reranker",
        help="'auto' prefers a cross-encoder, 'local' is dependency-free.",
    )
    minimum = st.slider(
        "Minimum retrieval confidence",
        0.0,
        0.6,
        settings.min_retrieval_confidence,
        0.01,
        key="set_minconf",
        help="Below this, the research agent refuses to answer instead of stretching "
        "loosely-related passages.",
    )
    rewrite = st.toggle(
        "Query rewriting and expansion",
        value=settings.enable_query_rewrite,
        key="set_rewrite",
    )
    expansion = st.toggle(
        "Keyword expansion", value=settings.enable_query_expansion, key="set_expand"
    )
    multi_queries = st.slider(
        "Extra queries per request", 1, 5, settings.multi_query_count, key="set_multi"
    )
    compress = st.toggle(
        "Context compression", value=settings.context_compression, key="set_compress"
    )

    if st.button("Save retrieval settings", type="primary", key="set_save_rag"):
        state.apply_settings(
            {
                "DENSE_WEIGHT": round(dense, 2),
                "SPARSE_WEIGHT": round(1.0 - dense, 2),
                "RETRIEVAL_TOP_K": int(top_k),
                "RETRIEVAL_CANDIDATE_K": int(candidates),
                "RERANKER": reranker,
                "MIN_RETRIEVAL_CONFIDENCE": round(minimum, 3),
                "ENABLE_QUERY_REWRITE": str(rewrite).lower(),
                "ENABLE_QUERY_EXPANSION": str(expansion).lower(),
                "MULTI_QUERY_COUNT": int(multi_queries),
                "CONTEXT_COMPRESSION": str(compress).lower(),
            }
        )
        st.success("Saved.")
        st.rerun()


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _render_agents(context: state.AppContext) -> None:
    settings = context.settings
    level = st.selectbox(
        "Default explanation level",
        ["beginner", "undergraduate", "graduate", "expert"],
        index=["beginner", "undergraduate", "graduate", "expert"].index(
            settings.default_explanation_level
        ),
        key="set_level",
    )
    style = st.selectbox(
        "Default teaching style",
        ["concise", "detailed", "socratic", "example_based", "mathematical", "analogy"],
        index=["concise", "detailed", "socratic", "example_based", "mathematical", "analogy"].index(
            settings.default_teaching_style
        ),
        key="set_style",
    )
    crew = st.toggle(
        "Enable CrewAI orchestration",
        value=settings.enable_crewai,
        key="set_crew",
        help="Crews are only used for genuinely multi-step work (deep research, exam "
        "prep, study sessions). Everything else runs on a single agent.",
    )
    verbose = st.toggle("Verbose CrewAI logging", value=settings.crewai_verbose, key="set_crew_verbose")
    max_steps = st.slider("Max agent steps", 1, 40, settings.max_agent_steps, key="set_steps")

    if st.button("Save agent settings", type="primary", key="set_save_agents"):
        state.apply_settings(
            {
                "DEFAULT_EXPLANATION_LEVEL": level,
                "DEFAULT_TEACHING_STYLE": style,
                "ENABLE_CREWAI": str(crew).lower(),
                "CREWAI_VERBOSE": str(verbose).lower(),
                "MAX_AGENT_STEPS": int(max_steps),
            }
        )
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.markdown("**Agent roster**")
    for name, agent in context.bundle.agents().items():
        theme.agent_card(name, str(agent.description))

    with st.expander("Catalogue (from the agent registry)", expanded=False):
        st.dataframe(
            [
                {
                    "Agent": entry["name"],
                    "Class": entry["class"],
                    "Responsibility": entry["description"],
                }
                for entry in agent_catalogue()
            ],
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------
def _render_learner(context: state.AppContext) -> None:
    profile = context.profile_snapshot()
    st.caption("These are stored in the database, not in `.env`.")
    left, right = st.columns(2, gap="medium")
    level = left.selectbox(
        "Preferred level",
        ["beginner", "undergraduate", "graduate", "expert"],
        index=["beginner", "undergraduate", "graduate", "expert"].index(profile.preferred_level),
        key="set_pref_level",
    )
    style = left.selectbox(
        "Preferred style",
        ["concise", "detailed", "socratic", "example_based", "mathematical", "analogy"],
        index=["concise", "detailed", "socratic", "example_based", "mathematical", "analogy"].index(
            profile.preferred_style
        ),
        key="set_pref_style",
    )
    subject = left.text_input(
        "Default subject", value=profile.default_subject or "", key="set_pref_subject"
    )
    hours = right.slider(
        "Weekly study hours", 0.5, 40.0, float(profile.weekly_study_hours or 5.0), 0.5, key="set_pref_hours"
    )

    if st.button("Save learner preferences", type="primary", key="set_save_learner"):
        state.set_preferences(
            context,
            preferred_level=level,
            preferred_style=style,
            default_subject=subject or None,
            weekly_study_hours=float(hours),
        )
        st.success("Saved to your learner profile.")
        st.rerun()

    if profile.misconceptions:
        st.markdown("**Misconceptions recorded**")
        for misconception in profile.misconceptions:
            theme.warning(misconception)
        if st.button("Clear all misconceptions", key="set_clear_mis"):
            with context.session() as repositories:
                from workcompanion.memory.learner_profile import LearnerProfile

                LearnerProfile(repositories).clear_misconceptions()
            st.rerun()


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
def _render_maintenance(context: state.AppContext) -> None:
    settings = context.settings
    status = context.bundle.status()

    st.markdown("**Paths**")
    theme.kv(
        [
            ("Project root", str(settings.data_dir.parent)),
            ("Data directory", str(settings.data_dir)),
            ("Documents", str(settings.documents_dir)),
            ("Cache", str(settings.cache_dir)),
            ("Log level", settings.log_level),
        ]
    )

    st.markdown("**Index**")
    theme.kv(
        [
            ("Chunks", status["chunks"]),
            ("Documents present", "yes" if status["has_documents"] else "no"),
            ("LLM", f"{status['llm'].get('provider')} / {status['llm'].get('model')}"),
            ("Embeddings", str(status["embeddings"])),
            ("Web research", "on" if status["web_enabled"] else "off"),
        ]
    )

    st.divider()
    left, right = st.columns(2, gap="small")
    if left.button("Clear the LLM cache", key="set_clear_cache"):
        removed = context.bundle.cache.clear()
        st.success(f"Removed {removed} cached response(s).")

    if right.button("Re-apply the current `.env`", key="set_reload"):
        state.apply_settings({})
        st.success("Configuration reloaded.")

    st.markdown("**Danger zone**")
    st.caption(
        "Drops every indexed chunk and the BM25 index. Your uploaded files and your "
        "progress history are kept."
    )
    confirm = st.checkbox("I understand this empties the knowledge index", key="set_confirm")
    if st.button(
        "Empty the index",
        key="set_empty",
        type="secondary",
        disabled=not confirm,
    ):
        with st.spinner("Clearing…"):
            context.bundle.reset()
        st.success("Index cleared. Re-upload your material in My Knowledge.")
        st.rerun()

    with st.expander("Diagnostic snapshot", expanded=False):
        st.json(status)


__all__ = ["render_settings"]