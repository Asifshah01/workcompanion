"""Dashboard: the ask-anything surface plus a health and quick-start overview."""

from __future__ import annotations

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import chat, components, state, theme

logger = get_logger(__name__)

QUICK_STARTS = [
    ("Explain a concept", "Explain the second law of thermodynamics to a beginner"),
    ("Answer from my notes", "What do my notes say about Gibbs free energy?"),
    ("Test me", "Quiz me on entropy with 6 medium questions"),
    ("Build cards", "Make 10 flashcards on enthalpy"),
    ("Plan revision", "Plan my revision for thermodynamics over 7 days at 2 hours a day"),
    ("Research a topic", "Summarise the current state of quantum error correction research"),
]


def render_dashboard(context: state.AppContext) -> None:
    """Landing page: stats, quick starts, then the routed chat surface."""
    theme.hero(
        context.settings.app_name,
        "Ask anything about your material. NEXUS routes each request to the right "
        "specialist agent, and every answer shows its sources and its confidence.",
    )

    _render_stats(context)
    _render_quick_starts(context)

    st.divider()
    level = st.select_slider(
        "Explanation level",
        options=["beginner", "undergraduate", "graduate", "expert"],
        value=context.settings.default_explanation_level,
        key="dash_level",
        help="Used as the default; NEXUS overrides it when you ask for a level.",
    )
    use_web = st.toggle(
        "Allow web research",
        value=context.settings.enable_web_research,
        key="dash_web",
        help="Only used for research-style questions.",
    )
    use_crew = st.toggle(
        "Use CrewAI orchestration when available",
        value=False,
        key="dash_crew",
        help="Multi-agent research/exam workflows. Falls back to single agents automatically.",
    )
    theme.hint(
        f"CrewAI is {'available' if context.settings.enable_crewai and context.settings.has_groq_key else 'not available'}"
        " (needs `GROQ_API_KEY` and `ENABLE_CREWAI=true`)."
    )

    chat.chat(
        context,
        placeholder="Ask about your material, or say “quiz me on …”",
        level=level,
        use_web=use_web,
        enable_crew=use_crew,
        key="dash_chat",
    )


def _render_stats(context: state.AppContext) -> None:
    progress = context.progress_snapshot()
    bundle = context.bundle
    theme.stat_row(
        [
            ("Documents", f"{progress.documents_indexed}/{progress.documents_total}"),
            ("Chunks", f"{bundle.pipeline.chunk_count:,}"),
            ("Questions", f"{progress.questions_answered}"),
            (
                "Accuracy",
                f"{progress.average_quiz_score:.0f}%" if progress.average_quiz_score else "—",
            ),
            ("Study time", f"{progress.study_minutes:.0f}m"),
        ]
    )
    if progress.recommendation:
        theme.badge_row([("Next step", "neutral", "🎯"), (progress.recommendation, "neutral", "")])


def _render_quick_starts(context: state.AppContext) -> None:
    with st.expander("Quick starts", expanded=not state.history()):
        columns = st.columns(3, gap="small")
        for index, (label, prompt) in enumerate(QUICK_STARTS):
            column = columns[index % len(columns)]
            if column.button(label, key=f"qs_{index}", help=prompt):
                st.session_state["dash_chat_queued"] = prompt
                st.rerun()
        if not context.bundle.pipeline.is_empty:
            st.caption(
                "Tip: every answer shows which passages it used. Click **Sources** to open them."
            )
        else:
            st.info(
                "No documents indexed yet — answers will be refused rather than invented. "
                "Upload material in **My Knowledge** to get grounded answers."
            )