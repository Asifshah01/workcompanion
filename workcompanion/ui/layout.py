"""Sidebar layout and navigation.

The page roster is declared here once, so the sidebar, the route key and the
availability rules can never drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import state, theme

logger = get_logger(__name__)

PAGE_DASHBOARD = "dashboard"
PAGE_TUTOR = "tutor"
PAGE_KNOWLEDGE = "knowledge"
PAGE_RESEARCH = "research"
PAGE_QUIZ = "quiz"
PAGE_FLASHCARDS = "flashcards"
PAGE_EXAM = "exam"
PAGE_PLANNER = "planner"
PAGE_PROGRESS = "progress"
PAGE_SETTINGS = "settings"


@dataclass(frozen=True)
class Page:
    """One sidebar entry."""

    key: str
    label: str
    icon: str
    render: Callable[[Any], None]
    caption: str
    needs_documents: bool = False


def _pages() -> dict[str, Page]:
    from workcompanion.ui import pages

    return {
        page.key: page
        for page in (
            Page(
                PAGE_DASHBOARD,
                "Dashboard",
                "🏠",
                pages.render_dashboard,
                "Ask anything, or jump into a workflow",
            ),
            Page(
                PAGE_TUTOR,
                "AI Tutor",
                "🎓",
                pages.render_tutor,
                "Explanations and Socratic mode",
            ),
            Page(
                PAGE_KNOWLEDGE,
                "My Knowledge",
                "📚",
                pages.render_knowledge,
                "Upload, index and search your material",
            ),
            Page(
                PAGE_RESEARCH,
                "Research",
                "🔬",
                pages.render_research,
                "Paper analysis and multi-source synthesis",
                needs_documents=True,
            ),
            Page(PAGE_QUIZ, "Quiz", "📝", pages.render_quiz, "Grounded practice questions"),
            Page(
                PAGE_FLASHCARDS,
                "Flashcards",
                "🗂️",
                pages.render_flashcards,
                "Atomic revision decks",
            ),
            Page(
                PAGE_EXAM,
                "Exam Mode",
                "⏱️",
                pages.render_exam,
                "Timed papers with grading",
            ),
            Page(
                PAGE_PLANNER,
                "Study Planner",
                "🗓️",
                pages.render_planner,
                "Spaced-repetition schedules",
            ),
            Page(
                PAGE_PROGRESS,
                "Progress",
                "📈",
                pages.render_progress,
                "Mastery, weak areas, history",
            ),
            Page(PAGE_SETTINGS, "Settings", "⚙️", pages.render_settings, "Configuration and health"),
        )
    }


PAGES = _pages()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(context: state.AppContext) -> str:
    """Render the sidebar and return the selected page key."""
    with st.sidebar:
        theme.brand()

        has_documents = not context.bundle.pipeline.is_empty
        available = [
            page for page in PAGES.values() if has_documents or not page.needs_documents
        ]
        key_by_label = {page.label: page.key for page in available}

        # A page may have queued a destination (e.g. "plan revision" from the
        # quiz results); apply it to the widget before it is instantiated.
        pending = st.session_state.pop(state.KEY_NAVIGATE, None)
        if pending in key_by_label.values():
            st.session_state["wc_nav"] = next(
                label for label, key in key_by_label.items() if key == pending
            )
            st.session_state["wc_page"] = pending

        selected = st.radio(
            "Mode",
            list(key_by_label),
            label_visibility="collapsed",
            key="wc_nav",
        )
        st.session_state["wc_page"] = key_by_label[selected]

        st.divider()
        _render_status(context)
        _render_roster(context)
        _render_help()
    return str(st.session_state["wc_page"])


def _render_status(context: state.AppContext) -> None:
    settings = context.settings
    offline = context.offline
    theme.badge_row(
        [
            ("offline mode" if offline else settings.groq_model, "medium" if offline else "high", "🔌"),
            (f"{context.bundle.pipeline.chunk_count} chunks", "neutral", "📦"),
            (
                "web on" if context.bundle.web_search.enabled else "web off",
                "neutral" if context.bundle.web_search.enabled else "low",
                "🌐",
            ),
        ]
    )
    if offline:
        st.caption(
            "No `GROQ_API_KEY` found. Answers are extracted from your own documents; "
            "add the key to `.env` for model-written prose."
        )


def _render_roster(context: state.AppContext) -> None:
    with st.expander("Agent roster", expanded=False):
        for name, agent in context.bundle.agents().items():
            theme.agent_card(name, str(agent.description))


def _render_help() -> None:
    with st.expander("What can I ask?", expanded=False):
        st.markdown(
            """
- **Explain** – “explain the second law, I'm a beginner”
- **Answer** – “what does my notes say about Gibbs free energy?”
- **Search** – “find the passage on entropy production”
- **Quiz / Exam** – “quiz me on enthalpy, 8 hard questions”
- **Flashcards** – “make 15 cards for phase equilibrium”
- **Plan** – “plan 7 days at 2 h/day, I'm weak on entropy”
- **Socratic** – “guide me through this without giving the answer”
- **Research** – “latest work on lattice gauge theory”
- **Progress** – “how am I doing?”
"""
        )


def dispatch(context: state.AppContext, page_key: str) -> None:
    """Render the selected page."""
    page = PAGES.get(page_key) or PAGES[PAGE_DASHBOARD]
    if page.needs_documents and context.bundle.pipeline.is_empty:
        theme.hero(page.label, page.caption)
        st.info(
            "This mode works from your uploaded material. Add a document in "
            "**My Knowledge** first — answers are never invented when there is no evidence."
        )
        return
    logger.debug("[ui] rendering page %s", page.key)
    page.render(context)


__all__ = [
    "PAGE_DASHBOARD",
    "PAGE_SETTINGS",
    "PAGES",
    "Page",
    "dispatch",
    "render_sidebar",
]