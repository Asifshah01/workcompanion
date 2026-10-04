"""Streamlit pages for WorkCompanion AI.

Every page is a plain function taking an :class:`~workcompanion.ui.state.AppContext`
and rendering itself.  Keeping them free of Streamlit globals beyond what they
need makes them inspectable from tests and avoids hidden coupling.

Pages are plain functions rather than ``st.navigation`` targets so the sidebar
order, the icons and the availability rules live in one place
(:mod:`workcompanion.ui.layout`).
"""

from __future__ import annotations

from workcompanion.ui.pages.dashboard import render_dashboard
from workcompanion.ui.pages.knowledge import render_knowledge
from workcompanion.ui.pages.exam import render_exam
from workcompanion.ui.pages.flashcards import render_flashcards
from workcompanion.ui.pages.planner import render_planner
from workcompanion.ui.pages.progress import render_progress
from workcompanion.ui.pages.quiz import render_quiz
from workcompanion.ui.pages.research import render_research
from workcompanion.ui.pages.settings import render_settings
from workcompanion.ui.pages.tutor import render_tutor

__all__ = [
    "render_dashboard",
    "render_exam",
    "render_flashcards",
    "render_knowledge",
    "render_planner",
    "render_progress",
    "render_quiz",
    "render_research",
    "render_settings",
    "render_tutor",
]