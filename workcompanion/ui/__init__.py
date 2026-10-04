"""Streamlit UI for WorkCompanion AI.

Layered so each piece is independently testable:

``theme``       - CSS, palette and small visual primitives
``state``       - bootstrap, per-rerun context and session-safe helpers
``components``  - shared renderers (answers, sources, quizzes, decks, plans)
``chat``        - the NEXUS-routed dispatch used by every conversational page
``layout``      - the sidebar and the page roster
``pages``       - one function per mode

Nothing in this package is imported at module scope by the agents, so the
library stays usable headless (CLI scripts, tests, cron jobs).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"