"""WorkCompanion AI - entry point.

Run with::

    streamlit run app.py

The script is deliberately thin: it configures the page, boots the shared
context once, and dispatches to the selected mode.  Everything else lives in
:mod:`workcompanion.ui`, which means the UI can be exercised from tests without
starting a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``streamlit run app.py`` from a clone that was never pip-installed.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from workcompanion.config.logging_config import get_logger  # noqa: E402
from workcompanion.ui import layout, state, theme  # noqa: E402

logger = get_logger(__name__)

PAGE_TITLE = "WorkCompanion AI"
PAGE_ICON = "🎓"


def main() -> None:
    """Configure the page, boot the context and render the selected mode."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"about": "Multi-agent study and research companion"},
    )

    context = state.bootstrap()
    theme.inject_css()

    page_key = layout.render_sidebar(context)
    layout.dispatch(context, page_key)

    with st.sidebar:
        st.divider()
        st.caption("WorkCompanion AI · v1.0")
        st.caption(
            f"{context.bundle.pipeline.chunk_count:,} passages indexed · "
            f"{'offline' if context.offline else context.settings.groq_model}"
        )


if __name__ == "__main__":
    main()