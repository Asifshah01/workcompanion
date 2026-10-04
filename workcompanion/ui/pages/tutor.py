"""AI Tutor: level-aware explanations plus the Socratic dialogue."""

from __future__ import annotations

from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import state, theme

logger = get_logger(__name__)

LEVEL_HELP = {
    "beginner": "No assumed knowledge; every term is defined.",
    "undergraduate": "Assumes first-year science background.",
    "graduate": "Assumes advanced disciplinary knowledge.",
    "expert": "Assumes specialist knowledge; focus on subtleties and edge cases.",
}

STYLE_HELP = {
    "concise": "The short answer, nothing else.",
    "detailed": "Full explanation with context.",
    "socratic": "Questions instead of answers.",
    "example_based": "Built around worked examples.",
    "mathematical": "Derivation and notation.",
    "analogy": "Anchored in everyday analogies.",
}


def render_tutor(context: state.AppContext) -> None:
    """Explanation controls, a free-form tutor and the Socratic mode tab."""
    theme.hero(
        "AI Tutor",
        "Explanations that adapt to your level, grounded in your own material where "
        "possible and honest about what it had to fill in from general knowledge.",
    )

    profile = context.profile_snapshot()
    level_options = list(LEVEL_HELP)
    style_options = list(STYLE_HELP)

    left, middle, right = st.columns(3, gap="medium")
    level = left.selectbox(
        "Level",
        level_options,
        index=level_options.index(profile.preferred_level)
        if profile.preferred_level in level_options
        else 1,
        key="tut_level",
        help=LEVEL_HELP.get(level_options[0], ""),
    )
    style = middle.selectbox(
        "Style",
        style_options,
        index=style_options.index(profile.preferred_style)
        if profile.preferred_style in style_options
        else 1,
        key="tut_style",
    )
    subject = right.text_input(
        "Subject (optional)", key="tut_subject", value=profile.default_subject or ""
    )
    st.caption(LEVEL_HELP.get(level, ""))

    tabs = st.tabs(["Teach me", "Socratic mode", "Check my understanding"])

    with tabs[0]:
        theme.hint(STYLE_HELP.get(style, ""))
        if st.button("Remember these preferences", key="tut_save"):
            state.set_preferences(
                context,
                preferred_level=level,
                preferred_style=style,
                default_subject=subject or None,
            )
            st.success("Preferences saved to your learner profile.")
        _render_chat(context, level=level, style=style, subject=subject)

    with tabs[1]:
        _render_socratic(context, level=level)

    with tabs[2]:
        _render_check(context, level=level, subject=subject)

    if profile.misconceptions:
        with st.expander(f"Recorded misconceptions ({len(profile.misconceptions)})", expanded=False):
            for misconception in profile.misconceptions:
                theme.warning(misconception)


def _render_chat(
    context: state.AppContext, *, level: str, style: str, subject: str | None
) -> None:
    from workcompanion.ui import chat as chat_ui

    chat_ui.chat(
        context,
        placeholder="What would you like explained?",
        level=level,
        style=style,
        topic=subject or None,
        key="tut_chat",
    )


def _render_socratic(context: state.AppContext, *, level: str) -> None:
    """The question-led dialogue, with explicit reveal and hint controls."""
    st.info(
        "The tutor asks one question at a time and escalates the scaffold (1→5). "
        "It will not reveal the answer until you ask it to."
    )
    turn_state = st.session_state.get(state.KEY_TURN) or {}
    if turn_state:
        theme.badge_row(
            [
                (f"scaffold {turn_state.get('scaffold_level', 0)}/5", "neutral", "🪜"),
                ("dialogue open", "neutral", "💬"),
            ]
        )
        columns = st.columns([1, 1, 3], gap="small")
        if columns[0].button("Reveal the answer", key="soc_reveal"):
            _socratic_send(context, "just tell me the answer", level=level)
            st.rerun()
        if columns[1].button("Give me a hint", key="soc_hint"):
            _socratic_send(context, "give me a hint", level=level)
            st.rerun()
        if columns[2].button("Start over", key="soc_reset"):
            st.session_state[state.KEY_TURN] = None
            state.clear_chat()
            st.rerun()

    from workcompanion.ui import chat as chat_ui

    chat_ui.chat(
        context,
        placeholder="Your attempt — or ask for a hint…",
        level=level,
        style="socratic",
        key="soc_chat",
    )


def _socratic_send(context: state.AppContext, message: str, *, level: str) -> None:
    """Run the Socratic handler directly for the reveal/hint buttons."""
    from workcompanion.ui import chat as chat_ui

    payload = chat_ui.handle_message(context, message, level=level, style="socratic")
    chat_ui.render_outcome(context, payload)


def _render_check(context: state.AppContext, *, level: str, subject: str | None) -> None:
    """Ask the learner's understanding to be checked on a specific claim."""
    st.caption(
        "Paste a claim from your material and the tutor will tell you whether it holds, "
        "and what a stronger version would look like."
    )
    claim = st.text_area(
        "Concept or claim",
        key="tut_claim",
        placeholder="Entropy always decreases in a spontaneous process.",
        height=90,
    )
    attempt = st.text_area(
        "Your explanation (optional — leave blank to get the check only)",
        key="tut_attempt",
        height=90,
    )
    expected = st.text_input(
        "What your material says (optional)", key="tut_expected", placeholder="Leave blank"
    )
    if not st.button("Check this", key="tut_check", type="primary", disabled=not claim.strip()):
        return
    with st.spinner("Checking…"):
        turn = context.bundle.tutor.check_understanding(
            claim.strip(), attempt.strip(), expected=expected.strip()
        )
    _render_turn(turn)
    for misconception in turn.misconceptions_detected or []:
        state.note_misconception(context, misconception)


def _render_turn(turn: Any) -> None:
    """Render a :class:`TutorTurn` (the Socratic / checking agent's output)."""
    st.markdown(turn.message)
    if turn.hint:
        theme.hint(f"**Hint:** {turn.hint}")
    if turn.reveal_answer:
        theme.badge_row([("answer revealed", "low", "🔓")])
    else:
        theme.badge_row([("still guiding", "neutral", "🔒")])
    if turn.misconceptions_detected:
        for misconception in turn.misconceptions_detected:
            theme.warning(f"Possible misconception: {misconception}")
    if turn.next_question:
        theme.hint(f"Next: {turn.next_question}")
    theme.badge_row([(f"scaffold {turn.scaffold_level}/5", "neutral", "🪜")])