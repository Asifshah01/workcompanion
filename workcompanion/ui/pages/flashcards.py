"""Flashcards: atomic, citation-backed revision decks with export."""

from __future__ import annotations

from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.flashcards import FlashcardDeck
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)

KEY_DECK = "deck"


def render_flashcards(context: state.AppContext) -> None:
    """Build a deck on a topic or across several topics, then review/export it."""
    theme.hero(
        "Flashcards",
        "One idea per card, each traced back to a passage, so a card you cannot justify "
        "from your material is easy to spot.",
    )
    deck = st.session_state.get(KEY_DECK)
    if deck is not None:
        _render_deck(context, deck)
        return
    _render_builder(context)


def _render_builder(context: state.AppContext) -> None:
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        weak = [stat.topic for stat in ProgressTracker(repositories, repositories.session).weak_topics()]

    tabs = st.tabs(["One topic", "Multiple topics"])
    with tabs[0]:
        seed = st.session_state.pop("fc_seed", "")
        left, right = st.columns([3, 1], gap="medium")
        topic = left.text_input(
            "Topic", key="fc_topic", value=seed or "", placeholder="Enthalpy and phase changes"
        )
        count = right.slider("Cards", 4, 40, 12, key="fc_count")
        difficulty = right.select_slider(
            "Difficulty", ["easy", "medium", "hard"], value="medium", key="fc_diff"
        )
        document_filter = left.text_input(
            "Restrict to documents matching…", key="fc_filter", placeholder="(optional)"
        )
        if not st.button("Build deck", key="fc_go", type="primary", disabled=not topic.strip()):
            return
        with st.spinner(f"Building {count} card(s)…"):
            deck = context.bundle.flashcards.generate_deck(
                topic.strip(),
                count=count,
                difficulty=difficulty,
                document_filter=document_filter.strip() or None,
            )
        st.session_state[KEY_DECK] = deck
        st.rerun()

    with tabs[1]:
        st.caption("One deck covering several topics - useful for a revision session.")
        raw = st.text_area(
            "Topics (one per line, or comma separated)",
            key="fc_topics",
            value="\n".join(weak[:4]),
            height=120,
        )
        per_topic = st.slider("Cards per topic", 2, 15, 5, key="fc_per_topic")
        if not st.button("Build combined deck", key="fc_go_multi", type="primary"):
            return
        topics = _parse_topics(raw)
        if not topics:
            st.warning("Add at least one topic.")
            return
        with st.spinner("Building the combined deck…"):
            deck = context.bundle.flashcards.generate_from_topics(
                topics, cards_per_topic=per_topic
            )
        st.session_state[KEY_DECK] = deck
        st.rerun()


def _parse_topics(raw: str) -> list[str]:
    topics: list[str] = []
    for line in (raw or "").splitlines():
        for part in line.split(","):
            cleaned = part.strip()
            if cleaned:
                topics.append(cleaned)
    seen: set[str] = set()
    unique: list[str] = []
    for topic in topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            unique.append(topic)
    return unique


def _render_deck(context: state.AppContext, deck: FlashcardDeck) -> None:
    components.render_deck(deck, key_prefix="fc")
    components.render_warnings(deck.warnings)

    st.divider()
    columns = st.columns([1, 1, 2, 3], gap="small")
    if columns[0].button("New deck", key="fc_new", type="primary"):
        st.session_state.pop(KEY_DECK, None)
        st.rerun()
    if columns[1].button("Quiz me on this", key="fc_to_quiz"):
        st.session_state["quiz_seed"] = deck.subject or deck.title
        state.navigate("quiz")
        st.rerun()
    if columns[2].button("Copy as markdown", key="fc_copy"):
        st.code(_to_markdown(deck), language="markdown")
    with columns[3].popover("Export"):
        st.download_button(
            "Download deck (markdown)",
            _to_markdown(deck),
            file_name=f"{_slug(deck.title)}.md",
            mime="text/markdown",
            key="fc_dl",
        )
        st.caption("Markdown keeps the front/back structure and the citation markers.")


def _to_markdown(deck: FlashcardDeck) -> str:
    lines = [f"# {deck.title}", "", f"_{len(deck.cards)} card(s) · {deck.subject}_", ""]
    lines += ["---", ""]
    for card in deck.cards:
        lines.append(f"**Q: {card.front}**")
        lines.append("")
        lines.append(f"A: {card.back}")
        if card.citation_refs:
            lines.append("")
            lines.append(f"*Source: {', '.join(card.citation_refs)}*")
        lines += ["", "---", ""]
    return "\n".join(lines).strip() + "\n"


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in (text or "deck").lower()).strip("-")


__all__ = ["render_flashcards"]