"""Reusable renderers shared by every page.

The pages should read as *product* code, so anything that appears twice (an
answer with its sources, a confidence readout, a warning) lives here.  Every
renderer is defensive: it accepts ``None`` and partially-filled objects, because
agents legitimately return empty citations, empty grounding and zero confidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.rag.citations import citations_markdown
from workcompanion.schemas.common import AgentResult, Citation, ConfidenceLevel
from workcompanion.schemas.flashcards import FlashcardDeck
from workcompanion.schemas.planner import StudyPlan
from workcompanion.schemas.quiz import QuizSet
from workcompanion.ui import theme

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
def render_answer(result: AgentResult, *, expand_warnings: bool = True) -> None:
    """The answer text plus its provenance, confidence and warnings."""
    st.markdown(result.answer or "_No answer was produced._")

    badges: list[tuple[str, str, str]] = []
    if result.agent:
        icon = theme.AGENT_ICONS.get(str(result.agent), "🤖")
        badges.append((str(result.agent), "neutral", icon))
    if result.intent:
        badges.append((str(result.intent), "neutral", "🎯"))
    theme.badge_row(badges)
    theme.confidence_meter(result.confidence, result.confidence_level)
    theme.grounding_row(result.grounding)

    if result.metadata.get("offline"):
        theme.hint(
            "Assembled locally from your documents (no LLM API key configured). "
            "Add `GROQ_API_KEY` to `.env` for model-written prose."
        )

    render_sources(result.sources)
    if expand_warnings:
        render_warnings(result.warnings)
    render_trace_details(result)


def render_sources(sources: Sequence[Citation] | None, *, title: str = "Sources") -> None:
    """Numbered source cards with page, section and a quote."""
    items = [source for source in sources or [] if source]
    if not items:
        return
    with st.expander(f"{title} ({len(items)})", expanded=False):
        for source in items:
            location = []
            if source.page is not None:
                location.append(f"p. {source.page}")
            if source.section:
                location.append(source.section)
            if source.chapter:
                location.append(source.chapter)
            meta = " · ".join(location)
            relevance = f" · relevance {source.relevance:.2f}" if source.relevance else ""
            st.markdown(
                f"""
                <div class="wc-source">
                  <div class="wc-source-title">{source.marker} {source.document_name}</div>
                  <div class="wc-source-meta">{meta}{relevance}</div>
                  {f'<div class="wc-quote">&ldquo;{source.quote}&rdquo;</div>' if source.quote else ''}
                  {f'<div class="wc-source-meta">{source.url}</div>' if source.url else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_warnings(warnings: Sequence[str] | None) -> None:
    """Non-blocking notices (fallbacks taken, degraded paths, dropped items)."""
    items = [w for w in warnings or [] if w]
    if not items:
        return
    for warning in items[:6]:
        theme.warning(warning)
    if len(items) > 6:
        theme.hint(f"…and {len(items) - 6} more notices.")


def render_trace_details(result: AgentResult) -> None:
    """Latency, tokens, model and retrieval statistics behind the answer."""
    stats = result.metadata.get("retrieval")
    rows: list[tuple[str, Any]] = [
        ("Agent", result.agent or "-"),
        ("Latency", f"{result.latency_ms:.0f} ms"),
    ]
    if result.model:
        rows.append(("Model", result.model))
    if result.prompt_tokens or result.completion_tokens:
        rows.append(
            ("Tokens", f"{result.prompt_tokens or 0} in / {result.completion_tokens or 0} out")
        )
    if isinstance(stats, dict):
        rows.extend(
            [
                ("Candidates", f"{stats.get('fused_candidates', 0)} fused"),
                (
                    "Retrieval",
                    f"dense {'✓' if stats.get('used_dense') else '✗'} · "
                    f"sparse {'✓' if stats.get('used_sparse') else '✗'} · "
                    f"reranker {stats.get('used_reranker', 'none')}",
                ),
                ("Pipeline", f"{stats.get('total_latency_ms', 0):.0f} ms total"),
            ]
        )
    extras = {
        key: value
        for key, value in (result.metadata or {}).items()
        if key not in ("retrieval", "offline")
        and isinstance(value, (str, int, float, bool))
    }
    rows.extend((key.replace("_", " ").title(), value) for key, value in extras.items())
    with st.expander("Provenance & trace", expanded=False):
        theme.kv(rows)
        if stats is not None and not isinstance(stats, dict):
            st.json(stats)


def render_confidence_banner(result: AgentResult) -> None:
    """A prominent low-confidence warning, used above answers that were rewritten."""
    if result.confidence_level is ConfidenceLevel.LOW:
        theme.warning(
            "Low confidence: the indexed material only partially supports this answer. "
            "Check the sources, or upload material that covers the topic."
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def render_retrieval(outcome: Any, *, show_chunks: bool = True) -> None:
    """Show what retrieval found: queries, confidence, passages."""
    if outcome is None:
        return
    query = outcome.query
    badges = [
        ("confidence", theme.level_key(outcome.confidence_level), "🎯"),
        (f"{len(outcome.chunks)} passages", "neutral", "📄"),
    ]
    if outcome.stats.used_multi_query:
        badges.append(("multi-query", "neutral", "🔀"))
    if outcome.stats.compressed:
        badges.append(("compressed", "neutral", "🗜️"))
    theme.badge_row(badges)
    theme.confidence_meter(outcome.confidence, outcome.confidence_level, label="retrieval")

    with st.expander("How retrieval worked", expanded=False):
        theme.kv(
            [
                ("Original query", query.original),
                ("Queries issued", " · ".join(query.queries) or "-"),
                ("Standalone query", query.standalone or "-"),
                ("Keywords", ", ".join(query.keywords[:12]) or "-"),
                ("Conversation resolved", "yes" if query.conversation_resolved else "no"),
                ("Reranker", outcome.stats.used_reranker),
                ("Compressed", "yes" if outcome.stats.compressed else "no"),
                ("Latency", f"{outcome.stats.total_latency_ms:.0f} ms"),
            ]
        )

    if show_chunks and outcome.chunks:
        with st.expander(f"Passages ({len(outcome.chunks)})", expanded=False):
            for index, chunk in enumerate(outcome.chunks, start=1):
                meta = chunk.metadata
                location = " · ".join(
                    bit
                    for bit in (
                        meta.document_name,
                        f"p. {meta.page}" if meta.page else "",
                        meta.section or "",
                    )
                    if bit
                )
                st.markdown(
                    f"**[{index}]** {location} `score={chunk.score:.3f}`\n\n"
                    f"{chunk.chunk.text[:600]}"
                )


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------
def render_quiz(quiz: QuizSet, *, key_prefix: str = "quiz") -> None:
    """Render a quiz as interactive radio questions with instant feedback."""
    st.markdown(f"### {quiz.title}")
    meta = [
        f"**{quiz.difficulty}**",
        f"{len(quiz.questions)} question(s)",
        f"{quiz.total_points} point(s)",
    ]
    if quiz.timed and quiz.duration_minutes:
        meta.append(f"⏱ {quiz.duration_minutes} min")
    theme.hint(" · ".join(meta))

    answers: dict[str, str] = {}
    for index, question in enumerate(quiz.questions, start=1):
        with st.container(border=True):
            points = f" ({question.points} pt)" if question.points != 1 else ""
            st.markdown(f"**{index}. {question.question}**{points}")
            if question.topic:
                theme.hint(f"Topic: {question.topic} · type: {question.question_type}")
            if question.options:
                answers[question.id] = st.radio(
                    "Select one",
                    options=list(question.options),
                    key=f"{key_prefix}_{question.id}",
                    label_visibility="collapsed",
                )
            else:
                answers[question.id] = st.text_input(
                    "Your answer",
                    key=f"{key_prefix}_{question.id}",
                    label_visibility="collapsed",
                )
            if question.hint:
                with st.popover("Hint"):
                    st.write(question.hint)
            if question.citation_refs:
                theme.hint(f"Grounded in {', '.join(question.citation_refs)}")
    st.session_state[f"{key_prefix}_answers"] = answers
    render_sources(quiz.citations)


def render_grading(
    quiz: QuizSet,
    report: Any,
    *,
    answer_key: str,
) -> None:
    """Show per-question feedback plus the overall report."""
    by_id = {question.id: question for question in quiz.questions}
    for evaluation in report.evaluations:
        question = by_id.get(evaluation.question_id)
        with st.container(border=True):
            icon = "✅" if evaluation.is_correct else "❌"
            st.markdown(f"{icon} **{evaluation.feedback}**")
            if question is not None:
                st.markdown(f"*Q: {question.question}*")
            st.markdown(f"**Your answer:** {evaluation.learner_answer}")
            st.markdown(f"**Expected:** {evaluation.expected_answer}")
            if evaluation.misconception:
                theme.warning(f"Likely misconception: {evaluation.misconception}")
            if evaluation.next_hint:
                theme.hint(f"Next step: {evaluation.next_hint}")

    theme.stat_row(
        [
            ("Score", f"{report.score_percent:.0f}%"),
            ("Correct", f"{report.correct}/{report.attempted}"),
            ("Points", f"{report.points_earned:.0f}/{report.points_possible:.0f}"),
            ("Duration", f"{report.duration_seconds:.0f}s"),
        ]
    )
    if report.topic_breakdown:
        with st.expander("Topic breakdown", expanded=True):
            for topic, score in sorted(
                report.topic_breakdown.items(), key=lambda item: item[1]
            ):
                theme.confidence_meter(score / 100.0, theme.level_key(score), label=topic)
    if report.misconceptions:
        with st.expander("Misconceptions detected", expanded=True):
            for misconception in report.misconceptions:
                theme.warning(misconception)
    if report.recommendations:
        with st.expander("What to do next", expanded=True):
            for recommendation in report.recommendations:
                st.markdown(f"- {recommendation}")
    render_sources(quiz.citations, title="Quiz sources")


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------
def render_deck(deck: FlashcardDeck, *, key_prefix: str = "deck") -> None:
    """A flip-card deck: front always visible, back behind a toggle."""
    st.markdown(f"### {deck.title}")
    theme.hint(f"{len(deck.cards)} card(s) · {deck.subject}")
    for index, card in enumerate(deck.cards, start=1):
        with st.container(border=True):
            st.markdown(
                f'<div class="wc-card"><div class="wc-card-front">'
                f"{index}. {card.front}</div></div>",
                unsafe_allow_html=True,
            )
            show = st.toggle(
                "Reveal answer", key=f"{key_prefix}_{card.id}", value=False
            )
            if show:
                st.markdown(
                    f'<div class="wc-card-back">{card.back}</div>',
                    unsafe_allow_html=True,
                )
            if card.citation_refs:
                theme.hint(f"From {', '.join(card.citation_refs)}")
            if card.hint:
                theme.hint(f"Hint: {card.hint}")
    render_sources(deck.citations)


# ---------------------------------------------------------------------------
# Study plan
# ---------------------------------------------------------------------------
def render_plan(plan: StudyPlan) -> None:
    """The day-by-day schedule as a table plus per-day detail."""
    st.markdown(f"### {plan.subject}")
    summary = [f"{plan.total_days} day(s)", f"{plan.total_hours:.1f} h total"]
    if plan.exam_date:
        summary.append(f"exam {plan.exam_date}")
    if plan.created_at_date:
        summary.append(f"built {plan.created_at_date}")
    theme.hint(" · ".join(summary))

    st.dataframe(
        [
            {
                "Day": day.day_number,
                "Date": str(day.day_date or ""),
                "Focus": day.focus,
                "Activities": ", ".join(day.activities),
                "Hours": round(float(day.hours), 2),
                "Priority": day.priority,
            }
            for day in plan.days
        ],
        hide_index=True,
        use_container_width=True,
    )

    for day in plan.days:
        with st.expander(f"Day {day.day_number} · {day.focus} ({day.hours:g} h)"):
            st.markdown(f"**Why this slot:** {day.rationale}")
            if day.activities:
                st.markdown("**Activities:** " + ", ".join(day.activities))
            if day.topics:
                theme.hint("Topics: " + ", ".join(day.topics))
            if day.resources:
                for resource in day.resources:
                    st.markdown(f"- {resource}")
    if plan.strategy_notes:
        with st.expander("Strategy notes", expanded=True):
            for note in plan.strategy_notes:
                st.markdown(f"- {note}")
    if plan.revision_schedule:
        # ``revision_schedule`` maps day number -> share of that day spent on
        # retrieval practice, not a topic index.
        with st.expander("Revision schedule", expanded=False):
            for day_number, share in sorted(plan.revision_schedule.items()):
                st.markdown(f"- **Day {day_number}** — {share:.0%} of the session")
    render_warnings(plan.warnings)


# ---------------------------------------------------------------------------
# Chat transcript
# ---------------------------------------------------------------------------
def render_transcript(entries: list[dict[str, Any]], *, render_sources_too: bool = True) -> None:
    """Replay the in-session transcript."""
    for entry in entries:
        with st.chat_message(entry.get("role", "assistant")):
            st.markdown(entry.get("content", ""))
            sources = entry.get("sources") or []
            confidence = entry.get("confidence")
            if confidence is not None:
                theme.confidence_meter(
                    float(confidence), ConfidenceLevel.from_score(float(confidence))
                )
            if sources and render_sources_too:
                render_sources(sources)


def follow_up_buttons(options: Sequence[str], *, key: str) -> str | None:
    """Suggested replies; returns the one the learner pressed, if any."""
    options = [option for option in options or [] if option]
    if not options:
        return None
    columns = st.columns(min(3, len(options)), gap="small")
    for column, option in zip(columns, options):
        if column.button(option, key=f"{key}_{option}"):
            return option
    return None


def sources_markdown(citations: Sequence[Citation]) -> str:
    """Markdown source block (used when copying results out of the app)."""
    return citations_markdown(citations)


__all__ = [
    "follow_up_buttons",
    "render_answer",
    "render_confidence_banner",
    "render_deck",
    "render_grading",
    "render_plan",
    "render_quiz",
    "render_retrieval",
    "render_sources",
    "render_trace_details",
    "render_transcript",
    "render_warnings",
    "sources_markdown",
]