"""Exam Mode: a timed paper with a countdown and a full mark report."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)

KEY_EXAM = "exam_set"
KEY_START = "exam_started"
KEY_REPORT = "exam_report"

COUNTDOWN_KEY = "exam_countdown"


def render_exam(context: state.AppContext) -> None:
    """Set up an exam, run it under a timer, then grade and record it."""
    theme.hero(
        "Exam Mode",
        "A timed paper built from your material, weighted toward your weak topics, "
        "graded question by question.",
    )
    if st.session_state.get(KEY_REPORT) and st.session_state.get(KEY_EXAM):
        _render_results(context)
        return
    exam = st.session_state.get(KEY_EXAM)
    if exam is not None:
        _render_exam(context, exam)
        return
    _render_setup(context)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def _render_setup(context: state.AppContext) -> None:
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        weak = [stat.topic for stat in ProgressTracker(repositories, repositories.session).weak_topics()]

    left, middle, right = st.columns([3, 1, 1], gap="medium")
    subject = left.text_input(
        "Subject", key="ex_subject", value="", placeholder="Thermodynamics"
    )
    questions = middle.slider("Questions", 5, 30, 12, key="ex_count")
    minutes = middle.slider("Minutes", 5, 120, 30, key="ex_minutes")
    difficulty = right.select_slider(
        "Difficulty", ["easy", "medium", "hard", "expert"], value="hard", key="ex_diff"
    )

    theme.hint(
        "The exam is weighted toward your weak topics"
        + (f": {', '.join(weak[:5])}" if weak else " (none detected yet).")
    )
    if not st.button("Start exam", key="ex_go", type="primary", disabled=not subject.strip()):
        return
    with st.spinner("Building the paper…"):
        exam = context.bundle.quiz.build_exam(
            subject.strip(),
            num_questions=questions,
            duration_minutes=minutes,
            difficulty=difficulty,
            weak_topics=weak or None,
        )
    st.session_state[KEY_EXAM] = exam
    st.session_state[KEY_START] = time.time()
    st.rerun()


# ---------------------------------------------------------------------------
# Paper in progress
# ---------------------------------------------------------------------------
def _render_exam(context: state.AppContext, exam: Any) -> None:
    started = float(st.session_state.get(KEY_START) or time.time())
    limit = float(exam.duration_minutes or 30) * 60
    remaining = max(0.0, limit - (time.time() - started))
    _countdown(remaining, limit)

    theme.badge_row(
        [
            (f"{exam.title}", "neutral", "📝"),
            (f"{len(exam.questions)} questions", "neutral", "❓"),
            (f"{exam.total_points} points", "neutral", "🎯"),
            (f"{exam.difficulty}", "neutral", "📶"),
        ]
    )
    if exam.warnings:
        components.render_warnings(exam.warnings)
    st.caption("Your answers are marked when you submit. Nothing is saved until then.")

    components.render_quiz(exam, key_prefix="ex")

    columns = st.columns([1, 1, 3], gap="small")
    if columns[0].button("Submit exam", key="ex_submit", type="primary"):
        _grade(context, exam, started)
        st.rerun()
    if columns[1].button("Abandon", key="ex_quit"):
        _reset()
        st.rerun()
    answered = sum(1 for value in st.session_state.get("ex_answers", {}).values() if value)
    columns[2].caption(
        f"{answered}/{len(exam.questions)} answered · "
        + ("time is up — submit to see your mark." if remaining <= 0 else "keep going.")
    )


def _countdown(remaining: float, limit: float) -> None:
    """A progress bar used as the clock; Streamlit reruns keep it live."""
    minutes, seconds = divmod(int(remaining), 60)
    label = f"⏱ {minutes:02d}:{seconds:02d} remaining"
    colour = "normal" if remaining > limit / 5 else "inverse"
    st.progress(max(0.0, min(1.0, remaining / limit)), text=label)
    st.session_state[COUNTDOWN_KEY] = label
    st.caption(
        "The clock updates every time you interact with the page."
        if colour == "normal"
        else "Almost out of time — submit now."
    )
    if remaining <= 0:
        theme.warning("Time is up. Submit to see your mark.")


def _grade(context: state.AppContext, exam: Any, started: float) -> None:
    answers = {
        key: value
        for key, value in st.session_state.get("ex_answers", {}).items()
        if key in {question.id for question in exam.questions}
    }
    if not answers:
        st.warning("Nothing was answered.")
        return
    elapsed = time.time() - started
    per_question = {key: elapsed / max(1, len(answers)) for key in answers}
    with st.spinner("Marking…"):
        report = context.bundle.quiz.grade(
            exam, answers, seconds_per_question=per_question, duration_seconds=elapsed
        )
    st.session_state[KEY_REPORT] = report
    state.record_answers(
        context,
        [(evaluation.topic, evaluation.is_correct) for evaluation in report.evaluations],
        subject=exam.subject or "General",
    )
    logger.info("[ui] exam marked: %.0f%% in %.0fs", report.score_percent, elapsed)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _render_results(context: state.AppContext) -> None:
    exam = st.session_state[KEY_EXAM]
    report = st.session_state[KEY_REPORT]
    st.markdown(f"### {exam.title} — mark sheet")
    components.render_grading(exam, report, answer_key="ex")

    with st.expander("Mark scheme", expanded=False):
        for index, question in enumerate(exam.questions, start=1):
            st.markdown(f"**{index}. {question.question}**")
            st.markdown(f"- Answer: **{question.correct_answer}**")
            if question.explanation:
                st.markdown(f"- Why: {question.explanation}")
            if question.citation_refs:
                st.markdown(f"- Source: {', '.join(question.citation_refs)}")

    st.divider()
    columns = st.columns(3, gap="small")
    if columns[0].button("New exam", key="ex_again", type="primary"):
        _reset()
        st.rerun()
    if columns[1].button("Revise weak spots", key="ex_to_cards"):
        topics = report.weak_topics or [exam.subject]
        st.session_state["fc_seed"] = ", ".join(str(topic) for topic in topics)
        state.navigate("flashcards")
        st.rerun()
    if columns[2].button("Copy mark sheet", key="ex_copy"):
        st.code(context.bundle.quiz.render_report(report), language="markdown")


def _reset() -> None:
    for key in (KEY_EXAM, KEY_REPORT, KEY_START, COUNTDOWN_KEY):
        st.session_state.pop(key, None)


__all__ = ["render_exam"]