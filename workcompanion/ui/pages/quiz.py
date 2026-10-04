"""Quiz: grounded question generation, live answering and grading."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)

KEY_QUIZ = "quiz_set"
KEY_STARTS = "quiz_started"
KEY_REPORT = "quiz_report"
KEY_GRADE = "quiz_graded"


def render_quiz(context: state.AppContext) -> None:
    """Build a quiz from a topic, answer it, then grade and record the attempt."""
    theme.hero(
        "Quiz",
        "Every question is generated from your own passages and carries a citation, "
        "so you can check the source of anything you get wrong.",
    )

    if st.session_state.get(KEY_REPORT) and st.session_state.get(KEY_QUIZ):
        _render_results(context)
        return

    quiz = st.session_state.get(KEY_QUIZ)
    if quiz is not None:
        _render_attempt(context, quiz)
        return

    _render_builder(context)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _render_builder(context: state.AppContext) -> None:
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repositories, repositories.session)
        weak = [stat.topic for stat in tracker.weak_topics()]
        strong = [stat.topic for stat in tracker.strong_topics()]
        stats = tracker.topic_stats()
    suggestions = [stat.topic for stat in stats if stat.attempts]

    seed = st.session_state.pop("quiz_seed", "")
    left, right = st.columns([3, 1], gap="medium")
    topic = left.text_input(
        "Topic", key="qz_topic", value=seed or "", placeholder="Entropy and the second law"
    )
    count = right.slider("Questions", 3, 20, 6, key="qz_count")
    difficulty = right.select_slider(
        "Difficulty", ["easy", "medium", "hard", "expert"], value="medium", key="qz_diff"
    )
    document_filter = left.text_input(
        "Restrict to documents matching…", key="qz_filter", placeholder="(optional)"
    )

    if weak:
        theme.hint(
            "Weak areas: "
            + ", ".join(weak[:6])
            + ("  ·  strong: " + ", ".join(strong[:4]) if strong else "")
        )
        if st.button("Quiz me on my weakest areas", key="qz_weak"):
            _targeted(context, weak)
            return
    if suggestions:
        theme.hint("Seen topics: " + ", ".join(suggestions[:6]))

    if not st.button("Generate quiz", key="qz_go", type="primary", disabled=not topic.strip()):
        return
    with st.spinner(f"Generating {count} question(s)…"):
        quiz = context.bundle.quiz.generate_quiz(
            topic.strip(),
            num_questions=count,
            difficulty=difficulty,
            document_filter=document_filter.strip() or None,
        )
    st.session_state[KEY_QUIZ] = quiz
    st.rerun()


def _targeted(context: state.AppContext, weak: list[str]) -> None:
    if not weak:
        st.info("No weak areas detected yet — take a quiz first.")
        return
    subject = st.session_state.get("qz_topic") or weak[0]
    with st.spinner("Targeting your weak areas…"):
        quiz = context.bundle.quiz.generate_targeted_quiz(subject, weak, num_questions=8)
    st.session_state[KEY_QUIZ] = quiz
    st.rerun()


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------
def _render_attempt(context: state.AppContext, quiz: Any) -> None:
    st.session_state[KEY_STARTS] = st.session_state.get(KEY_STARTS) or time.time()
    components.render_quiz(quiz, key_prefix="qz")

    columns = st.columns([1, 1, 4], gap="small")
    if columns[0].button("Submit answers", key="qz_submit", type="primary"):
        _grade(context, quiz)
        st.rerun()
    if columns[1].button("Regenerate", key="qz_regen"):
        st.session_state.pop(KEY_QUIZ, None)
        st.session_state.pop(KEY_STARTS, None)
        st.rerun()
    unanswered = sum(1 for value in st.session_state.get("qz_answers", {}).values() if not value)
    columns[2].caption(f"{unanswered} question(s) unanswered — they will be marked wrong.")


def _grade(context: state.AppContext, quiz: Any) -> None:
    answers = {
        key: value
        for key, value in st.session_state.get("qz_answers", {}).items()
        if key in {question.id for question in quiz.questions}
    }
    if not answers:
        st.warning("Answer at least one question first.")
        return
    elapsed = time.time() - float(st.session_state.get(KEY_STARTS) or time.time())
    per_question = {key: elapsed / max(1, len(answers)) for key in answers}
    with st.spinner("Grading…"):
        report = context.bundle.quiz.grade(
            quiz, answers, seconds_per_question=per_question, duration_seconds=elapsed
        )
    st.session_state[KEY_REPORT] = report
    state.record_answers(
        context,
        [(evaluation.topic, evaluation.is_correct) for evaluation in report.evaluations],
        subject=quiz.subject or "General",
    )
    for evaluation in report.evaluations:
        if evaluation.misconception and not evaluation.is_correct:
            state.note_misconception(context, evaluation.misconception)
    logger.info(
        "[ui] quiz graded: %.0f%% over %d question(s)", report.score_percent, report.attempted
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _render_results(context: state.AppContext) -> None:
    quiz = st.session_state[KEY_QUIZ]
    report = st.session_state[KEY_REPORT]
    st.markdown(f"### {quiz.title}")
    components.render_grading(quiz, report, answer_key="qz")

    if report.recommendations:
        left, right = st.columns(2, gap="small")
        if left.button("Flashcards on my weak spots", key="qz_to_cards", type="primary"):
            topics = report.weak_topics or quiz.subject or quiz.title
            st.session_state["fc_seed"] = ", ".join(str(t) for t in topics)
            state.navigate("flashcards")
            st.rerun()
        if right.button("Plan revision for these gaps", key="qz_to_plan"):
            st.session_state["plan_seed_topics"] = ", ".join(
                str(t) for t in (report.weak_topics or [quiz.subject])
            )
            state.navigate("planner")
            st.rerun()

    st.divider()
    columns = st.columns(3, gap="small")
    if columns[0].button("New quiz", key="qz_again"):
        _reset()
        st.rerun()
    if columns[1].button("Harder version", key="qz_harder"):
        _reset()
        st.session_state["qz_diff"] = "hard"
        st.session_state["qz_topic"] = quiz.subject or quiz.title
        st.rerun()
    if columns[2].button("Copy report", key="qz_copy"):
        st.code(context.bundle.quiz.render_report(report), language="markdown")


def _reset() -> None:
    for key in (KEY_QUIZ, KEY_REPORT, KEY_STARTS):
        st.session_state.pop(key, None)


__all__ = ["render_quiz"]