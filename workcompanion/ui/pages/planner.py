"""Study Planner: constraints in, a spaced-repetition schedule out."""

from __future__ import annotations

from datetime import date as date_type
from datetime import timedelta
from typing import Any

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.planner import StudyPlan, StudyPlanInput
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)

KEY_PLAN = "study_plan"

ACTIVITY_OPTIONS = ["learn", "practice", "quiz", "flashcards", "revise", "mock_exam", "summary"]

#: The tutor uses a four-step scale; the planner uses a three-step one. Map
#: between them rather than duplicating the vocabulary in two schemas.
PLAN_LEVELS = ["beginner", "intermediate", "advanced"]
LEVEL_TO_PLAN = {
    "beginner": "beginner",
    "undergraduate": "intermediate",
    "graduate": "advanced",
    "expert": "advanced",
}


def plan_level(level: str) -> str:
    """Translate a tutor-level label into a planner-level label."""
    return LEVEL_TO_PLAN.get(level, "intermediate")


def render_planner(context: state.AppContext) -> None:
    """Collect constraints, build the plan, then adjust it around reality."""
    theme.hero(
        "Study Planner",
        "Weak areas first, revision on an expanding 1/3/7/14-day ladder, and every block "
        "ending in retrieval practice.",
    )
    plan = st.session_state.get(KEY_PLAN)
    if plan is not None:
        _render_plan(context, plan)
        return
    _render_form(context)


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------
def _render_form(context: state.AppContext) -> None:
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repositories, repositories.session)
        weak = [stat.topic for stat in tracker.weak_topics()]
        strong = [stat.topic for stat in tracker.strong_topics()]
        stats = {stat.topic for stat in tracker.topic_stats() if stat.attempts}
    profile = context.profile_snapshot()

    st.session_state.setdefault("plan_seed_topics", ", ".join(weak[:4]))

    left, right = st.columns(2, gap="medium")
    subject = left.text_input(
        "Subject", key="pl_subject", value=profile.default_subject or "", placeholder="Thermodynamics"
    )
    level = left.selectbox("Current level", PLAN_LEVELS, index=1, key="pl_level")
    topics = left.text_area(
        "Topics to cover",
        key="pl_topics",
        value=st.session_state.get("plan_seed_topics", "") or ", ".join(sorted(stats)[:6]),
        height=80,
    )
    weak_areas = left.text_area(
        "Weak areas (scheduled first)",
        key="pl_weak",
        value=", ".join(weak[:6]),
        height=70,
    )
    strong_areas = left.text_area(
        "Already strong (light touch)",
        key="pl_strong",
        value=", ".join(strong[:6]),
        height=70,
    )

    days = right.number_input(
        "Days available", min_value=1, max_value=90, value=7, step=1, key="pl_days"
    )
    hours = right.slider("Hours per day", 0.5, 8.0, 2.0, step=0.25, key="pl_hours")
    today = date_type.today()
    exam = right.date_input(
        "Exam date (optional)",
        value=today + timedelta(days=int(days)),
        key="pl_exam",
    )
    methods = right.multiselect(
        "Preferred methods", ACTIVITY_OPTIONS, default=["learn", "quiz", "flashcards"], key="pl_methods"
    )

    if weak:
        theme.hint(
            "Detected from your quiz history — weak areas are placed on the earliest days."
        )
    if not st.button("Build plan", key="pl_go", type="primary", disabled=not subject.strip()):
        return

    plan_input = StudyPlanInput(
        subject=subject.strip(),
        exam_date=exam,
        days_available=int(days),
        hours_per_day=float(hours),
        current_level=level,
        topics=_split(topics),
        weak_areas=_split(weak_areas),
        strong_areas=_split(strong_areas),
        preferred_methods=methods,
    )
    with st.spinner("Building the schedule…"):
        plan = context.bundle.planner.create_plan(plan_input)
    _persist(context, plan, plan_input)
    st.session_state[KEY_PLAN] = plan
    st.rerun()


def _split(raw: str) -> list[str]:
    topics: list[str] = []
    for line in (raw or "").splitlines():
        for part in line.split(","):
            cleaned = part.strip()
            if cleaned and cleaned.lower() not in {"not specified", "none"}:
                topics.append(cleaned)
    return topics


def _persist(context: state.AppContext, plan: StudyPlan, plan_input: StudyPlanInput) -> None:
    try:
        with context.session() as repositories:
            repositories.plans.save(repositories.session, plan)
    except Exception as exc:  # pragma: no cover - never lose the plan over persistence
        logger.warning("Could not save the study plan: %s", exc)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def _render_plan(context: state.AppContext, plan: StudyPlan) -> None:
    components.render_plan(plan)

    st.divider()
    st.markdown("**Reality check**")
    st.caption(
        "Tick the days you actually finished, or shrink the time available, and the "
        "remaining days are re-planned."
    )
    done = st.multiselect(
        "Completed days",
        options=[day.day_number for day in plan.days],
        default=[],
        key="pl_done",
        format_func=lambda number: f"Day {number}",
    )
    available = st.slider(
        "Hours I can actually study per day",
        0.25,
        max(0.5, float(plan.total_hours / max(1, plan.total_days)) + 1.0),
        value=round(plan.total_hours / max(1, plan.total_days), 2),
        step=0.25,
        key="pl_available",
    )
    new_exam = st.date_input(
        "New exam date (optional)",
        value=plan.exam_date or date_type.today(),
        key="pl_new_exam",
    )

    columns = st.columns([1, 1, 2], gap="small")
    if columns[0].button("Re-plan", key="pl_adjust", type="primary"):
        with st.spinner("Re-planning…"):
            adjusted = context.bundle.planner.adjust_plan(
                plan,
                completed_day_numbers=done,
                new_exam_date=new_exam if new_exam != (plan.exam_date or date_type.today()) else None,
                hours_per_day=float(available),
            )
        st.session_state[KEY_PLAN] = adjusted
        st.rerun()
    if columns[1].button("Start fresh", key="pl_reset"):
        st.session_state.pop(KEY_PLAN, None)
        st.rerun()
    if columns[2].button("Copy plan", key="pl_copy"):
        st.code(context.bundle.planner.render_plan(plan), language="markdown")

    if plan.revision_schedule:
        days = sorted(plan.revision_schedule)
        st.caption(
            f"Retrieval practice is scheduled on days "
            f"{', '.join(str(day) for day in days)} — "
            f"about {sum(plan.revision_schedule.values()) / max(1, plan.total_days):.0%} "
            "of the total time."
        )


__all__ = ["render_planner"]