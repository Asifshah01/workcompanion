"""Progress: mastery by topic, weak-area detection and activity history."""

from __future__ import annotations

import streamlit as st

from workcompanion.config.logging_config import get_logger
from workcompanion.ui import components, state, theme

logger = get_logger(__name__)


def render_progress(context: state.AppContext) -> None:
    """Everything the app knows about how the learner is doing."""
    theme.hero(
        "Progress",
        "Topic-level mastery drives what the planner schedules and what the quiz agent "
        "tests you on, so this page is the input to every other mode.",
    )

    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repositories, repositories.session)
        subjects = repositories.subjects_with_progress()
        subject = st.selectbox("Subject", ["(all)"] + list(subjects), key="pg_subject")
        chosen = None if subject == "(all)" else subject
        stats = tracker.topic_stats(chosen)
        weak = tracker.weak_topics(chosen)
        strong = tracker.strong_topics(chosen)
        mastery = tracker.overall_mastery(chosen)
        recommendation = tracker.recommend(chosen)
        snapshot = tracker.snapshot(chosen)
        activity = repositories.sessions.recent(repositories.session, limit=25)

    theme.stat_row(
        [
            ("Mastery", f"{mastery:.0f}%"),
            ("Questions", f"{snapshot.questions_answered}"),
            (
                "Accuracy",
                f"{snapshot.average_quiz_score:.0f}%" if snapshot.average_quiz_score else "—",
            ),
            ("Flashcards", f"{snapshot.flashcards_generated}"),
            ("Study time", f"{snapshot.study_minutes:.0f}m"),
            ("Sessions", f"{snapshot.sessions_count}"),
        ]
    )

    if not stats:
        st.info(
            "Nothing recorded yet. Take a quiz or answer a question and this page fills in. "
            "Weak areas are detected automatically and scheduled first."
        )
        return

    tabs = st.tabs(["Topics", "Recommendations", "Activity", "Learner profile"])

    with tabs[0]:
        _render_topics(stats, weak, strong)
    with tabs[1]:
        _render_recommendation(recommendation, weak)
    with tabs[2]:
        _render_activity(snapshot, activity)
    with tabs[3]:
        _render_profile(context, chosen)


def _render_topics(stats: list, weak: list, strong: list) -> None:
    weak_topics = {stat.topic for stat in weak}
    strong_topics = {stat.topic for stat in strong}
    rows = []
    for stat in stats:
        kind = (
            "weak"
            if stat.topic in weak_topics
            else "strong"
            if stat.topic in strong_topics
            else "medium"
        )
        rows.append(
            {
                "Topic": stat.topic,
                "Subject": stat.subject,
                "Attempts": stat.attempts,
                "Accuracy": f"{stat.accuracy:.0%}",
                "Mastery": f"{stat.mastery:.0%}",
                "Avg seconds": f"{stat.avg_seconds:.0f}",
                "Status": kind,
                "Last seen": str(stat.last_seen or ""),
            }
        )
    st.dataframe(sorted(rows, key=lambda row: float(row["Accuracy"].rstrip("%"))), hide_index=True, use_container_width=True)

    worst = sorted(stats, key=lambda stat: stat.accuracy)[:5]
    if worst:
        st.markdown("**Needs attention**")
        for stat in worst:
            theme.confidence_meter(
                float(stat.accuracy), theme.level_key(stat.accuracy * 100), label=stat.topic
            )
        topics = ", ".join(stat.topic for stat in worst)
        columns = st.columns(2, gap="small")
        if columns[0].button("Quiz me on these", key="pg_quiz", type="primary"):
            st.session_state["quiz_seed"] = topics
            state.navigate("quiz")
            st.rerun()
        if columns[1].button("Plan revision for these", key="pg_plan"):
            st.session_state["plan_seed_topics"] = topics
            state.navigate("planner")
            st.rerun()


def _render_recommendation(recommendation: object, weak: list) -> None:
    theme.kv(
        [
            ("Headline", recommendation.headline or "—"),
            ("Detail", recommendation.detail or "—"),
            ("Priority topics", ", ".join(recommendation.topics or []) or "—"),
            ("Next action", recommendation.action_hint or "—"),
            ("Priority", f"{recommendation.priority:.2f}"),
        ]
    )
    if weak:
        theme.hint("Detected weak areas: " + ", ".join(stat.topic for stat in weak))
    st.markdown(
        "Mastery blends accuracy with how recently you were tested, so a topic you were "
        "good at three weeks ago does not stay 'strong' for free."
    )


def _render_activity(snapshot: object, activity: list) -> None:
    if snapshot.activity_by_day:
        st.markdown("**Activity**")
        st.bar_chart(
            {str(day): int(count) for day, count in snapshot.activity_by_day.items()}
        )
    if snapshot.recent_scores:
        st.markdown("**Recent scores**")
        st.line_chart([float(score) for score in snapshot.recent_scores])
    if activity:
        st.markdown("**Recent sessions**")
        st.dataframe(
            [
                {
                    "Mode": row.mode or "-",
                    "Subject": row.subject or "-",
                    "Minutes": round(float(row.duration_minutes or 0), 1),
                    "Started": str(row.started_at or "")[:16],
                    "Ended": str(row.ended_at or "")[:16] if row.ended_at else "in progress",
                }
                for row in activity
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No sessions recorded yet.")


def _render_profile(context: state.AppContext, subject: str | None) -> None:
    profile = context.profile_snapshot(subject)
    theme.kv(
        [
            ("Preferred level", profile.preferred_level),
            ("Preferred style", profile.preferred_style),
            ("Default subject", profile.default_subject or "-"),
            ("Weekly study hours", profile.weekly_study_hours),
            ("Questions asked", profile.questions_asked),
            ("Strong topics", ", ".join(profile.strong_topics) or "-"),
            ("Weak topics", ", ".join(profile.weak_topics) or "-"),
            ("Known topics", ", ".join(profile.known_topics) or "-"),
        ]
    )
    if profile.misconceptions:
        st.markdown("**Misconceptions recorded**")
        for misconception in profile.misconceptions:
            theme.warning(misconception)
        if st.button("Clear misconception list", key="pg_clear_mis"):
            with context.session() as repositories:
                from workcompanion.memory.learner_profile import LearnerProfile

                LearnerProfile(repositories).clear_misconceptions()
            st.rerun()
    else:
        st.caption("No misconceptions recorded.")


__all__ = ["render_progress"]