"""Learner profile: preferences, misconceptions and study context.

Deliberately stores only what improves teaching - preferences, topic mastery
signals and observed misconceptions.  No names, emails, identifiers or any other
sensitive personal data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from workcompanion.config.logging_config import get_logger
from workcompanion.database.repositories import RepositoryBundle

logger = get_logger(__name__)

#: Max misconception strings retained (oldest are dropped).
MAX_MISCONCEPTIONS = 25
MAX_KNOWN_TOPICS = 60


@dataclass(slots=True)
class LearnerProfileSnapshot:
    """In-memory view of the learner, used to build agent prompts."""

    preferred_level: str = "undergraduate"
    preferred_style: str = "detailed"
    default_subject: str | None = None
    weekly_study_hours: float = 2.0
    strong_topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    known_topics: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    questions_asked: int = 0
    average_score: float = 0.0
    study_minutes: float = 0.0

    def prompt_context(self) -> str:
        """A compact, prompt-friendly description of the learner."""
        lines: list[str] = []
        if self.default_subject:
            lines.append(f"Subject in focus: {self.default_subject}")
        lines.append(f"Preferred explanation level: {self.preferred_level}")
        lines.append(f"Preferred teaching style: {self.preferred_style}")
        if self.weak_topics:
            lines.append(f"Weak areas (needs extra care): {', '.join(self.weak_topics[:5])}")
        if self.strong_topics:
            lines.append(f"Strong areas (can go faster): {', '.join(self.strong_topics[:5])}")
        if self.misconceptions:
            lines.append(
                "Recurring misconceptions to address proactively: "
                + "; ".join(self.misconceptions[:4])
            )
        if self.average_score:
            lines.append(f"Recent quiz average: {self.average_score:.0f}%")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not (self.weak_topics or self.strong_topics or self.misconceptions or self.known_topics)


class LearnerProfile:
    """Read/write access to the persistent learner profile."""

    def __init__(self, repos: RepositoryBundle):
        self._repos = repos
        self._session: Session = repos.session

    # ------------------------------------------------------------------
    def set_preferences(
        self,
        *,
        level: str | None = None,
        style: str | None = None,
        subject: str | None = None,
        weekly_hours: float | None = None,
        show_citations: bool | None = None,
    ) -> None:
        """Update the learner's teaching preferences."""
        self._repos.users.update_preferences(
            self._session,
            preferred_level=level,
            preferred_style=style,
            default_subject=subject,
            weekly_study_hours=weekly_hours,
            show_citations_always=show_citations,
        )

    # ------------------------------------------------------------------
    def snapshot(self, subject: str | None = None) -> LearnerProfileSnapshot:
        """Build the current profile snapshot."""
        user = self._repos.users.get_or_create(self._session)
        strong = self._repos.topics.strong_topics(self._session, limit=5, subject=subject)
        weak = self._repos.topics.weak_topics(self._session, limit=5, subject=subject)
        known = self._repos.topics.known_topics(self._session, subject=subject)
        misconceptions = self.get_misconceptions()
        stats = self._repos.quizzes.stats(self._session)

        return LearnerProfileSnapshot(
            preferred_level=user.preferred_level or "undergraduate",
            preferred_style=user.preferred_style or "detailed",
            default_subject=user.default_subject,
            weekly_study_hours=float(user.weekly_study_hours or 2.0),
            strong_topics=[row.topic for row in strong if row.attempts > 0],
            weak_topics=[row.topic for row in weak if row.attempts > 0],
            known_topics=known[:MAX_KNOWN_TOPICS],
            misconceptions=misconceptions,
            subjects=self._repos.subjects_with_progress(),
            questions_asked=int(stats["questions_answered"]),
            average_score=float(stats["average_score"]),
            study_minutes=self._repos.sessions.total_minutes(self._session),
        )

    # ------------------------------------------------------------------
    def record_misconception(self, misconception: str) -> None:
        """Remember a recurring misconception so the tutor can address it again."""
        text = (misconception or "").strip()
        if len(text) < 8:
            return
        user = self._repos.users.get_or_create(self._session)
        prefs = dict(user.preferences or {})
        items = list(prefs.get("misconceptions") or [])
        if text not in items:
            items.append(text[:300])
        prefs["misconceptions"] = items[-MAX_MISCONCEPTIONS:]
        prefs["misconceptions_updated_at"] = datetime.now(timezone.utc).isoformat()
        user.preferences = prefs
        self._session.flush()

    def get_misconceptions(self) -> list[str]:
        """Stored misconceptions, most recent last."""
        user = self._repos.users.get_or_create(self._session)
        prefs = user.preferences or {}
        return [str(item) for item in (prefs.get("misconceptions") or [])][-MAX_MISCONCEPTIONS:]

    def clear_misconceptions(self) -> None:
        user = self._repos.users.get_or_create(self._session)
        prefs = dict(user.preferences or {})
        prefs["misconceptions"] = []
        user.preferences = prefs
        self._session.flush()

    # ------------------------------------------------------------------
    def note_session(
        self, *, mode: str = "chat", subject: str | None = None, minutes: float | None = None
    ) -> None:
        """Record a completed study session."""
        record = self._repos.sessions.start(self._session, mode=mode, subject=subject)
        if minutes:
            record.duration_minutes = float(minutes)
        self._repos.sessions.finish(self._session, record)