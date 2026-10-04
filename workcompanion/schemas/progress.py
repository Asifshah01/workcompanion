"""Learning-progress schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TopicStat(BaseModel):
    """Per-topic mastery estimate for one subject."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic: str
    subject: str = "General"
    attempts: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_seconds: float = 0.0
    mastery: float = 0.0
    last_seen: datetime | None = None
    status: str = "unseen"

    @property
    def label_percent(self) -> str:
        return f"{self.mastery * 100:.0f}%"


class ProgressSnapshot(BaseModel):
    """Everything the dashboard needs, in one payload."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subjects: list[str] = Field(default_factory=list)
    topics: list[TopicStat] = Field(default_factory=list)
    overall_mastery: float = 0.0
    documents_indexed: int = 0
    documents_total: int = 0
    total_chunks: int = 0
    quiz_attempts: int = 0
    questions_answered: int = 0
    questions_correct: int = 0
    average_quiz_score: float = 0.0
    flashcards_generated: int = 0
    study_minutes: float = 0.0
    sessions_count: int = 0
    questions_asked: int = 0
    activity_by_day: dict[str, int] = Field(default_factory=dict)
    recent_scores: list[tuple[str, float]] = Field(default_factory=list)
    recommendation: str = ""
    recommendation_topics: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def strongest_topic(self) -> TopicStat | None:
        seen = [t for t in self.topics if t.attempts > 0]
        return max(seen, key=lambda t: t.mastery) if seen else None

    @property
    def weakest_topic(self) -> TopicStat | None:
        seen = [t for t in self.topics if t.attempts > 0]
        return min(seen, key=lambda t: t.mastery) if seen else None


class LearnerPreferences(BaseModel):
    """Non-sensitive learner preferences used to personalise answers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    preferred_level: str = "undergraduate"
    preferred_style: str = "detailed"
    default_subject: str | None = None
    weekly_study_hours: float = 2.0
    preferred_note_format: str = "markdown"
    show_citations_always: bool = True
    updated_at: date = Field(default_factory=date.today)

    def as_row(self) -> dict[str, Any]:
        return self.model_dump()