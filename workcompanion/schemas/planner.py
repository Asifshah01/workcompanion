"""Study planner schemas."""

from __future__ import annotations

from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PlanActivity = Literal["learn", "revise", "quiz", "flashcards", "practice", "mock_exam", "rest"]
Priority = Literal["high", "medium", "low"]


class StudyPlanInput(BaseModel):
    """Learner-supplied constraints for plan generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str
    exam_date: date_type | None = None
    days_available: int | None = Field(default=None, ge=1, le=365)
    hours_per_day: float = Field(default=2.0, ge=0.25, le=16.0)
    current_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    topics: list[str] = Field(default_factory=list)
    weak_areas: list[str] = Field(default_factory=list)
    strong_areas: list[str] = Field(default_factory=list)
    preferred_methods: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _resolve_days(self) -> "StudyPlanInput":
        if self.days_available is None and self.exam_date is not None:
            remaining = (self.exam_date - date.today()).days
            self.days_available = max(1, min(remaining, 120))
        if self.days_available is None:
            self.days_available = 7
        return self


class StudyDay(BaseModel):
    """One day of the generated plan."""

    day_number: int
    day_date: date_type | None = None
    focus: str
    activities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    hours: float = 1.0
    priority: Priority = "medium"
    rationale: str = ""
    resources: list[str] = Field(default_factory=list)

    @property
    def minutes(self) -> int:
        return int(round(self.hours * 60))


class StudyPlan(BaseModel):
    """A complete, prioritised study schedule."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subject: str
    created_at_date: date_type = Field(default_factory=date_type.today)
    exam_date: date_type | None = None
    total_days: int = 1
    total_hours: float = 0.0
    days: list[StudyDay] = Field(default_factory=list)
    strategy_notes: list[str] = Field(default_factory=list)
    revision_schedule: dict[int, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_minutes(self) -> int:
        return int(round(self.total_hours * 60))