"""Quiz / assessment schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workcompanion.schemas.common import Citation

QuestionType = Literal[
    "mcq", "true_false", "short_answer", "long_answer", "numerical", "viva", "conceptual"
]
Difficulty = Literal["easy", "medium", "hard", "expert"]


class QuizQuestion(BaseModel):
    """One assessment item - always grounded in retrieved evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    question: str
    question_type: QuestionType = "mcq"
    difficulty: Difficulty = "medium"
    topic: str = "General"
    options: list[str] = Field(default_factory=list)
    correct_answer: str = ""
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str = ""
    hint: str = ""
    points: float = 1.0
    citation_refs: list[str] = Field(default_factory=list, description="Chunk ids backing the item.")
    misconception_targeted: str | None = None

    @property
    def is_objective(self) -> bool:
        return self.question_type in ("mcq", "true_false", "numerical")

    def acceptable(self) -> list[str]:
        return [self.correct_answer, *self.accepted_answers]


class QuizSet(BaseModel):
    """A generated quiz: questions plus the evidence they came from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str = "Quiz"
    subject: str = "General"
    difficulty: Difficulty = "medium"
    questions: list[QuizQuestion] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    interactive: bool = True
    timed: bool = False
    duration_minutes: int | None = None
    total_points: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    def recompute_points(self) -> "QuizSet":
        self.total_points = sum(q.points for q in self.questions)
        return self


class AnswerEvaluation(BaseModel):
    """Result of grading one learner response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question_id: str
    is_correct: bool
    score: float = 0.0
    max_score: float = 1.0
    feedback: str = ""
    misconception: str | None = None
    expected_answer: str = ""
    learner_answer: str = ""
    topic: str = "General"
    next_hint: str | None = None

    @property
    def percentage(self) -> float:
        return round(100 * self.score / self.max_score, 2) if self.max_score else 0.0


class QuizReport(BaseModel):
    """Aggregate outcome of a finished attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    quiz_id: str
    title: str = "Quiz"
    total_questions: int = 0
    attempted: int = 0
    correct: int = 0
    score_percent: float = 0.0
    points_earned: float = 0.0
    points_possible: float = 0.0
    duration_seconds: float | None = None
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    topic_breakdown: dict[str, float] = Field(default_factory=dict)
    weak_topics: list[str] = Field(default_factory=list)
    strong_topics: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    study_minutes_logged: float = 0.0


class WeakTopicSignal(str, Enum):
    """Coarse weak-signal categories used by the adaptive engine."""

    MISSED = "missed"
    SLOW = "slow"
    UNSEEN = "unseen"
    MASTERED = "mastered"