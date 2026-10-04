"""Router (NEXUS) and tutor schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ExplanationLevel = Literal["beginner", "undergraduate", "graduate", "expert"]
TeachingStyle = Literal["concise", "detailed", "socratic", "example_based", "mathematical", "analogy"]


class Intent(str, Enum):
    """Task types NEXUS can route."""

    EXPLAIN = "explain"
    ANSWER = "answer"
    TEACH = "teach"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    QUIZ = "quiz"
    EXAM = "exam"
    FLASHCARDS = "flashcards"
    PLAN = "plan"
    PAPER_ANALYSIS = "paper_analysis"
    RESEARCH = "research"
    SOCRATIC = "socratic"
    SEARCH = "search"
    PROGRESS = "progress"
    SMALL_TALK = "small_talk"
    UNCLEAR = "unclear"


class RouteDecision(BaseModel):
    """NEXUS's structured routing output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intent: Intent = Intent.UNCLEAR
    intents: list[Intent] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_documents: bool = True
    use_web: bool = False
    explanation_level: ExplanationLevel | None = None
    teaching_style: TeachingStyle | None = None
    difficulty: Literal["easy", "medium", "hard", "expert"] | None = None
    question_count: int | None = None
    flashcard_count: int | None = None
    topic: str | None = None
    document_filter: str | None = None
    interactive: bool = False
    timed: bool = False
    rationale: str = ""
    follow_ups: list[str] = Field(default_factory=list)

    def all_intents(self) -> list[Intent]:
        return self.intents or [self.intent]


class TutorRequest(BaseModel):
    """Normalised tutoring request handed to the Tutor / Socratic agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    level: ExplanationLevel = "undergraduate"
    style: TeachingStyle = "detailed"
    subject: str | None = None
    topic: str | None = None
    misconceptions: list[str] = Field(default_factory=list)
    history_summary: str | None = None
    use_documents: bool = True
    allow_general_knowledge: bool = True
    follow_up_check: bool = True
    extra_context: dict[str, Any] = Field(default_factory=dict)


class TutorTurn(BaseModel):
    """A single Socratic dialogue turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    hint: str | None = None
    reveal_answer: bool = False
    scaffold_level: int = 1
    checks_understanding: bool = True
    next_question: str | None = None
    misconceptions_detected: list[str] = Field(default_factory=list)


class MisconceptionReport(BaseModel):
    """Output of grading a free-text learner answer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_correct: bool
    score: float = 0.0
    expected: str = ""
    feedback: str = ""
    misconception: str | None = None
    severity: Literal["none", "minor", "major"] = "none"
    corrected_explanation: str = ""
    follow_up_question: str | None = None
    citations: list[str] = Field(default_factory=list)