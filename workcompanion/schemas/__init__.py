"""Pydantic schemas - the structured contracts between every component.

Agents exchange these models instead of parsing free-form natural language,
which is what keeps the multi-agent layer reliable.
"""

from workcompanion.schemas.common import (
    AgentResult,
    Citation,
    ConfidenceLevel,
    GroundingLabel,
    SourceKind,
    success_result,
    failure_result,
)
from workcompanion.schemas.documents import (
    ChunkMetadata,
    DocumentChunk,
    ParsedDocument,
    RawPage,
)
from workcompanion.schemas.retrieval import (
    RetrievalOutcome,
    RetrievalQuery,
    RetrievedChunk,
    RetrievalStats,
)
from workcompanion.schemas.quiz import (
    AnswerEvaluation,
    QuizQuestion,
    QuizReport,
    QuizSet,
)
from workcompanion.schemas.flashcards import Flashcard, FlashcardDeck
from workcompanion.schemas.planner import StudyDay, StudyPlan, StudyPlanInput
from workcompanion.schemas.research import PaperAnalysis, WebSource
from workcompanion.schemas.tutor import (
    ExplanationLevel,
    MisconceptionReport,
    RouteDecision,
    TeachingStyle,
    TutorRequest,
    TutorTurn,
)
from workcompanion.schemas.progress import ProgressSnapshot, TopicStat

__all__ = [
    "AgentResult",
    "Citation",
    "ConfidenceLevel",
    "GroundingLabel",
    "SourceKind",
    "success_result",
    "failure_result",
    "ChunkMetadata",
    "DocumentChunk",
    "ParsedDocument",
    "RawPage",
    "RetrievalOutcome",
    "RetrievalQuery",
    "RetrievedChunk",
    "RetrievalStats",
    "AnswerEvaluation",
    "QuizQuestion",
    "QuizReport",
    "QuizSet",
    "Flashcard",
    "FlashcardDeck",
    "StudyDay",
    "StudyPlan",
    "StudyPlanInput",
    "PaperAnalysis",
    "WebSource",
    "ExplanationLevel",
    "RouteDecision",
    "TeachingStyle",
    "TutorRequest",
    "ProgressSnapshot",
    "TopicStat",
]