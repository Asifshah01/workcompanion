"""Progress tracking: topic mastery, weak-area detection and recommendations.

This is the adaptive loop's brain:

    attempt -> topic mastery update -> weak/strong classification
            -> targeted next recommendation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from workcompanion.config.logging_config import get_logger
from workcompanion.database.repositories import (
    QuizRepository,
    RepositoryBundle,
    TopicProgressRepository,
)
from workcompanion.schemas.progress import ProgressSnapshot, TopicStat

logger = get_logger(__name__)

#: Mastery bands used for labelling in the UI.
MASTERY_MASTERED = 0.85
MASTERY_COMPETENT = 0.65
MASTERY_DEVELOPING = 0.40


@dataclass(slots=True)
class Recommendation:
    """A concrete 'what to do next' suggestion."""

    headline: str
    detail: str = ""
    topics: list[str] = field(default_factory=list)
    action_hint: str = ""
    priority: int = 0


class ProgressTracker:
    """Aggregates quiz/session history into a learner-facing snapshot."""

    def __init__(self, repos: RepositoryBundle, session: Session):
        self._repos = repos
        self._session = session
        self._topics: TopicProgressRepository = repos.topics
        self._quizzes: QuizRepository = repos.quizzes

    # ------------------------------------------------------------------
    def record_answer(
        self,
        topic: str,
        *,
        correct: bool,
        subject: str = "General",
        seconds: float = 0.0,
    ) -> TopicStat:
        """Update mastery for one answered question."""
        record = self._topics.update(
            self._session, topic, correct=correct, subject=subject, seconds=seconds
        )
        return self._to_stat(record)

    # ------------------------------------------------------------------
    def topic_stats(self, subject: str | None = None) -> list[TopicStat]:
        rows = self._topics.all(self._session, subject=subject)
        return [self._to_stat(row) for row in rows]

    @staticmethod
    def _to_stat(record: object) -> TopicStat:
        attempts = int(getattr(record, "attempts", 0) or 0)
        correct = int(getattr(record, "correct", 0) or 0)
        mastery = float(getattr(record, "mastery", 0.0) or 0.0)
        if attempts == 0:
            status = "unseen"
        elif mastery >= MASTERY_MASTERED:
            status = "mastered"
        elif mastery >= MASTERY_COMPETENT:
            status = "competent"
        elif mastery >= MASTERY_DEVELOPING:
            status = "developing"
        else:
            status = "needs_work"
        return TopicStat(
            topic=str(getattr(record, "topic", "General")),
            subject=str(getattr(record, "subject", "General")),
            attempts=attempts,
            correct=correct,
            accuracy=round(100.0 * correct / attempts, 2) if attempts else 0.0,
            avg_seconds=round(float(getattr(record, "total_seconds", 0.0) or 0.0) / attempts, 2)
            if attempts
            else 0.0,
            mastery=mastery,
            last_seen=getattr(record, "last_seen", None),
            status=status,
        )

    # ------------------------------------------------------------------
    def weak_topics(self, subject: str | None = None, limit: int = 5) -> list[TopicStat]:
        stats = [s for s in self.topic_stats(subject) if s.attempts > 0 and s.mastery < MASTERY_COMPETENT]
        stats.sort(key=lambda s: (s.mastery, -s.attempts))
        return stats[:limit]

    def strong_topics(self, subject: str | None = None, limit: int = 5) -> list[TopicStat]:
        stats = [s for s in self.topic_stats(subject) if s.attempts > 0]
        stats.sort(key=lambda s: (-s.mastery, s.attempts))
        return stats[:limit]

    def overall_mastery(self, subject: str | None = None) -> float:
        stats = [s for s in self.topic_stats(subject) if s.attempts > 0]
        if not stats:
            return 0.0
        # Attempt-weighted so a single lucky answer does not dominate.
        total = sum(s.attempts for s in stats)
        return round(sum(s.mastery * s.attempts for s in stats) / total, 4) if total else 0.0

    # ------------------------------------------------------------------
    def snapshot(self, subject: str | None = None) -> ProgressSnapshot:
        """Full dashboard payload."""
        stats = self.topic_stats(subject)
        quiz_stats = self._quizzes.stats(self._session)
        docs_total, docs_indexed = self._repos.documents.counts(self._session)
        chunks = sum(
            int(row.chunk_count or 0) for row in self._repos.documents.list_all(self._session, status="indexed")
        )
        recommendation = self.recommend(subject=subject)

        return ProgressSnapshot(
            subjects=self._repos.subjects_with_progress(),
            topics=stats,
            overall_mastery=self.overall_mastery(subject),
            documents_indexed=docs_indexed,
            total_chunks=chunks,
            documents_total=docs_total,
            quiz_attempts=int(quiz_stats["attempts"]),
            questions_answered=int(quiz_stats["questions_answered"]),
            questions_correct=int(quiz_stats["questions_correct"]),
            average_quiz_score=round(quiz_stats["average_score"], 2),
            flashcards_generated=self._repos.flashcards.total_cards(self._session),
            study_minutes=round(self._repos.sessions.total_minutes(self._session), 2),
            sessions_count=len(self._repos.sessions.recent(self._session, limit=500)),
            questions_asked=self._repos.conversations.total_messages(self._session),
            activity_by_day=self._repos.recent_activity(),
            recent_scores=self._repos.quizzes.aggregate_scores(self._session)[-20:],
            recommendation=recommendation.headline,
            recommendation_topics=recommendation.topics,
        )

    # ------------------------------------------------------------------
    def recommend(self, subject: str | None = None) -> Recommendation:
        """Decide what the learner should do next."""
        weak = self.weak_topics(subject, limit=3)
        unseen = [s for s in self.topic_stats(subject) if s.attempts == 0]
        strong = self.strong_topics(subject, limit=3)
        today = date.today()

        if weak:
            names = [s.topic for s in weak]
            worst = weak[0]
            if worst.mastery < 0.4 and worst.attempts >= 2:
                headline = (
                    f"Review {names[0]} fundamentals before attempting harder questions"
                )
                detail = (
                    f"Accuracy on {worst.topic} is {worst.accuracy:.0f}% over {worst.attempts} "
                    f"attempts. Start with a targeted explanation, then retake a short quiz."
                )
            else:
                headline = f"Practise {', '.join(names[:2])} with focused questions"
                detail = (
                    "These topics are below the mastery threshold. Re-read the relevant "
                    "sections, then run a targeted quiz limited to these topics."
                )
            return Recommendation(
                headline=headline,
                detail=detail,
                topics=names,
                action_hint="Generate a targeted quiz on these topics",
                priority=1,
            )

        if unseen:
            names = [s.topic for s in unseen[:3]]
            return Recommendation(
                headline=f"Start assessing {', '.join(names[:2])}",
                detail=(
                    "You have not been tested on these topics yet. A short diagnostic quiz "
                    "will show what needs attention."
                ),
                topics=names,
                action_hint="Run a diagnostic quiz",
                priority=2,
            )

        if strong:
            names = [s.topic for s in strong[:2]]
            return Recommendation(
                headline=f"You're strong in {', '.join(names)} - move to harder material",
                detail=(
                    f"Mastery is {strong[0].mastery * 100:.0f}% on {strong[0].topic}. "
                    f"It is a good moment to attempt {today.strftime('%A')}'s harder questions "
                    "or an exam-mode paper."
                ),
                topics=names,
                action_hint="Start an exam-mode session",
                priority=3,
            )

        return Recommendation(
            headline="Upload material and ask a question to begin",
            detail=(
                "Once you index documents and answer some questions, WorkCompanion will "
                "recommend what to study next."
            ),
            topics=[],
            action_hint="Open My Knowledge",
            priority=4,
        )