"""Repositories: all database access lives here.

Agents and UI code never write SQL.  Keeping persistence behind repositories
also makes the app portable to PostgreSQL (swap ``DATABASE_URL``) and easy to
fake in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from workcompanion.config.logging_config import get_logger
from workcompanion.database.models import (
    Conversation,
    ConversationMessage,
    Document,
    Flashcard,
    FlashcardDeck,
    PaperAnalysisRecord,
    QuizAttempt,
    QuizQuestionRecord,
    StudyPlanRecord,
    StudySession,
    TopicProgress,
    User,
    new_id,
    utcnow,
)
from workcompanion.schemas.documents import ParsedDocument
from workcompanion.schemas.flashcards import Flashcard as FlashcardSchema
from workcompanion.schemas.flashcards import FlashcardDeck as FlashcardDeckSchema
from workcompanion.schemas.planner import StudyPlan
from workcompanion.schemas.quiz import AnswerEvaluation, QuizQuestion, QuizReport, QuizSet
from workcompanion.schemas.research import PaperAnalysis

logger = get_logger(__name__)


# ===========================================================================
# Users & preferences
# ===========================================================================
class UserRepository:
    """Local learner profile."""

    def get_or_create(self, session: Session, *, display_name: str = "Learner") -> User:
        user = session.scalar(select(User).order_by(User.created_at).limit(1))
        if user is None:
            user = User(display_name=display_name, preferences={})
            session.add(user)
            session.flush()
        return user

    def update_preferences(self, session: Session, **fields: Any) -> User:
        user = self.get_or_create(session)
        for key, value in fields.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)
        user.last_active_at = utcnow()
        session.flush()
        return user

    def merge_preferences(self, session: Session, extra: dict[str, Any]) -> User:
        user = self.get_or_create(session)
        prefs = dict(user.preferences or {})
        prefs.update({k: v for k, v in extra.items() if v is not None})
        user.preferences = prefs
        session.flush()
        return user


# ===========================================================================
# Documents
# ===========================================================================
class DocumentRepository:
    """Document metadata + indexing lifecycle."""

    def register(
        self,
        session: Session,
        document: ParsedDocument,
        *,
        stored_path: str | None = None,
        size_bytes: int = 0,
        sha256: str | None = None,
        subject: str | None = None,
        status: str = "pending",
    ) -> Document:
        record = Document(
            id=document.document_id,
            name=document.document_name,
            stored_path=stored_path or document.source_path,
            source_type=document.source_type,
            size_bytes=size_bytes or document.char_count,
            sha256=sha256,
            content_hash=document.content_hash(),
            title=document.title,
            author=document.author,
            subject=subject or document.subject,
            tags=document.tags,
            page_count=document.page_count,
            char_count=document.char_count,
            used_ocr=document.used_ocr,
            status=status,
            warnings=document.warnings,
            extra=document.metadata,
        )
        session.add(record)
        session.flush()
        return record

    def mark_indexed(self, session: Session, document_id: str, *, chunk_count: int, warnings: list[str] | None = None) -> Document | None:
        record = session.get(Document, document_id)
        if record is None:
            return None
        record.status = "indexed"
        record.chunk_count = chunk_count
        record.indexed_at = utcnow()
        record.error = None
        if warnings:
            record.warnings = warnings
        session.flush()
        return record

    def mark_failed(self, session: Session, document_id: str, error: str) -> Document | None:
        record = session.get(Document, document_id)
        if record is None:
            return None
        record.status = "failed"
        record.error = error[:2000]
        session.flush()
        return record

    def get(self, session: Session, document_id: str) -> Document | None:
        return session.get(Document, document_id)

    def list_all(self, session: Session, *, status: str | None = None) -> list[Document]:
        statement = select(Document).order_by(Document.created_at.desc())
        if status:
            statement = statement.where(Document.status == status)
        return list(session.scalars(statement))

    def find_by_hash(self, session: Session, sha256: str, size_bytes: int) -> Document | None:
        return session.scalar(
            select(Document).where(Document.sha256 == sha256, Document.size_bytes == size_bytes)
        )

    def delete(self, session: Session, document_id: str) -> bool:
        record = session.get(Document, document_id)
        if record is None:
            return False
        session.delete(record)
        session.flush()
        return True

    def update_metadata(self, session: Session, document_id: str, **fields: Any) -> Document | None:
        record = session.get(Document, document_id)
        if record is None:
            return None
        for key, value in fields.items():
            if hasattr(record, key) and value is not None:
                setattr(record, key, value)
        session.flush()
        return record

    def counts(self, session: Session) -> tuple[int, int]:
        total = session.scalar(select(func.count(Document.id))) or 0
        indexed = session.scalar(
            select(func.count(Document.id)).where(Document.status == "indexed")
        ) or 0
        return int(total), int(indexed)

    def subjects(self, session: Session) -> list[str]:
        rows = session.scalars(
            select(Document.subject).where(Document.subject.is_not(None)).distinct()
        )
        return sorted({str(row) for row in rows if row})


# ===========================================================================
# Study sessions
# ===========================================================================
class SessionRepository:
    """Study session tracking for time-on-task metrics."""

    def start(
        self, session: Session, *, mode: str = "chat", subject: str | None = None, user_id: str | None = None
    ) -> StudySession:
        record = StudySession(mode=mode, subject=subject, user_id=user_id)
        session.add(record)
        session.flush()
        return record

    def finish(self, session: Session, record: StudySession, *, questions_asked: int = 0, notes: str | None = None) -> StudySession:
        record.ended_at = utcnow()
        start = record.started_at
        end = record.ended_at
        if start is not None and end is not None:
            start_aware = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            end_aware = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            record.duration_minutes = max(0.0, (end_aware - start_aware).total_seconds() / 60.0)
        record.questions_asked = questions_asked
        if notes:
            record.notes = notes[:4000]
        session.flush()
        return record

    def total_minutes(self, session: Session) -> float:
        return float(session.scalar(select(func.sum(StudySession.duration_minutes))) or 0.0)

    def recent(self, session: Session, limit: int = 20) -> list[StudySession]:
        return list(
            session.scalars(select(StudySession).order_by(StudySession.started_at.desc()).limit(limit))
        )


# ===========================================================================
# Quizzes & adaptive progress
# ===========================================================================
class QuizRepository:
    """Quiz attempts plus the topic-level adaptive signal."""

    def start_attempt(
        self,
        session: Session,
        quiz: QuizSet,
        *,
        mode: str = "practice",
    ) -> QuizAttempt:
        attempt = QuizAttempt(
            quiz_id=quiz.id,
            title=quiz.title,
            subject=quiz.subject,
            difficulty=quiz.difficulty,
            mode=mode,
            timed=quiz.timed,
            total_questions=len(quiz.questions),
            points_possible=quiz.total_points or float(len(quiz.questions)),
        )
        session.add(attempt)
        session.flush()
        return attempt

    def record_answer(
        self,
        session: Session,
        attempt: QuizAttempt,
        question: QuizQuestion,
        evaluation: AnswerEvaluation,
        *,
        seconds_taken: float = 0.0,
    ) -> QuizQuestionRecord:
        record = QuizQuestionRecord(
            attempt_id=attempt.id,
            question_id=question.id,
            question_text=question.question,
            question_type=question.question_type,
            difficulty=question.difficulty,
            topic=(question.topic or "General")[:160],
            learner_answer=evaluation.learner_answer[:4000],
            expected_answer=evaluation.expected_answer[:4000],
            is_correct=evaluation.is_correct,
            score=evaluation.score,
            max_score=evaluation.max_score,
            seconds_taken=seconds_taken,
            feedback=evaluation.feedback[:4000],
            misconception=evaluation.misconception,
        )
        session.add(record)
        attempt.attempted += 1
        if evaluation.is_correct:
            attempt.correct_count += 1
            attempt.points_earned += evaluation.score
        session.flush()
        return record

    def complete_attempt(
        self,
        session: Session,
        attempt: QuizAttempt,
        report: QuizReport,
        *,
        duration_seconds: float | None = None,
    ) -> QuizAttempt:
        attempt.completed_at = utcnow()
        attempt.attempted = report.attempted or attempt.attempted
        attempt.correct_count = report.correct
        attempt.score_percent = report.score_percent
        attempt.points_earned = report.points_earned
        attempt.points_possible = report.points_possible or attempt.points_possible
        attempt.duration_seconds = duration_seconds if duration_seconds is not None else report.duration_seconds
        attempt.weak_topics = report.weak_topics[:10]
        session.flush()
        return attempt

    def recent_attempts(self, session: Session, limit: int = 20) -> list[QuizAttempt]:
        return list(
            session.scalars(select(QuizAttempt).order_by(QuizAttempt.started_at.desc()).limit(limit))
        )

    def attempt_questions(self, session: Session, attempt_id: str) -> list[QuizQuestionRecord]:
        return list(
            session.scalars(
                select(QuizQuestionRecord)
                .where(QuizQuestionRecord.attempt_id == attempt_id)
                .order_by(QuizQuestionRecord.created_at)
            )
        )

    def aggregate_scores(self, session: Session) -> list[tuple[str, float]]:
        rows = session.execute(
            select(QuizAttempt.started_at, QuizAttempt.score_percent)
            .where(QuizAttempt.completed_at.is_not(None))
            .order_by(QuizAttempt.started_at)
        ).all()
        return [(row[0].strftime("%Y-%m-%d") if hasattr(row[0], "strftime") else str(row[0]), float(row[1])) for row in rows]

    def stats(self, session: Session) -> dict[str, float]:
        attempts = session.scalar(select(func.count(QuizAttempt.id))) or 0
        answered = session.scalar(select(func.count(QuizQuestionRecord.id))) or 0
        correct = session.scalar(
            select(func.count(QuizQuestionRecord.id)).where(QuizQuestionRecord.is_correct.is_(True))
        ) or 0
        avg = session.scalar(select(func.avg(QuizAttempt.score_percent))) or 0.0
        return {
            "attempts": float(attempts),
            "questions_answered": float(answered),
            "questions_correct": float(correct),
            "average_score": float(avg),
        }


class TopicProgressRepository:
    """Per-topic mastery - the adaptive learning engine's memory."""

    #: Mastery update rate (exponential moving average weight).
    ALPHA = 0.4

    def update(
        self,
        session: Session,
        topic: str,
        *,
        correct: bool,
        subject: str = "General",
        seconds: float = 0.0,
    ) -> TopicProgress:
        topic = (topic or "General").strip()[:160] or "General"
        record = session.scalar(
            select(TopicProgress).where(
                TopicProgress.topic == topic, TopicProgress.subject == subject
            )
        )
        if record is None:
            record = TopicProgress(topic=topic, subject=subject)
            session.add(record)
            session.flush()

        record.attempts += 1
        if correct:
            record.correct += 1
        record.total_seconds += max(0.0, seconds)
        observed = 1.0 if correct else 0.0
        # Weight early attempts more heavily so mastery reacts quickly.
        weight = self.ALPHA if record.attempts > 2 else 0.6
        record.mastery = round((1 - weight) * record.mastery + weight * observed, 4)
        record.last_seen = utcnow()
        session.flush()
        return record

    def all(self, session: Session, *, subject: str | None = None) -> list[TopicProgress]:
        statement = select(TopicProgress).order_by(TopicProgress.mastery)
        if subject:
            statement = statement.where(TopicProgress.subject == subject)
        return list(session.scalars(statement))

    def weak_topics(self, session: Session, *, limit: int = 5, subject: str | None = None) -> list[TopicProgress]:
        seen = [row for row in self.all(session, subject=subject) if row.attempts > 0]
        return sorted(seen, key=lambda row: (row.mastery, -row.attempts))[:limit]

    def strong_topics(self, session: Session, *, limit: int = 5, subject: str | None = None) -> list[TopicProgress]:
        seen = [row for row in self.all(session, subject=subject) if row.attempts > 0]
        return sorted(seen, key=lambda row: (-row.mastery, row.attempts))[:limit]

    def accuracy(self, session: Session, topic: str, subject: str = "General") -> float:
        record = session.scalar(
            select(TopicProgress).where(
                TopicProgress.topic == topic, TopicProgress.subject == subject
            )
        )
        if record is None or record.attempts == 0:
            return 0.0
        return round(100.0 * record.correct / record.attempts, 2)

    def known_topics(self, session: Session, *, subject: str | None = None) -> list[str]:
        return [row.topic for row in self.all(session, subject=subject)]


# ===========================================================================
# Flashcards
# ===========================================================================
class FlashcardRepository:
    """Deck persistence."""

    def save_deck(
        self,
        session: Session,
        deck: FlashcardDeckSchema,
        *,
        source_document_id: str | None = None,
    ) -> FlashcardDeck:
        record = FlashcardDeck(
            id=deck.id,
            title=deck.title,
            subject=deck.subject,
            source_document_id=source_document_id,
            card_count=len(deck.cards),
            citations=[citation.model_dump() for citation in deck.citations],
        )
        session.add(record)
        for card in deck.cards:
            session.add(
                Flashcard(
                    id=card.id,
                    deck_id=deck.id,
                    front=card.front,
                    back=card.back,
                    topic=(card.topic or "General")[:160],
                    difficulty=card.difficulty,
                    hint=card.hint,
                    citation_refs=list(card.citation_refs),
                )
            )
        session.flush()
        return record

    def get_deck(self, session: Session, deck_id: str) -> FlashcardDeck | None:
        return session.get(FlashcardDeck, deck_id)

    def list_decks(self, session: Session, limit: int = 20) -> list[FlashcardDeck]:
        return list(
            session.scalars(
                select(FlashcardDeck).order_by(FlashcardDeck.created_at.desc()).limit(limit)
            )
        )

    def cards(self, session: Session, deck_id: str) -> list[Flashcard]:
        return list(
            session.scalars(
                select(Flashcard).where(Flashcard.deck_id == deck_id).order_by(Flashcard.created_at)
            )
        )

    def review(self, session: Session, card_id: str, *, correct: bool) -> None:
        card = session.get(Flashcard, card_id)
        if card is None:
            return
        card.times_reviewed += 1
        if correct:
            card.times_correct += 1
        session.flush()

    def total_cards(self, session: Session) -> int:
        return int(session.scalar(select(func.count(Flashcard.id))) or 0)


# ===========================================================================
# Conversations
# ===========================================================================
class ConversationRepository:
    """Chat history with rolling-window summaries for cheap context."""

    #: Number of recent turns kept verbatim in the prompt.
    WINDOW = 6

    def get_or_create(self, session: Session, *, conversation_id: str | None = None) -> Conversation:
        if conversation_id:
            record = session.get(Conversation, conversation_id)
            if record is not None:
                return record
        record = Conversation(title="New conversation")
        session.add(record)
        session.flush()
        return record

    def add_message(
        self,
        session: Session,
        conversation: Conversation,
        role: str,
        content: str,
        *,
        agent: str | None = None,
        intent: str | None = None,
        confidence: float = 0.0,
        citations: Sequence[dict[str, Any]] | None = None,
        grounding: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            conversation_id=conversation.id,
            role=role,
            content=content,
            agent=agent,
            intent=intent,
            confidence=confidence,
            citations=list(citations or []),
            grounding=list(grounding or []),
            mode=mode,
        )
        session.add(message)
        conversation.message_count += 1
        conversation.updated_at = utcnow()
        if role == "user" and conversation.title in ("", "New conversation"):
            conversation.title = content.strip()[:80] or "New conversation"
        session.flush()
        return message

    def messages(self, session: Session, conversation_id: str, limit: int = 50) -> list[ConversationMessage]:
        return list(
            session.scalars(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.created_at)
                .limit(limit)
            )
        )

    def recent_turns(self, session: Session, conversation_id: str, window: int | None = None) -> list[tuple[str, str]]:
        """Last ``window`` (role, content) pairs for prompt context."""
        window = window or self.WINDOW
        rows = self.messages(session, conversation_id, limit=200)
        return [(row.role, row.content) for row in rows[-window:]]

    def summarize_if_needed(self, session: Session, conversation_id: str, *, threshold: int = 12) -> str | None:
        """Create/refresh a rolling summary once a thread gets long.

        Summarisation is done by a caller-supplied callable in the memory layer;
        here we only maintain the stored summary string.
        """
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return None
        return conversation.summary

    def list_conversations(self, session: Session, limit: int = 20) -> list[Conversation]:
        return list(
            session.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit))
        )

    def clear(self, session: Session, conversation_id: str) -> None:
        session.execute(delete(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id))
        conversation = session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.message_count = 0
            conversation.summary = None
        session.flush()

    def total_messages(self, session: Session) -> int:
        return int(session.scalar(select(func.count(ConversationMessage.id))) or 0)


# ===========================================================================
# Plans & paper analyses
# ===========================================================================
class PlanRepository:
    """Study plan persistence."""

    def save(self, session: Session, plan: StudyPlan) -> StudyPlanRecord:
        record = StudyPlanRecord(
            id=plan.id,
            subject=plan.subject,
            exam_date=plan.exam_date,
            total_days=plan.total_days,
            total_hours=plan.total_hours,
            plan=plan.model_dump(mode="json"),
        )
        session.add(record)
        session.flush()
        return record

    def list_plans(self, session: Session, limit: int = 20) -> list[StudyPlanRecord]:
        return list(
            session.scalars(select(StudyPlanRecord).order_by(StudyPlanRecord.created_at.desc()).limit(limit))
        )


class PaperAnalysisRepository:
    """Cached paper analyses keyed by document id."""

    def save(self, session: Session, analysis: PaperAnalysis) -> PaperAnalysisRecord:
        session.add(
            PaperAnalysisRecord(
                document_id=analysis.document_id,
                document_name=analysis.document_name,
                analysis=analysis.model_dump(mode="json"),
            )
        )
        session.flush()

    def latest_for(self, session: Session, document_id: str) -> PaperAnalysis | None:
        record = session.scalar(
            select(PaperAnalysisRecord)
            .where(PaperAnalysisRecord.document_id == document_id)
            .order_by(PaperAnalysisRecord.created_at.desc())
            .limit(1)
        )
        if record is None:
            return None
        try:
            return PaperAnalysis.model_validate(record.analysis)
        except Exception:  # pragma: no cover - schema drift
            return None

    def list_for(self, session: Session, document_id: str, limit: int = 10) -> list[PaperAnalysisRecord]:
        return list(
            session.scalars(
                select(PaperAnalysisRecord)
                .where(PaperAnalysisRecord.document_id == document_id)
                .order_by(PaperAnalysisRecord.created_at.desc())
                .limit(limit)
            )
        )


# ===========================================================================
# Convenience facade
# ===========================================================================
class RepositoryBundle:
    """Single object the UI and agents use to reach every table."""

    def __init__(self, session: Session):
        self.session = session
        self.users = UserRepository()
        self.documents = DocumentRepository()
        self.sessions = SessionRepository()
        self.quizzes = QuizRepository()
        self.topics = TopicProgressRepository()
        self.flashcards = FlashcardRepository()
        self.conversations = ConversationRepository()
        self.plans = PlanRepository()
        self.papers = PaperAnalysisRepository()

    # ------------------------------------------------------------------
    def topic_breakdown(self, subject: str | None = None) -> dict[str, float]:
        return {
            row.topic: round(100.0 * row.correct / row.attempts, 1)
            for row in self.topics.all(self.session, subject=subject)
            if row.attempts > 0
        }

    def subjects_with_progress(self) -> list[str]:
        return sorted({row.subject for row in self.topics.all(self.session) if row.attempts > 0})

    def recent_activity(self, days: int = 14) -> dict[str, int]:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.session.execute(
            select(ConversationMessage.created_at).where(ConversationMessage.created_at >= since)
        ).all()
        counts: dict[str, int] = {}
        for (created_at,) in rows:
            stamp = created_at.date().isoformat() if hasattr(created_at, "date") else str(created_at)[:10]
            counts[stamp] = counts.get(stamp, 0) + 1
        return dict(sorted(counts.items()))

    def new_conversation_id(self) -> str:  # pragma: no cover - trivial
        return new_id()


def get_repositories(session: Session) -> RepositoryBundle:
    """Build the repository facade for a session."""
    return RepositoryBundle(session)