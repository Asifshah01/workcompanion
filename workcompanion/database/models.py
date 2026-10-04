"""SQLAlchemy ORM models.

SQLite by default; the models avoid SQLite-only types so the same schema works
on PostgreSQL later (just change ``DATABASE_URL``).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Short, URL-safe identifier."""
    return uuid.uuid4().hex[:16]


class Base(DeclarativeBase):
    """Declarative base with portable column defaults."""


# ---------------------------------------------------------------------------
class User(Base):
    """Local learner profile. No sensitive personal data is stored."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    display_name: Mapped[str] = mapped_column(String(120), default="Learner")
    preferred_level: Mapped[str] = mapped_column(String(24), default="undergraduate")
    preferred_style: Mapped[str] = mapped_column(String(24), default="detailed")
    default_subject: Mapped[str | None] = mapped_column(String(120), nullable=True)
    weekly_study_hours: Mapped[float] = mapped_column(Float, default=2.0)
    show_citations_always: Mapped[bool] = mapped_column(Boolean, default=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    documents: Mapped[list["Document"]] = relationship(back_populates="owner")


# ---------------------------------------------------------------------------
class Document(Base):
    """Metadata for an indexed document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(320), index=True)
    stored_path: Mapped[str | None] = mapped_column(String(600), nullable=True)
    source_type: Mapped[str] = mapped_column(String(24), default="unknown", index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(String(400), nullable=True)
    author: Mapped[str | None] = mapped_column(String(240), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    used_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    owner: Mapped[User | None] = relationship(back_populates="documents")

    __table_args__ = (
        UniqueConstraint("sha256", "size_bytes", name="uq_document_dedupe"),
        Index("ix_document_status_created", "status", "created_at"),
    )


# ---------------------------------------------------------------------------
class StudySession(Base):
    """One continuous interaction with the application."""

    __tablename__ = "study_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(32), default="chat", index=True)
    subject: Mapped[str | None] = mapped_column(String(160), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    questions_asked: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
class QuizAttempt(Base):
    """A completed (or in-progress) quiz."""

    __tablename__ = "quiz_attempts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    quiz_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(240), default="Quiz")
    subject: Mapped[str] = mapped_column(String(160), default="General", index=True)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    mode: Mapped[str] = mapped_column(String(16), default="practice")
    timed: Mapped[bool] = mapped_column(Boolean, default=False)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    attempted: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    score_percent: Mapped[float] = mapped_column(Float, default=0.0)
    points_earned: Mapped[float] = mapped_column(Float, default=0.0)
    points_possible: Mapped[float] = mapped_column(Float, default=0.0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    weak_topics: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    questions: Mapped[list["QuizQuestionRecord"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class QuizQuestionRecord(Base):
    """One graded question inside an attempt (the adaptive-learning signal)."""

    __tablename__ = "quiz_question_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    attempt_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("quiz_attempts.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    question_text: Mapped[str] = mapped_column(Text, default="")
    question_type: Mapped[str] = mapped_column(String(24), default="mcq")
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    topic: Mapped[str] = mapped_column(String(160), default="General", index=True)
    learner_answer: Mapped[str] = mapped_column(Text, default="")
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, default=1.0)
    seconds_taken: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[str] = mapped_column(Text, default="")
    misconception: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    attempt: Mapped[QuizAttempt] = relationship(back_populates="questions")

    __table_args__ = (Index("ix_question_topic_correct", "topic", "is_correct"),)


# ---------------------------------------------------------------------------
class TopicProgress(Base):
    """Rolling per-topic mastery estimate powering the adaptive engine."""

    __tablename__ = "topic_progress"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    subject: Mapped[str] = mapped_column(String(160), default="General", index=True)
    topic: Mapped[str] = mapped_column(String(160), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    total_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    mastery: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (UniqueConstraint("subject", "topic", name="uq_topic_progress"),)


# ---------------------------------------------------------------------------
class FlashcardDeck(Base):
    """A generated deck."""

    __tablename__ = "flashcard_decks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(240), default="Flashcard deck")
    subject: Mapped[str] = mapped_column(String(160), default="General", index=True)
    source_document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    card_count: Mapped[int] = mapped_column(Integer, default=0)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    cards: Mapped[list["Flashcard"]] = relationship(
        back_populates="deck", cascade="all, delete-orphan"
    )


class Flashcard(Base):
    """A single question/answer card."""

    __tablename__ = "flashcards"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    deck_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("flashcard_decks.id", ondelete="CASCADE"), index=True
    )
    front: Mapped[str] = mapped_column(Text)
    back: Mapped[str] = mapped_column(Text)
    topic: Mapped[str] = mapped_column(String(160), default="General", index=True)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_refs: Mapped[list] = mapped_column(JSON, default=list)
    times_reviewed: Mapped[int] = mapped_column(Integer, default=0)
    times_correct: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    deck: Mapped[FlashcardDeck] = relationship(back_populates="cards")


# ---------------------------------------------------------------------------
class Conversation(Base):
    """A chat thread."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(240), default="New conversation")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="ConversationMessage.created_at"
    )


class ConversationMessage(Base):
    """One turn in a conversation, with the grounding metadata needed to audit it."""

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), index=True)
    content: Mapped[str] = mapped_column(Text)
    agent: Mapped[str | None] = mapped_column(String(48), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    grounding: Mapped[list] = mapped_column(JSON, default=list)
    mode: Mapped[str | None] = mapped_column(String(24), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
class StudyPlanRecord(Base):
    """Persisted study plan."""

    __tablename__ = "study_plans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    subject: Mapped[str] = mapped_column(String(160), index=True)
    exam_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_days: Mapped[int] = mapped_column(Integer, default=0)
    total_hours: Mapped[float] = mapped_column(Float, default=0.0)
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PaperAnalysisRecord(Base):
    """Cached research-paper analysis (avoids re-paying for the same paper)."""

    __tablename__ = "paper_analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    document_name: Mapped[str] = mapped_column(String(320), default="")
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_paper_document_created", "document_id", "created_at"),)