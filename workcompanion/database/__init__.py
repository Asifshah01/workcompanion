"""Persistence layer: SQLAlchemy models, engine management and repositories."""

from workcompanion.database.database import (
    database_summary,
    get_engine,
    get_session_factory,
    init_db,
    reset_db,
    session_scope,
)
from workcompanion.database.models import Base
from workcompanion.database.repositories import (
    ConversationRepository,
    DocumentRepository,
    FlashcardRepository,
    PaperAnalysisRepository,
    PlanRepository,
    QuizRepository,
    RepositoryBundle,
    SessionRepository,
    TopicProgressRepository,
    UserRepository,
    get_repositories,
)

__all__ = [
    "database_summary",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_db",
    "session_scope",
    "Base",
    "ConversationRepository",
    "DocumentRepository",
    "FlashcardRepository",
    "PaperAnalysisRepository",
    "PlanRepository",
    "QuizRepository",
    "RepositoryBundle",
    "SessionRepository",
    "TopicProgressRepository",
    "UserRepository",
    "get_repositories",
]