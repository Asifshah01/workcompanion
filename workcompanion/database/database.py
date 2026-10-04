"""Database engine / session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.database.models import Base

logger = get_logger(__name__)

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None
_ENGINE_KEY: str = ""


def _engine_key(settings: Settings) -> str:
    return f"{settings.database_url}|{settings.echo_sql}"


def get_engine(settings: Settings | None = None, *, force: bool = False) -> Engine:
    """Return the process-wide SQLAlchemy engine."""
    global _ENGINE, _ENGINE_KEY
    settings = settings or get_settings()
    key = _engine_key(settings)
    if _ENGINE is not None and _ENGINE_KEY == key and not force:
        return _ENGINE

    url = settings.database_url or "sqlite:///workcompanion.db"
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {"echo": getattr(settings, "echo_sql", False), "future": True}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Streamlit runs handlers on a worker thread; give SQLite a moment.
        connect_args["timeout"] = 30
        engine_kwargs["pool_pre_ping"] = True

    _ENGINE = create_engine(url, connect_args=connect_args, **engine_kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(_ENGINE, "connect")
        def _configure_sqlite(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    _ENGINE_KEY = key
    logger.info("Database engine ready: %s", url.split("://")[0])
    return _ENGINE


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return the session factory bound to the current engine."""
    global _SESSION_FACTORY
    settings = settings or get_settings()
    engine = get_engine(settings)
    if _SESSION_FACTORY is None or _ENGINE_KEY != _engine_key(settings):
        _SESSION_FACTORY = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _SESSION_FACTORY


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Transactional session context manager (commits on success)."""
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(settings: Settings | None = None, *, drop: bool = False) -> Engine:
    """Create all tables (idempotent)."""
    settings = settings or get_settings()
    engine = get_engine(settings)
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Database schema ready (%d tables)", len(Base.metadata.tables))
    return engine


def reset_db(settings: Settings | None = None) -> None:
    """Drop and recreate every table - used by tests and the Settings page."""
    settings = settings or get_settings()
    init_db(settings, drop=True)
    global _SESSION_FACTORY
    _SESSION_FACTORY = None


def database_summary(settings: Settings | None = None) -> dict[str, object]:
    """Small health/size summary for the Settings page."""
    settings = settings or get_settings()
    engine = get_engine(settings)
    summary: dict[str, object] = {"url": (settings.database_url or "").split("://")[0], "tables": len(Base.metadata.tables)}
    if settings.database_url and settings.database_url.startswith("sqlite:///"):
        path = Path(settings.database_url.replace("sqlite:///", "", 1))
        summary["path"] = str(path)
        summary["size_bytes"] = path.stat().st_size if path.exists() else 0
    try:
        with engine.connect() as connection:
            for table in ("documents", "quiz_attempts", "topic_progress", "flashcards"):
                summary[f"rows_{table}"] = connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
    except Exception:  # pragma: no cover - tables may not exist yet
        pass
    return summary