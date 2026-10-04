"""Per-rerun application state.

Streamlit re-executes the whole script on every widget interaction, so nothing
here may rebuild an expensive object, and nothing may outlive the database
session it was created with.  The split is:

* **process-wide** - the agent bundle (embedding model, Chroma handle, provider).
  Cached in :mod:`workcompanion.agents.bundle`, never rebuilt per rerun.
* **per operation** - the SQLAlchemy session.  Repositories, the learner profile,
  conversation memory and progress tracker are all created *inside* a
  :meth:`AppContext.session` block and discarded when it closes.

:class:`AppContext` is the single object every page receives.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from workcompanion.agents.bundle import AgentBundle, get_bundle, reset_bundle
from workcompanion.config.logging_config import configure_logging, get_logger
from workcompanion.config.settings import (
    PROJECT_ROOT,
    Settings,
    get_settings,
    reload_settings,
)
from workcompanion.crew.llm import crew_enabled
from workcompanion.database.database import database_summary, init_db, session_scope
from workcompanion.database.repositories import RepositoryBundle, get_repositories
from workcompanion.memory.conversation_memory import ConversationMemory
from workcompanion.memory.learner_profile import LearnerProfile
from workcompanion.memory.progress_tracker import ProgressSnapshot, ProgressTracker
from workcompanion.rag.vector_store import reset_vector_store_cache

logger = get_logger(__name__)

#: Session-state keys owned by this module.
KEY_INIT = "wc_initialised"
KEY_CHAT = "wc_chat"
KEY_CONVERSATION = "wc_conversation_id"
KEY_TURN = "wc_turn"
KEY_TRACE = "wc_trace"
KEY_LAST = "wc_last_result"
KEY_TRACE_NOTES = "wc_trace_notes"
KEY_NAVIGATE = "wc_navigate"

DEFAULT_CONVERSATION = "default"


def navigate(page_key: str) -> None:
    """Switch modes from inside a page.

    The sidebar radio is a widget, so writing ``wc_page`` alone would be
    overwritten on the next rerun. Queueing the destination and letting
    :func:`workcompanion.ui.layout.render_sidebar` apply it to the widget keeps
    one source of truth.
    """
    st.session_state[KEY_NAVIGATE] = str(page_key)


@dataclass
class AppContext:
    """Everything a page needs, assembled once per rerun."""

    settings: Settings
    bundle: AgentBundle

    # ------------------------------------------------------------- database
    @contextmanager
    def session(self) -> Iterator[RepositoryBundle]:
        """Repositories bound to a session that is committed and closed."""
        with session_scope(self.settings) as session:
            yield get_repositories(session)

    @contextmanager
    def learner(self) -> Iterator[tuple[AgentBundle, RepositoryBundle, ConversationMemory]]:
        """Bundle + repositories + memory, all alive for the duration of the block.

        The learner profile snapshot is pushed into the tutor here so every agent
        sees the same, current profile for this request.
        """
        with self.session() as repositories:
            profile = LearnerProfile(repositories)
            self.bundle.tutor.set_profile(profile.snapshot())
            memory = ConversationMemory(
                repositories.session, repositories.conversations, self.bundle.llm
            )
            try:
                yield self.bundle, repositories, memory
            finally:
                self.bundle.tutor.set_profile(profile.snapshot())

    def progress_snapshot(self, subject: str | None = None) -> ProgressSnapshot:
        """Read the learner's progress (its own short-lived session)."""
        with self.session() as repositories:
            return ProgressTracker(repositories, repositories.session).snapshot(subject)

    def profile_snapshot(self, subject: str | None = None):
        """Read the learner profile (its own short-lived session)."""
        with self.session() as repositories:
            return LearnerProfile(repositories).snapshot(subject)

    # ------------------------------------------------------------- routing
    @property
    def conversation_id(self) -> str:
        return str(st.session_state.get(KEY_CONVERSATION) or DEFAULT_CONVERSATION)

    @conversation_id.setter
    def conversation_id(self, value: str) -> None:
        st.session_state[KEY_CONVERSATION] = value

    @property
    def offline(self) -> bool:
        return self.bundle.llm.name == "offline"

    def log_action(self, label: str, **fields: Any) -> None:
        logger.info("[ui] %s %s", label, fields if fields else "")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap() -> AppContext:
    """Prepare directories, logging, the database and the shared bundle."""
    if not st.session_state.get(KEY_INIT):
        settings = get_settings()
        configure_logging(level=settings.log_level, force=True)
        settings.ensure_directories()
        try:
            init_db(settings)
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            logger.error("Database initialisation failed: %s", exc)
            st.error(f"Could not initialise the database: {exc}")
            st.stop()
        st.session_state[KEY_INIT] = True
        _log_banner(settings)

    settings = get_settings()
    try:
        bundle = get_bundle(settings)
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        logger.exception("Could not build the agent bundle")
        st.error(f"Startup failed while loading the AI stack: {exc}")
        if settings.debug:
            st.exception(exc)
        st.stop()

    bundle.nexus.set_document_state(not bundle.pipeline.is_empty)
    return AppContext(settings=settings, bundle=bundle)


def _log_banner(settings: Settings) -> None:
    logger.info(
        "[ui] %s starting (provider=%s, model=%s, store=%s)",
        settings.app_name,
        settings.llm_provider,
        settings.groq_model if settings.has_groq_key else "offline-extractive-v1",
        settings.vector_store_provider,
    )
    if not settings.has_groq_key:
        logger.info(
            "[ui] GROQ_API_KEY is not set - running in offline extractive mode. Answers "
            "are assembled from your own documents instead of a hosted model."
        )
    if not crew_enabled(settings):
        logger.info("[ui] CrewAI orchestration unavailable; agents run directly.")


# ---------------------------------------------------------------------------
# Transcript (in-session rendering only; the durable copy lives in SQLite)
# ---------------------------------------------------------------------------
def history() -> list[dict[str, Any]]:
    """The transcript rendered by the chat pages."""
    return st.session_state.setdefault(KEY_CHAT, [])


def push_turn(role: str, content: str, **meta: Any) -> None:
    """Append a transcript entry (role is ``user`` or ``assistant``)."""
    history().append({"role": role, "content": content, **meta})


def clear_chat() -> None:
    for key in (KEY_CHAT, KEY_TURN, KEY_TRACE, KEY_LAST, KEY_TRACE_NOTES):
        st.session_state.pop(key, None)


def set_trace(stages: list[str], notes: dict[str, str] | None = None) -> None:
    """Remember the pipeline a request went through, for the trace panel."""
    st.session_state[KEY_TRACE] = list(stages)
    st.session_state[KEY_TRACE_NOTES] = dict(notes or {})


def current_trace() -> tuple[list[str], dict[str, str]]:
    return list(st.session_state.get(KEY_TRACE) or []), dict(
        st.session_state.get(KEY_TRACE_NOTES) or {}
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def record_turn(
    context: AppContext,
    result: Any,
    *,
    question: str,
    mode: str,
    answer: str | None = None,
    citations: list[Any] | None = None,
    minutes: float | None = None,
    subject: str | None = None,
    topic: str | None = None,
    correct: bool | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Persist a completed agent turn to memory, progress and the transcript.

    Everything happens inside one database transaction so a turn is either fully
    recorded or not recorded at all.
    """
    payload = answer if answer is not None else (getattr(result, "answer", "") or "")
    sources = citations if citations is not None else (getattr(result, "sources", None) or [])
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    grounding = [str(getattr(g, "value", g)) for g in getattr(result, "grounding", None) or []]
    intent = str(getattr(result, "intent", mode) or mode)

    push_turn("user", question, intent=intent)
    push_turn(
        "assistant",
        payload,
        intent=intent,
        agent=str(getattr(result, "agent", mode)),
        confidence=confidence,
        sources=list(sources),
    )
    st.session_state[KEY_LAST] = result
    context.log_action("turn", intent=intent, confidence=confidence)

    try:
        with context.session() as repositories:
            profile = LearnerProfile(repositories)
            memory = ConversationMemory(
                repositories.session, repositories.conversations, context.bundle.llm
            )
            profile.note_session(mode=mode, subject=subject, minutes=minutes)
            for message in warnings or getattr(result, "warnings", None) or []:
                logger.info("[ui] agent warning (%s): %s", mode, message)
            memory.add_turn(
                context.conversation_id,
                "user",
                question,
                agent="nexus",
                intent=intent,
            )
            memory.add_turn(
                context.conversation_id,
                "assistant",
                payload,
                agent=str(getattr(result, "agent", mode)),
                intent=intent,
                confidence=confidence,
                citations=[
                    {
                        "marker": getattr(source, "marker", ""),
                        "document_name": getattr(source, "document_name", ""),
                        "page": getattr(source, "page", None),
                    }
                    for source in sources
                ],
                grounding=grounding,
                mode=mode,
            )
            if topic and correct is not None:
                ProgressTracker(repositories, repositories.session).record_answer(
                    topic, correct=correct, subject=subject or "General"
                )
    except Exception as exc:  # pragma: no cover - never break the UI over memory
        logger.warning("Could not persist the turn: %s", exc)


def record_answers(
    context: AppContext, answers: list[tuple[str, bool]], *, subject: str = "General"
) -> None:
    """Update topic mastery for a graded quiz attempt."""
    if not answers:
        return
    with context.session() as repositories:
        tracker = ProgressTracker(repositories, repositories.session)
        for topic, correct in answers:
            if topic.strip():
                tracker.record_answer(topic, correct=correct, subject=subject)


def note_misconception(context: AppContext, misconception: str) -> None:
    """Add a misconception to the persistent profile."""
    if not misconception.strip():
        return
    with context.session() as repositories:
        LearnerProfile(repositories).record_misconception(misconception.strip())


def set_preferences(context: AppContext, **preferences: Any) -> None:
    """Persist learner preferences (level, style, subject, weekly hours)."""
    with context.session() as repositories:
        LearnerProfile(repositories).set_preferences(**preferences)


def list_conversations(context: AppContext, limit: int = 20) -> list[Any]:
    """Recent conversations, newest first (for the conversation switcher)."""
    with context.session() as repositories:
        return repositories.conversations.list_conversations(repositories.session, limit=limit)


# ---------------------------------------------------------------------------
# Settings changes
# ---------------------------------------------------------------------------
ENV_LOCAL = PROJECT_ROOT / ".env.local"


def apply_settings(overrides: dict[str, Any]) -> Settings:
    """Persist overrides to ``.env.local``, drop caches and re-bootstrap.

    Comments and ordering are preserved so the file stays hand-editable.
    """
    path = ENV_LOCAL
    preserved: list[str] = []
    applied: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.split("=", 1)[0].strip()
            if key in overrides:
                if key not in applied:
                    preserved.append(f"{key}={overrides[key]}")
                    applied.add(key)
            else:
                preserved.append(line)
    for key, value in overrides.items():
        if key not in applied:
            preserved.append(f"{key}={value}")

    path.write_text("\n".join(preserved) + "\n", encoding="utf-8")
    reload_settings()
    reset_bundle()
    reset_vector_store_cache()
    st.session_state[KEY_INIT] = False
    fresh = bootstrap()
    logger.info("[ui] settings updated: %s", sorted(overrides))
    return fresh.settings


def database_health(context: AppContext) -> dict[str, Any]:
    return dict(database_summary(context.settings))


def environment_report(context: AppContext) -> list[tuple[str, str]]:
    """Rows for the Settings page's health check."""
    settings = context.settings
    return [
        ("Python", sys.version.split()[0]),
        ("LLM provider", settings.llm_provider),
        ("Model", settings.groq_model if settings.has_groq_key else "offline-extractive-v1"),
        ("API key", "configured" if settings.has_groq_key else "missing (offline mode)"),
        ("Embeddings", f"{settings.embedding_provider} / {settings.embedding_model}"),
        ("Vector store", settings.vector_store_provider),
        (
            "Hybrid weights",
            f"dense {settings.dense_weight:g} / sparse {settings.sparse_weight:g}",
        ),
        ("Reranker", settings.reranker),
        ("CrewAI", "enabled" if crew_enabled(settings) else "unavailable"),
        (
            "Web research",
            settings.web_search_provider if settings.enable_web_research else "disabled",
        ),
        ("Index", f"{context.bundle.pipeline.chunk_count} chunks"),
        ("Database", settings.database_url or "-"),
    ]


__all__ = [
    "AppContext",
    "apply_settings",
    "bootstrap",
    "navigate",
    "clear_chat",
    "current_trace",
    "database_health",
    "environment_report",
    "history",
    "list_conversations",
    "note_misconception",
    "push_turn",
    "record_answers",
    "record_turn",
    "set_preferences",
    "set_trace",
]