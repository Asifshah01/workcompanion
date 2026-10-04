"""One place that wires every agent, store and memory together.

The Streamlit pages, the CLI scripts and the CrewAI crews all need the same
object graph (one vector store, one BM25 index, one provider, one database
session per request).  Building it here means the wiring exists exactly once -
adding a dependency to an agent is a change in this file, not in five others.

The bundle is intentionally *lazy*: Streamlit re-runs the whole script on every
interaction, so embedding models and Chroma connections must not be rebuilt on
each run.  :func:`get_bundle` caches the shared parts and only recreates the
per-session database handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.agents.flashcard_agent import FlashcardAgent
from workcompanion.agents.nexus_agent import NexusAgent
from workcompanion.agents.planner_agent import PlannerAgent
from workcompanion.agents.quiz_agent import QuizAgent
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.agents.research_agent import ResearchAgent
from workcompanion.agents.socratic_agent import SocraticAgent
from workcompanion.agents.tutor_agent import TutorAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMProvider
from workcompanion.llm.registry import get_llm
from workcompanion.memory.conversation_memory import ConversationMemory
from workcompanion.memory.learner_profile import LearnerProfile
from workcompanion.memory.progress_tracker import ProgressTracker
from workcompanion.rag.embeddings import EmbeddingProvider, get_embedding_provider
from workcompanion.rag.ingestion import IngestionService
from workcompanion.rag.pipeline import RAGPipeline, get_rag_pipeline
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import VectorStore, get_vector_store
from workcompanion.tools.web_search import WebSearchTool
from workcompanion.utils.cache import DiskCache

logger = get_logger(__name__)


@dataclass
class AgentBundle:
    """Every long-lived object the application needs, wired together."""

    settings: Settings
    llm: LLMProvider
    embeddings: EmbeddingProvider
    store: VectorStore
    sparse: BM25Index
    pipeline: RAGPipeline
    ingestion: IngestionService
    cache: DiskCache
    web_search: WebSearchTool

    nexus: NexusAgent = field(init=False)
    rag: RagAgent = field(init=False)
    tutor: TutorAgent = field(init=False)
    socratic: SocraticAgent = field(init=False)
    research: ResearchAgent = field(init=False)
    quiz: QuizAgent = field(init=False)
    flashcards: FlashcardAgent = field(init=False)
    planner: PlannerAgent = field(init=False)

    profile: LearnerProfile | None = None
    memory: ConversationMemory | None = None
    progress: Any = None

    def __post_init__(self) -> None:
        shared = dict(settings=self.settings, llm=self.llm, cache=self.cache)
        self.nexus = NexusAgent(**shared)
        self.rag = RagAgent(self.pipeline, **shared)
        self.tutor = TutorAgent(self.rag, **shared)
        self.socratic = SocraticAgent(self.rag, **shared)
        self.research = ResearchAgent(self.rag, web_search=self.web_search, **shared)
        self.quiz = QuizAgent(self.rag, **shared)
        self.flashcards = FlashcardAgent(self.rag, **shared)
        self.planner = PlannerAgent(**shared)
        self.nexus.set_document_state(not self.pipeline.is_empty)

    # ------------------------------------------------------------------
    def attach_memory(self, repos: Any, *, conversation_id: str | None = None) -> None:
        """Bind the learner profile, conversation memory and progress tracker.

        ``repos`` is a :class:`~workcompanion.database.repositories.RepositoryBundle`,
        which already holds the active SQLAlchemy session.
        """
        self.profile = LearnerProfile(repos)
        self.memory = ConversationMemory(repos.session, repos.conversations, self.llm)
        self.progress = ProgressTracker(repos, repos.session)
        self.tutor.set_profile(self.profile.snapshot())

    # ------------------------------------------------------------------
    def agents(self) -> dict[str, AgentBase]:
        """Name -> agent, for the UI trace panel and health checks."""
        return {
            self.nexus.name: self.nexus,
            self.rag.name: self.rag,
            self.tutor.name: self.tutor,
            self.socratic.name: self.socratic,
            self.research.name: self.research,
            self.quiz.name: self.quiz,
            self.flashcards.name: self.flashcards,
            self.planner.name: self.planner,
        }

    def status(self) -> dict[str, Any]:
        """Health snapshot for the dashboard / Settings page."""
        return {
            "llm": self.llm.describe(),
            "embeddings": getattr(self.embeddings, "describe", lambda: {})(),
            "chunks": self.pipeline.chunk_count,
            "has_documents": not self.pipeline.is_empty,
            "web_enabled": self.web_search.enabled,
            "cache": self.cache.stats() if hasattr(self.cache, "stats") else {},
            "offline": self.llm.name == "offline",
        }

    def reset(self) -> None:
        """Drop every indexed document (used after a destructive re-index)."""
        self.store.reset()
        self.sparse.clear()
        self.nexus.set_document_state(not self.pipeline.is_empty)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_SHARED: dict[str, Any] = {}


def _shared_key(settings: Settings) -> str:
    return (
        f"{settings.vector_store_provider}|{settings.vector_store_collection}|"
        f"{settings.embedding_model}|{settings.llm_provider}|{settings.groq_model}"
    )


def build_bundle(settings: Settings | None = None) -> AgentBundle:
    """Construct a bundle from scratch (no caching)."""
    settings = settings or get_settings()
    settings.ensure_directories()

    llm = get_llm(settings)
    embeddings = get_embedding_provider(settings)
    store = get_vector_store(settings)
    sparse = _load_sparse_index(settings)
    pipeline = get_rag_pipeline(store, sparse, embeddings, llm, settings=settings)
    ingestion = IngestionService(store, sparse, embeddings, settings=settings)
    cache = DiskCache(
        settings.cache_dir / "llm",
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
    )
    return AgentBundle(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        store=store,
        sparse=sparse,
        pipeline=pipeline,
        ingestion=ingestion,
        cache=cache,
        web_search=WebSearchTool(settings),
    )


def get_bundle(settings: Settings | None = None) -> AgentBundle:
    """Return the process-wide bundle, rebuilding it only when settings change."""
    settings = settings or get_settings()
    key = _shared_key(settings)
    bundle = _SHARED.get(key)
    if bundle is None:
        bundle = build_bundle(settings)
        _SHARED[key] = bundle
        logger.info(
            "[bundle] built (provider=%s, chunks=%d)", bundle.llm.name, bundle.pipeline.chunk_count
        )
    return bundle


def reset_bundle(settings: Settings | None = None) -> None:
    """Forget the cached bundle (Settings page, tests)."""
    if settings is None:
        _SHARED.clear()
        return
    _SHARED.pop(_shared_key(settings), None)


def _load_sparse_index(settings: Settings) -> BM25Index:
    """Load the persisted BM25 index, if any."""
    index = BM25Index()
    path = settings.data_dir / "bm25_index.json"
    if path.exists():
        try:
            index.load(path)
            logger.info("[bundle] loaded BM25 index with %d documents", index.size)
        except Exception as exc:
            logger.warning("[bundle] BM25 index could not be loaded (%s); starting empty.", exc)
    return index


__all__ = ["AgentBundle", "build_bundle", "get_bundle", "reset_bundle"]