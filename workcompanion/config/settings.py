"""Centralised, validated application settings.

Everything tunable lives here so that behaviour can be changed through ``.env``
without touching application code.  Secrets are never logged: :meth:`Settings.safe_dump`
redacts any field whose name contains ``key``/``secret``/``token``/``password``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repository root == parent of the ``workcompanion`` package.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent

_REDACT_TOKENS = ("key", "secret", "token", "password", "credential")


class Settings(BaseSettings):
    """Application settings resolved from the environment / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ app
    app_name: str = "WorkCompanion AI"
    app_tagline: str = "Multi-agent AI study & research companion"
    debug: bool = Field(default=False, description="Show stack traces in the UI.")
    log_level: str = Field(default="INFO")

    # ----------------------------------------------------------------- paths
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    documents_dir: Path | None = Field(default=None)
    vector_db_path: Path | None = Field(default=None)
    cache_dir: Path | None = Field(default=None)

    # ------------------------------------------------------------- database
    database_url: str | None = Field(default=None)
    echo_sql: bool = Field(default=False, description="Log every SQL statement.")

    # ------------------------------------------------------------------ llm
    llm_provider: Literal["groq", "offline"] = Field(
        default="groq", description="'offline' forces the deterministic stub LLM."
    )
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    groq_max_tokens: int = Field(default=2200, ge=128, le=32768)
    groq_timeout_seconds: float = Field(default=90.0, gt=0)
    groq_max_retries: int = Field(default=3, ge=0, le=10)

    # ----------------------------------------------------------- embeddings
    embedding_provider: Literal["sentence-transformers", "hashing", "auto"] = Field(
        default="auto",
        description=(
            "'auto' uses sentence-transformers when installed, otherwise the "
            "dependency-free hashing embedder."
        ),
    )
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=64, ge=1, le=512)

    # --------------------------------------------------------- vector store
    vector_store_provider: Literal["chroma", "memory"] = Field(default="chroma")
    vector_store_collection: str = Field(default="workcompanion")

    # ------------------------------------------------------------ rag tuning
    chunk_target_chars: int = Field(default=1100, ge=200, le=6000)
    chunk_overlap_chars: int = Field(default=180, ge=0, le=1500)
    chunk_min_chars: int = Field(default=180, ge=0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    retrieval_candidate_k: int = Field(default=24, ge=2, le=200)
    multi_query_count: int = Field(default=2, ge=1, le=5)
    context_compression: bool = Field(default=True)
    compressor_max_sentences: int = Field(default=6, ge=1, le=40)
    enable_query_rewrite: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=True)
    reranker: Literal["auto", "cross-encoder", "local", "llm"] = Field(default="auto")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_top_n: int = Field(default=12, ge=1, le=100)
    citation_grounding_check: bool = Field(default=True)
    #: Retrieval confidence below this is treated as "no usable evidence", so the
    #: agents refuse instead of answering from loosely related passages.
    min_retrieval_confidence: float = Field(default=0.15, ge=0.0, le=1.0)

    # ---------------------------------------------------------------- agents
    enable_crewai: bool = Field(default=True)
    crewai_verbose: bool = Field(default=False)
    max_agent_steps: int = Field(default=8, ge=1, le=40)
    default_explanation_level: Literal["beginner", "undergraduate", "graduate", "expert"] = (
        "undergraduate"
    )
    default_teaching_style: Literal[
        "concise", "detailed", "socratic", "example_based", "mathematical", "analogy"
    ] = "detailed"

    # -------------------------------------------------------- web research
    enable_web_research: bool = Field(default=True)
    web_search_provider: Literal["none", "duckduckgo"] = Field(default="duckduckgo")
    web_search_timeout: float = Field(default=15.0, gt=0)
    web_max_results: int = Field(default=5, ge=1, le=20)

    # ---------------------------------------------------------------- ocr
    enable_ocr_fallback: bool = Field(default=True)
    ocr_engine: Literal["auto", "none"] = Field(default="auto")
    ocr_min_chars_per_page: int = Field(default=180, ge=0)
    ocr_max_pages: int = Field(default=25, ge=1, le=500)

    # ----------------------------------------------------------- ingestion
    max_upload_mb: int = Field(default=40, ge=1, le=1024)
    #: ``NoDecode`` stops pydantic-settings from JSON-decoding the raw env value,
    #: so the validator below sees the literal comma-separated string instead.
    allowed_upload_extensions: Annotated[
        tuple[str, ...],
        NoDecode,
    ] = Field(
        default=(
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".markdown",
            ".pptx",
            ".csv",
            ".html",
            ".htm",
        )
    )

    # --------------------------------------------------------------- cache
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=86_400, ge=0)
    embedding_cache_enabled: bool = Field(default=True)

    # ------------------------------------------------------------- telemetry
    telemetry_enabled: bool = Field(default=True)

    # ----------------------------------------------------------------- demo
    demo_mode_autoload: bool = Field(default=True)

    # ------------------------------------------------------------- security
    enable_file_validation: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> Any:
        return str(value).upper() if value is not None else value

    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def _split_extensions(cls, value: Any) -> Any:
        """Accept ``.pdf,.md`` as well as a real list.

        pydantic-settings parses complex types out of the environment as JSON,
        which is awkward to write in a ``.env`` file and inconsistent with how
        the in-app Settings page writes values.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            # A JSON array is still accepted, so an existing .env keeps working.
            if stripped.startswith("["):
                import json

                try:
                    return json.loads(stripped)
                except ValueError:
                    pass
            parts = [item.strip().lower() for item in stripped.split(",")]
            return tuple(item for item in parts if item)
        return value

    @field_validator(
        "documents_dir", "vector_db_path", "cache_dir", "data_dir", mode="after"
    )
    @classmethod
    def _expand(cls, value: Path | None) -> Path | None:
        return Path(os.path.expandvars(str(value))).expanduser() if value else value

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        """Fill in dependent paths and validate cross-field constraints."""
        if self.documents_dir is None:
            self.documents_dir = self.data_dir / "documents"
        if self.vector_db_path is None:
            self.vector_db_path = self.data_dir / "vectorstore"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        if self.database_url is None:
            self.database_url = f"sqlite:///{(self.data_dir / 'database' / 'workcompanion.db').as_posix()}"

        total = self.dense_weight + self.sparse_weight
        if total <= 0:
            raise ValueError("dense_weight and sparse_weight cannot both be zero")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def database_dir(self) -> Path:
        if self.database_url and self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1)).parent
        return self.data_dir / "database"

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.strip())

    def ensure_directories(self) -> None:
        """Create every directory the application writes to."""
        for directory in (
            self.data_dir,
            self.documents_dir,
            self.vector_db_path,
            self.cache_dir,
            self.database_dir,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def safe_dump(self) -> dict[str, Any]:
        """Settings as a dict with secrets redacted (safe for logs / the UI)."""
        data = self.model_dump(mode="json")
        for key, value in list(data.items()):
            if any(token in key.lower() for token in _REDACT_TOKENS) and value:
                data[key] = "***redacted***"
        return data


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache (used by tests and the Settings UI page)."""
    get_settings.cache_clear()
    return get_settings()