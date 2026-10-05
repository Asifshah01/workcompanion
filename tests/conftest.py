"""Shared fixtures.

Every test runs against a throwaway data directory and the deterministic
offline provider, so the suite is reproducible, needs no API key, and does not
spend anyone's rate-limit budget.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Applied before anything imports ``workcompanion`` so the offline provider is
# baked into every Settings instance built during the session.
_OFFLINE_ENV = {
    "LLM_PROVIDER": "offline",
    "ENABLE_CREWAI": "false",
    "ENABLE_WEB_RESEARCH": "false",
    "CACHE_ENABLED": "false",
    "EMBEDDING_CACHE_ENABLED": "false",
    "TELEMETRY_ENABLED": "false",
    "LOG_LEVEL": "ERROR",
}


@pytest.fixture(autouse=True)
def _offline_environment() -> Iterator[None]:
    """Force the offline provider and silence logging for the whole session."""
    previous = {key: os.environ.get(key) for key in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    os.environ.pop("GROQ_API_KEY", None)
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def settings(tmp_path: Path):  # noqa: ANN201 - pydantic type is not worth importing
    """A Settings instance rooted entirely inside ``tmp_path``."""
    from workcompanion.config.settings import Settings

    return Settings(
        data_dir=tmp_path,
        documents_dir=tmp_path / "documents",
        vector_db_path=tmp_path / "vectorstore",
        cache_dir=tmp_path / "cache",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        vector_store_provider="memory",
        llm_provider="offline",
        cache_enabled=False,
        enable_web_research=False,
        log_level="ERROR",
        # 'auto' would download a cross-encoder on first use, which is far too
        # slow and network-dependent for a unit test. 'local' is deterministic.
        reranker="local",
        _env_file=None,
    )


#: Long enough to exercise chunking, short enough to stay readable.
SAMPLE_NOTES = """# Thermodynamics - Lecture 3

## The Second Law
The second law of thermodynamics states that the entropy of an isolated system
never decreases. A spontaneous process has a positive total entropy change.
Heat flows spontaneously from hot to cold, never the reverse.

## Gibbs Free Energy
At constant temperature and pressure, spontaneity is decided by the Gibbs free
energy, G = H - T S. A process is spontaneous when the change in G is negative.
A process at equilibrium has dG equal to zero. Outside constant T and P the
correct potential is the Helmholtz free energy A = U - T S.

## Enthalpy and Phase Changes
Enthalpy absorbs the energy of a phase change without a temperature change.
This energy is latent heat. The latent heat of fusion of water at 0 degrees
Celsius is about 334 kJ/kg.
"""


@pytest.fixture()
def notes_file(tmp_path: Path) -> Path:
    """A markdown study note on disk, ready to ingest."""
    directory = tmp_path / "documents"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "thermodynamics.md"
    path.write_text(SAMPLE_NOTES, encoding="utf-8")
    return path


@pytest.fixture()
def bundle(settings):  # noqa: ANN201 - see settings
    """A fully wired AgentBundle on temporary storage."""
    from workcompanion.agents.bundle import build_bundle

    return build_bundle(settings)


@pytest.fixture()
def seeded(bundle, notes_file):
    """A bundle with one thermodynamics note already indexed.

    Grounded agents (quiz, flashcards, research) refuse to generate anything
    when nothing relevant is in the index - that refusal is the hallucination
    control working, not a defect - so tests that need grounded output use
    this fixture while plain orchestration tests use ``bundle``.
    """
    result = bundle.ingestion.ingest_file(notes_file, subject="Physics")
    assert result.ok, result.error
    return bundle
