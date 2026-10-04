"""Bridge between WorkCompanion's provider layer and CrewAI.

Groq exposes an OpenAI-compatible API, and CrewAI reaches any such endpoint via
its natively supported ``openai`` provider plus a custom ``base_url``. That path
needs no LiteLLM, so a crew works with the dependencies in ``requirements.txt``
alone. Everything else in the app uses
:class:`workcompanion.llm.base.LLMProvider`; this module is the only place the
two meet.

The builder never raises: when CrewAI is not installed, the setting is off, or
no API key is present it returns ``None`` and the caller falls back to calling
agents directly.
"""

from __future__ import annotations

from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings

logger = get_logger(__name__)

#: Groq's OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

#: Tried in order. ``openai/`` is CrewAI's own provider and always works;
#: ``groq/`` needs LiteLLM, so it is only reached if that is installed.
MODEL_PREFIXES = ("openai/", "groq/")


def crewai_available() -> bool:
    """Whether the optional ``crewai`` dependency can be imported."""
    try:
        import crewai  # noqa: F401
    except Exception:
        return False
    return True


def crew_enabled(settings: Settings | None = None) -> bool:
    """Whether crews may run at all (installed, enabled, and a key is present)."""
    settings = settings or get_settings()
    if not settings.enable_crewai:
        return False
    if not crewai_available():
        return False
    return bool(settings.groq_api_key)


def build_crew_llm(
    settings: Settings | None = None, *, temperature: float | None = None
) -> Any | None:
    """Build a CrewAI ``LLM`` bound to Groq, or ``None`` when unavailable."""
    settings = settings or get_settings()
    if not settings.groq_api_key:
        logger.info("[crew] no GROQ_API_KEY configured; crews stay disabled.")
        return None
    try:
        from crewai.llm import LLM
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("[crew] crewai is not installed (%s); crews stay disabled.", exc)
        return None

    bare = settings.groq_model
    for prefix in MODEL_PREFIXES:
        model = bare if bare.startswith(prefix) else f"{prefix}{bare}"
        try:
            return LLM(
                model=model,
                api_key=settings.groq_api_key,
                base_url=GROQ_BASE_URL,
                temperature=(
                    settings.groq_temperature if temperature is None else temperature
                ),
                max_tokens=settings.groq_max_tokens,
                timeout=settings.groq_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - misconfiguration
            logger.warning(
                "[crew] CrewAI rejected model %r (%s); trying the next prefix.",
                model,
                str(exc).splitlines()[0][:120],
            )
    logger.warning("[crew] no usable CrewAI LLM; crews will run the direct-agent fallback.")
    return None