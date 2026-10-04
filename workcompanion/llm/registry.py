"""Provider registry / factory.

Adding a backend is a two-step change: implement :class:`LLMProvider`, then add
it to :data:`PROVIDER_FACTORIES`.  Nothing else in the application changes.
"""

from __future__ import annotations

from collections.abc import Callable

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import LLMConfigurationError, LLMProvider
from workcompanion.llm.groq_provider import GroqProvider
from workcompanion.llm.offline_provider import OfflineProvider

logger = get_logger(__name__)

PROVIDER_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {
    "groq": GroqProvider,
    "offline": lambda settings: OfflineProvider(),
}

_CACHE: dict[str, LLMProvider] = {}


def available_providers() -> list[str]:
    """Names of all registered providers."""
    return sorted(PROVIDER_FACTORIES)


def get_llm(
    settings: Settings | None = None,
    *,
    provider: str | None = None,
    use_cache: bool = True,
    allow_fallback: bool = True,
) -> LLMProvider:
    """Return the configured LLM provider.

    Args:
        settings: settings override (defaults to the process singleton).
        provider: explicit provider name; overrides ``settings.llm_provider``.
        use_cache: reuse the cached instance for the same key.
        allow_fallback: when the requested provider is unusable (e.g. no API
            key), fall back to the offline provider instead of raising. Set to
            ``False`` to fail loudly.

    Raises:
        LLMConfigurationError: provider unknown, or unavailable and fallback
            is disabled.
    """
    settings = settings or get_settings()
    name = (provider or settings.llm_provider or "groq").lower()

    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        raise LLMConfigurationError(
            f"Unknown LLM provider '{name}'. Available: {', '.join(available_providers())}"
        )

    cache_key = f"{name}|{settings.groq_model}"
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    instance = factory(settings)
    if not instance.available:
        message = (
            f"LLM provider '{name}' is not configured "
            f"({'GROQ_API_KEY missing' if name == 'groq' else 'unavailable'})."
        )
        if not allow_fallback:
            raise LLMConfigurationError(message)
        logger.warning("%s Falling back to the offline extractive provider.", message)
        instance = OfflineProvider()
        cache_key = "offline|offline-extractive-v1"

    if use_cache:
        _CACHE[cache_key] = instance
    logger.info("LLM provider ready: %s", instance.describe())
    return instance


def reset_llm_cache() -> None:
    """Drop cached provider instances (used by the Settings page and tests)."""
    _CACHE.clear()