"""Provider-agnostic LLM abstraction.

The rest of the application only ever talks to :class:`LLMProvider`, so adding
Ollama / OpenAI / Anthropic / Gemini later means implementing one interface and
registering it - no application code changes.
"""

from workcompanion.llm.base import (
    AssistantMessage,
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    Message,
    Role,
    SystemMessage,
    UserMessage,
    build_messages,
)
from workcompanion.llm.registry import available_providers, get_llm, reset_llm_cache

__all__ = [
    "AssistantMessage",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "Message",
    "Role",
    "SystemMessage",
    "UserMessage",
    "build_messages",
    "available_providers",
    "get_llm",
    "reset_llm_cache",
]