"""Core LLM provider interface, request/response models and error taxonomy."""

from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from workcompanion.utils.token_utils import count_tokens, estimate_tokens


class LLMError(RuntimeError):
    """Base class for every LLM failure surfaced to the application."""

    retryable: bool = False


class LLMConfigurationError(LLMError):
    """Missing or invalid provider configuration (e.g. absent API key)."""


class LLMRateLimitError(LLMError):
    """Provider rate limit / quota exceeded."""

    retryable: bool = True


class LLMTimeoutError(LLMError):
    """Provider request timed out."""

    retryable: bool = True


class LLMProviderError(LLMError):
    """Any other provider-side failure (5xx, bad gateway, content filter...)."""

    retryable: bool = True


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(frozen=True)

    role: Role = Role.USER
    content: str = ""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def as_tuple(self) -> tuple[str, str]:
        return self.role.value, self.content

    def as_payload(self) -> dict[str, str]:
        """Wire format for the OpenAI-compatible chat API.

        Providers expect a mapping with a ``role`` discriminator, not a
        positional tuple.
        """
        return {"role": self.role.value, "content": self.content}


class SystemMessage(Message):
    role: Role = Role.SYSTEM


class UserMessage(Message):
    role: Role = Role.USER


class AssistantMessage(Message):
    role: Role = Role.ASSISTANT


def build_messages(
    system: str | None = None,
    user: str | Sequence[Message] | str | None = None,
    *,
    history: Sequence[Message] | None = None,
) -> list[Message]:
    """Compose a chat message list from a system prompt, history and user input."""
    messages: list[Message] = []
    if system:
        messages.append(SystemMessage(content=system))
    if history:
        messages.extend(history)
    if isinstance(user, str):
        messages.append(UserMessage(content=user))
    elif isinstance(user, Message):
        messages.append(user)
    elif user:
        messages.extend(user)
    return messages


class LLMRequest(BaseModel):
    """A provider-agnostic completion request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Message]
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=list)
    json_mode: bool = False
    seed: int | None = None
    model_override: str | None = None
    timeout: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_text(self) -> str:
        return "\n".join(f"{m.role.value}: {m.content}" for m in self.messages)

    def estimated_prompt_tokens(self, model: str | None = None) -> int:
        return sum(count_tokens(m.content, model) for m in self.messages)


class LLMResponse(BaseModel):
    """A normalised completion result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    model: str = ""
    provider: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def truncated(self) -> bool:
        """Whether the provider stopped because it hit the output-token cap."""
        return (self.finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}
    cached: bool = False
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(abc.ABC):
    """Interface every LLM backend implements.

    Subclasses must implement :meth:`generate`; :meth:`stream` defaults to
    chunked delivery of :meth:`generate` so simple backends work immediately.
    """

    #: Stable provider identifier used in logs and the UI.
    name: str = "base"

    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Model identifier currently in use."""

    @property
    def available(self) -> bool:
        """Whether the provider is usable right now."""
        return True

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a complete completion."""

    def stream(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text incrementally.

        The default implementation degrades gracefully to a single chunk so that
        every provider supports streaming from the UI's perspective.
        """
        response = self.generate(messages, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)
        if response.text:
            yield response.text

    def generate_json(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any] | None:
        """Return a parsed JSON object.

        Uses provider JSON mode when available and always runs the result
        through :func:`workcompanion.llm.json_utils.extract_json`, because
        models still occasionally wrap JSON in prose or code fences.
        """
        from workcompanion.llm.json_utils import extract_json

        last_error: Exception | None = None
        # Work on a copy so the caller's request (and its metadata) is untouched.
        working: LLMRequest = (
            messages.model_copy(deep=True)
            if isinstance(messages, LLMRequest)
            else LLMRequest(messages=list(messages))
        )

        for attempt in range(max(1, retries)):
            try:
                response = self.generate(
                    working,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True,
                    **kwargs,
                )
            except LLMError as exc:
                last_error = exc
                if not getattr(exc, "retryable", False):
                    raise
                continue

            parsed = extract_json(response.text)
            if parsed is not None:
                response.raw["json_parsed"] = True
                return parsed
            last_error = ValueError("Model did not return parseable JSON")
            if attempt + 1 < retries:
                working.messages = list(working.messages) + [
                    UserMessage(
                        content=(
                            "Your previous reply could not be parsed as JSON. "
                            "Reply with ONLY a valid JSON document - no prose, no code fences."
                        )
                    )
                ]
        if last_error and isinstance(last_error, LLMError):
            raise last_error
        return None

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Small dict for the Settings page / health checks."""
        return {
            "provider": self.name,
            "model": self.model,
            "available": self.available,
        }


ResponseFormat = Literal["text", "json"]