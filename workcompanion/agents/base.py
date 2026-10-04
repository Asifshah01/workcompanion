"""Shared agent infrastructure.

Every agent returns an :class:`~workcompanion.schemas.common.AgentResult` built
through :class:`AgentBase`, so callers get consistent envelopes, latency/token
accounting, caching and error handling regardless of which agent ran.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterator, Sequence
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.llm.base import (
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
)
from workcompanion.llm.json_utils import extract_json
from workcompanion.schemas.common import (
    AgentResult,
    Citation,
    ConfidenceLevel,
    failure_result,
)
from workcompanion.utils.cache import DiskCache, cache_key
from workcompanion.utils.telemetry import get_telemetry

logger = get_logger(__name__)


class AgentBase(abc.ABC):
    """Base class for all WorkCompanion agents."""

    #: Stable agent identifier used in logs, results and the UI.
    name: str = "agent"
    #: One-line description shown in the agent trace panel.
    description: str = ""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm
        self._cache = cache
        self._telemetry = get_telemetry()

    # ------------------------------------------------------------------
    @property
    def llm(self) -> LLMProvider | None:
        return self._llm

    @property
    def system_prompt(self) -> str:
        """Agent-specific system prompt (subclasses override)."""
        return (
            "You are a careful academic assistant. Be accurate, concise and never "
            "invent facts or citations."
        )

    @property
    def is_offline(self) -> bool:
        return self._llm is not None and self._llm.name == "offline"

    # ------------------------------------------------------------------
    def build_request(
        self,
        task: str,
        *,
        system: str | None = None,
        user: str = "",
        history: Sequence[Message] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMRequest:
        """Create an :class:`LLMRequest` tagged with the calling task.

        The ``task`` tag lets the offline provider pick a deterministic strategy
        and improves telemetry/cache traceability.
        """
        messages: list[Message] = []
        messages.append(Message(role=Role.SYSTEM, content=system or self.system_prompt))
        messages.extend(history or [])
        messages.append(Message(role=Role.USER, content=user))
        return LLMRequest(
            messages=messages,
            temperature=self._settings.groq_temperature if temperature is None else temperature,
            max_tokens=self._settings.groq_max_tokens if max_tokens is None else max_tokens,
            metadata={"task": task, "agent": self.name},
        )

    # ------------------------------------------------------------------
    def _response_cache(self) -> DiskCache | None:
        if self._cache is None and self._settings.cache_enabled:
            self._cache = DiskCache(
                self._settings.cache_dir / "llm",
                ttl_seconds=self._settings.cache_ttl_seconds,
                enabled=True,
            )
        return self._cache

    def _cache_key(self, request: LLMRequest) -> str:
        return cache_key(
            "llm",
            self._llm.model if self._llm else "none",
            request.metadata.get("task", ""),
            request.temperature,
            request.max_tokens,
            request.prompt_text(),
        )

    # ------------------------------------------------------------------
    def complete(self, request: LLMRequest, *, use_cache: bool = True) -> LLMResponse:
        """Run a completion, transparently using the disk cache."""
        if self._llm is None:
            raise LLMError("No LLM provider is configured for this agent.")
        cache = self._response_cache() if use_cache else None
        key = self._cache_key(request)

        if cache is not None:
            cached = cache.get(key)
            if isinstance(cached, dict) and "text" in cached:
                response = LLMResponse(**{**cached, "cached": True})
                logger.info("[%s] cache hit for task=%s", self.name, request.metadata.get("task"))
                return response

        started = time.perf_counter()
        # Pass the whole LLMRequest (not just its messages) so providers can read
        # request.metadata - the offline provider dispatches on the "task" tag.
        response = self._llm.generate(
            request,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_mode=request.json_mode,
        )
        response.latency_ms = response.latency_ms or (time.perf_counter() - started) * 1000
        self._telemetry.record("agent.latency_ms", response.latency_ms)
        self._telemetry.count(f"agent.{self.name}.calls")

        if cache is not None:
            cache.set(
                key,
                {
                    "text": response.text,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_ms": response.latency_ms,
                },
            )

        if response.truncated and not request.json_mode:
            # Never let a response end mid-sentence without saying so: a
            # truncated answer reads as a complete one, which is worse than an
            # obviously unfinished one. JSON responses are left alone because
            # extra prose would break parsing (and a truncated JSON body already
            # fails to parse, which the caller handles).
            self._telemetry.count("agent.truncated_responses")
            logger.warning(
                "[%s] response hit the output-token limit (task=%s, max_tokens=%s)",
                self.name,
                request.metadata.get("task"),
                request.max_tokens,
            )
            response.text = (
                response.text.rstrip()
                + "\n\n_Answer cut off: it reached the output-token limit. "
                "Ask a follow-up to continue from here._"
            )
        return response

    def structured(self, request: LLMRequest, *, retries: int = 2, use_cache: bool = True) -> dict[str, Any] | None:
        """Run a completion and parse the JSON payload."""
        if self._llm is None:
            raise LLMError("No LLM provider is configured for this agent.")

        cache = self._response_cache() if use_cache else None
        key = self._cache_key(request) + "|json"
        if cache is not None:
            cached = cache.get(key)
            if isinstance(cached, dict):
                return cached

        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                payload = self._llm.generate_json(
                    request,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    retries=1,
                )
            except LLMError as exc:
                if not getattr(exc, "retryable", False):
                    raise
                last_error = exc
                continue
            if isinstance(payload, dict):
                if cache is not None:
                    cache.set(key, payload)
                return payload
            last_error = ValueError("Response was not valid JSON")
            if attempt + 1 < retries:
                request.messages = list(request.messages) + [
                    Message(
                        role=Role.USER,
                        content=(
                            "Your previous reply could not be parsed as JSON. Reply with ONLY "
                            "a valid JSON document - no prose, no markdown fences."
                        ),
                    )
                ]

        logger.warning("[%s] structured output failed: %s", self.name, last_error)
        return None

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Stream a completion (falls back to a single chunk)."""
        if self._llm is None:
            raise LLMError("No LLM provider is configured for this agent.")
        yield from self._llm.stream(
            request, temperature=request.temperature, max_tokens=request.max_tokens
        )

    # ------------------------------------------------------------------
    def result(
        self,
        answer: str,
        *,
        intent: str = "answer",
        sources: Sequence[Citation] | None = None,
        confidence: float = 0.0,
        grounding: Sequence[Any] | None = None,
        latency_ms: float = 0.0,
        warnings: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        response: LLMResponse | None = None,
        success: bool = True,
    ) -> AgentResult:
        """Build a consistent :class:`AgentResult` envelope."""
        return AgentResult(
            success=success,
            agent=self.name,
            intent=intent,
            answer=answer,
            sources=list(sources or []),
            confidence=round(float(confidence), 4),
            confidence_level=ConfidenceLevel.from_score(float(confidence)),
            grounding=list(grounding or []),
            latency_ms=round(latency_ms, 2),
            model=response.model if response else (self._llm.model if self._llm else None),
            prompt_tokens=response.prompt_tokens if response else 0,
            completion_tokens=response.completion_tokens if response else 0,
            warnings=list(warnings or []),
            metadata=dict(metadata or {}),
        )

    def failure(self, message: str, *, intent: str = "error", **kwargs: Any) -> AgentResult:
        """Build a failed result envelope."""
        return failure_result(self.name, message, intent=intent, **kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def safe_json(text: str) -> Any | None:
        """Parse JSON from arbitrary LLM text."""
        return extract_json(text)

    def trace(self, message: str, **fields: Any) -> None:
        """Structured agent-level trace log."""
        logger.info("[%s] %s%s", self.name, message, f" {fields}" if fields else "")

    def describe(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "description": self.description,
            "llm": self._llm.describe() if self._llm else None,
        }