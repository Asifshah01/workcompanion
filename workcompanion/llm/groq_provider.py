"""Groq LLM provider (primary backend).

Wraps the official ``groq`` SDK behind :class:`~workcompanion.llm.base.LLMProvider`
and adds: lazy import, exponential backoff on rate limits, timeout mapping and
token accounting.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Iterator, Sequence
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings
from workcompanion.llm.base import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    Message,
)
from workcompanion.utils.telemetry import get_telemetry

logger = get_logger(__name__)

#: Groq tells us exactly how long to wait ("Please try again in 30.36s"). On the
#: free tier the output-token budget is ~1k tokens/minute, so these limits are
#: hit routinely and obeying them is the difference between a working app and a
#: stream of errors.
_RETRY_AFTER_RE = re.compile(r"try again in\s*([\d.]+)\s*(m?s)", re.IGNORECASE)
_RETRY_AFTER_HEADER = "retry-after"

_RATE_LIMIT_MARKERS = ("rate_limit", "429", "too many requests", "quota", "overloaded", "503")

#: 4xx codes that will never succeed on a retry: the request itself is wrong, or
#: the credential is not accepted. Retrying these only adds backoff delay to a
#: failure the caller has to handle anyway.
_PERMANENT_MARKERS = (
    "400",
    "401",
    "403",
    "404",
    "405",
    "413",
    "415",
    "422",
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "model_decommissioned",
)


class GroqProvider(LLMProvider):
    """Chat-completions provider backed by Groq."""

    name = "groq"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Any | None = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self._settings.groq_model

    @property
    def available(self) -> bool:
        return self._settings.has_groq_key

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.has_groq_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example) "
                "or switch the provider to 'offline' mode."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMConfigurationError(
                "The 'groq' package is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self._client = Groq(
            api_key=self._settings.groq_api_key,
            timeout=self._settings.groq_timeout_seconds,
            max_retries=0,  # we implement our own backoff
        )
        return self._client

    # ------------------------------------------------------------------
    def _to_request(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        stop: Sequence[str] | None,
        model_override: str | None,
        seed: int | None,
    ) -> tuple[LLMRequest, str]:
        settings = self._settings
        if isinstance(messages, LLMRequest):
            request = messages
        else:
            request = LLMRequest(messages=list(messages))
        if temperature is not None:
            request.temperature = temperature
        if max_tokens is not None:
            # ``GROQ_MAX_TOKENS`` is the real ceiling. Agents ask for more
            # (a detailed answer needs ~1800) and on a rate-limited tier that
            # exceeds the whole per-minute budget, so the request is rejected
            # before it is even sent.
            request.max_tokens = min(max_tokens, settings.groq_max_tokens)
        if stop:
            request.stop = list(stop)
        request.json_mode = json_mode or request.json_mode
        model = model_override or request.model_override or settings.groq_model
        return request, model

    @staticmethod
    def _classify(error: Exception) -> LLMError:
        """Map a provider exception onto one of our error types.

        Order matters: a timeout must win over the incidental digits in an HTTP
        status, and rate limits must win over the generic permanent markers.
        """
        text = f"{type(error).__name__}: {error}".lower()
        if isinstance(error, TimeoutError) or "timeout" in text:
            return LLMTimeoutError(f"Groq request timed out: {error}")
        if any(marker in text for marker in _RATE_LIMIT_MARKERS):
            return LLMRateLimitError(f"Groq rate limit/quota error: {error}")
        if any(marker in text for marker in _PERMANENT_MARKERS):
            permanent = LLMProviderError(f"Groq rejected the request: {error}")
            permanent.retryable = False
            return permanent
        return LLMProviderError(f"Groq API error: {error}")

    # ------------------------------------------------------------------
    @staticmethod
    def _server_requested_delay(error: Exception) -> float | None:
        """How long the provider asked us to wait, if it said.

        Prefers the ``Retry-After`` header, then the message text. Returns
        ``None`` when the provider gave no guidance.
        """
        response = getattr(error, "response", None) or getattr(
            error, "http_response", None
        )
        headers = getattr(response, "headers", None)
        if headers:
            raw = headers.get(_RETRY_AFTER_HEADER)
            if raw:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        match = _RETRY_AFTER_RE.search(str(error))
        if match:
            amount = float(match.group(1))
            return amount * 60 if match.group(2).lower() == "m" else amount
        return None

    def _backoff(self, error: LLMError, attempt: int, cause: Exception) -> float:
        """Delay before the next attempt.

        A provider-supplied wait always wins: backing off for 8 seconds when the
        server said 30 simply guarantees another 429.
        """
        requested = self._server_requested_delay(cause)
        if requested is not None:
            # Cap it so one bad response cannot hang the UI for minutes.
            return max(0.5, min(requested, 45.0))
        return min(2.0 * (2**attempt), 20.0) + random.uniform(0, 0.6)

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        model_override: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request, model = self._to_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stop=stop,
            model_override=model_override,
            seed=seed,
        )
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.as_payload() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or settings.groq_max_tokens,
            "top_p": request.top_p,
        }
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        telemetry = get_telemetry()
        last_error: LLMError | None = None
        # ``total_started`` spans every attempt, so the number reported to the
        # user includes the backoff sleeps they actually sat through.
        total_started = time.perf_counter()
        attempts_made = 0
        for attempt in range(self._settings.groq_max_retries + 1):
            started = time.perf_counter()
            attempts_made = attempt + 1
            try:
                completion = client.chat.completions.create(**payload)
            except Exception as exc:  # SDK raises a family of APIError subclasses
                error = self._classify(exc)
                last_error = error
                telemetry.count("llm.errors")
                if not error.retryable or attempt >= self._settings.groq_max_retries:
                    raise error from exc
                delay = self._backoff(error, attempt, exc)
                logger.warning(
                    "Groq call failed (%s); retrying in %.1fs (%d/%d)",
                    error.__class__.__name__, delay, attempt + 1, self._settings.groq_max_retries,
                )
                time.sleep(delay)
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            total_ms = (time.perf_counter() - total_started) * 1000
            telemetry.record("llm.latency_ms", latency_ms)
            telemetry.record("llm.total_latency_ms", total_ms)
            telemetry.count("llm.calls")
            if attempts_made > 1:
                telemetry.count("llm.retried_calls")
                logger.info("Groq recovered after %d attempts", attempts_made)
            choice = completion.choices[0]
            usage = getattr(completion, "usage", None)
            response = LLMResponse(
                text=(choice.message.content or "").strip(),
                model=getattr(completion, "model", model),
                provider=self.name,
                finish_reason=getattr(choice, "finish_reason", None),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                latency_ms=latency_ms,
                raw={"id": getattr(completion, "id", None), "attempts": attempts_made},
            )
            logger.info(
                "Groq completion model=%s tokens=%d latency=%.0fms total=%.0fms attempts=%d",
                response.model, response.total_tokens, latency_ms, total_ms, attempts_made,
            )
            return response

        raise last_error or LLMProviderError("Groq request failed for an unknown reason")

    # ------------------------------------------------------------------
    def stream(
        self,
        messages: Sequence[Message] | LLMRequest,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        request, model = self._to_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            stop=stop,
            model_override=kwargs.pop("model_override", None),
            seed=kwargs.pop("seed", None),
        )
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.as_payload() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or self._settings.groq_max_tokens,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.stop:
            payload["stop"] = request.stop

        telemetry = get_telemetry()
        started = time.perf_counter()
        emitted = 0
        retries = self._settings.groq_max_retries
        attempt = 0
        while True:
            try:
                for event in client.chat.completions.create(**payload):
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    piece = getattr(delta, "content", None)
                    if piece:
                        emitted += len(piece)
                        yield piece
                telemetry.record("llm.stream_latency_ms", (time.perf_counter() - started) * 1000)
                telemetry.count("llm.stream_calls")
                return
            except Exception as exc:
                error = self._classify(exc)
                attempt += 1
                if emitted or not error.retryable or attempt > retries:
                    raise error from exc
                time.sleep(self._backoff(error, attempt - 1, exc))