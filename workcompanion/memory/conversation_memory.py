"""Conversation memory: rolling window + summarisation.

Sending the entire chat history to the LLM is expensive and usually unhelpful.
This module keeps a small verbatim window plus a rolling summary of everything
older, so long tutoring threads stay cheap and coherent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from workcompanion.config.logging_config import get_logger
from workcompanion.database.repositories import ConversationRepository
from workcompanion.llm.base import LLMProvider, LLMRequest, SystemMessage, UserMessage
from workcompanion.utils.token_utils import count_tokens

logger = get_logger(__name__)

SUMMARY_PROMPT = """Summarise a tutoring conversation so it can continue later. \
Keep concrete facts: concepts covered, definitions agreed, mistakes made, and any \
open questions the learner still has. Be terse - under 120 words. Use short bullets.

Return STRICT JSON: {"summary": "..."}"""


@dataclass(slots=True)
class ConversationContext:
    """What gets injected into an agent prompt."""

    turns: list[tuple[str, str]] = field(default_factory=list)
    summary: str | None = None
    total_messages: int = 0

    def as_pairs(self) -> list[tuple[str, str]]:
        return list(self.turns)

    def render(self, limit: int = 6, *, max_chars: int = 2400) -> str:
        """Render the context for a prompt, newest turns first in priority."""
        parts: list[str] = []
        if self.summary:
            parts.append(f"EARLIER CONVERSATION SUMMARY:\n{self.summary.strip()}")
        if self.turns:
            recent = "\n".join(
                f"{'Learner' if role == 'user' else 'Tutor'}: {content.strip()[:700]}"
                for role, content in self.turns[-limit:]
            )
            parts.append(f"RECENT CONVERSATION:\n{recent}")
        text = "\n\n".join(parts)
        return text[:max_chars] if len(text) > max_chars else text


class ConversationMemory:
    """Persistent, token-aware chat memory."""

    #: Verbatim turns kept in the prompt window.
    WINDOW = 6
    #: Start summarising once the thread exceeds this many messages.
    SUMMARIZE_AFTER = 12
    #: Approximate character budget for the memory block.
    MAX_CHARS = 2400

    def __init__(
        self,
        session: Session,
        repo: ConversationRepository,
        llm: LLMProvider | None = None,
        *,
        window: int | None = None,
    ):
        self._session = session
        self._repo = repo
        self._llm = llm
        self._window = window or self.WINDOW

    # ------------------------------------------------------------------
    def context(self, conversation_id: str) -> ConversationContext:
        """Load the rolling window + stored summary."""
        turns = self._repo.recent_turns(self._session, conversation_id, window=self._window)
        conversation = self._session.get(_conversation_model(), conversation_id)
        summary = conversation.summary if conversation else None
        total = len(self._repo.messages(self._session, conversation_id, limit=500))
        return ConversationContext(turns=turns, summary=summary, total_messages=total)

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        agent: str | None = None,
        intent: str | None = None,
        confidence: float = 0.0,
        citations: Sequence[dict] | None = None,
        grounding: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> None:
        """Persist a single turn."""
        conversation = self._repo.get_or_create(self._session, conversation_id=conversation_id)
        self._repo.add_message(
            self._session,
            conversation,
            role,
            content,
            agent=agent,
            intent=intent,
            confidence=confidence,
            citations=list(citations or []),
            grounding=list(grounding or []),
            mode=mode,
        )

    # ------------------------------------------------------------------
    def maybe_summarize(self, conversation_id: str) -> str | None:
        """Summarise older turns when the thread grows, merging into the summary."""
        conversation = self._repo.get_or_create(self._session, conversation_id=conversation_id)
        messages = self._repo.messages(self._session, conversation_id, limit=500)
        if len(messages) <= self.SUMMARIZE_AFTER:
            return conversation.summary

        older = messages[: -self._window]
        if not older:
            return conversation.summary

        transcript = "\n".join(
            f"{'Learner' if row.role == 'user' else 'Tutor'}: {row.content.strip()[:400]}"
            for row in older
        )
        previous = conversation.summary or ""
        if self._llm is None:
            # Without an LLM, fall back to a cheap deterministic digest.
            conversation.summary = _digest(previous, transcript)
            self._session.flush()
            return conversation.summary

        request = LLMRequest(
            messages=[
                SystemMessage(content=SUMMARY_PROMPT),
                UserMessage(content=f"PREVIOUS SUMMARY:\n{previous or '(none)'}\n\nTRANSCRIPT:\n{transcript}"),
            ],
            temperature=0.0,
            max_tokens=320,
            metadata={"task": "summarize_conversation"},
        )
        try:
            payload = self._llm.generate_json(request, retries=1)
        except Exception as exc:
            logger.warning("Conversation summarisation failed (%s); using the digest.", exc)
            conversation.summary = _digest(previous, transcript)
            self._session.flush()
            return conversation.summary

        if isinstance(payload, dict) and payload.get("summary"):
            conversation.summary = str(payload["summary"])[:2000]
        else:
            conversation.summary = _digest(previous, transcript)
        self._session.flush()
        return conversation.summary

    # ------------------------------------------------------------------
    def estimate_prompt_tokens(self, conversation_id: str) -> int:
        """Token cost of the current memory block (for cost visibility)."""
        return count_tokens(self.context(conversation_id).render(max_chars=self.MAX_CHARS))

    def clear(self, conversation_id: str) -> None:
        self._repo.clear(self._session, conversation_id)


def _digest(previous: str, transcript: str, *, limit: int = 1200) -> str:
    """Deterministic fallback summary: keep the head and tail of the transcript."""
    parts = [part for part in (previous.strip(), transcript.strip()) if part]
    combined = "\n---\n".join(parts)
    if len(combined) <= limit:
        return combined
    half = limit // 2
    return f"{combined[:half]}\n...\n{combined[-half:]}"


def _conversation_model() -> type:
    """Lazily resolve the Conversation ORM class (avoids an import cycle)."""
    from workcompanion.database.models import Conversation

    return Conversation