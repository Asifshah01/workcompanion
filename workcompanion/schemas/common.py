"""Shared agent-level contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExplanationLevelName = Literal["beginner", "undergraduate", "graduate", "expert"]


class SourceKind(str, Enum):
    """Where a piece of evidence came from - never mix these silently."""

    DOCUMENT = "document"
    WEB = "web"
    GENERAL_KNOWLEDGE = "general_knowledge"
    USER_INPUT = "user_input"
    LEARNER_HISTORY = "learner_history"


class GroundingLabel(str, Enum):
    """Hallucination-control labels attached to claims in an answer."""

    RETRIEVED_FACT = "retrieved_fact"
    MODEL_REASONING = "model_reasoning"
    GENERAL_KNOWLEDGE = "general_knowledge"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"high": 3, "medium": 2, "low": 1}[self.value]

    @staticmethod
    def from_score(score: float) -> "ConfidenceLevel":
        if score >= 0.62:
            return ConfidenceLevel.HIGH
        if score >= 0.34:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


class Citation(BaseModel):
    """A verifiable pointer back into the user's own material."""

    model_config = ConfigDict(frozen=True)

    marker: str = Field(..., description="In-text marker such as [1].")
    document_id: str | None = None
    document_name: str = "Unknown source"
    page: int | None = None
    section: str | None = None
    chapter: str | None = None
    chunk_id: str | None = None
    source_kind: SourceKind = SourceKind.DOCUMENT
    url: str | None = None
    quote: str | None = None
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)

    def label(self) -> str:
        """Human-readable citation label used in the UI."""
        parts: list[str] = []
        if self.source_kind is SourceKind.WEB:
            parts.append(f"[web] {self.document_name}")
            if self.url:
                parts.append(self.url)
        else:
            parts.append(self.document_name)
            location: list[str] = []
            if self.page is not None:
                location.append(f"p. {self.page}")
            if self.section:
                location.append(self.section)
            if location:
                parts.append(", ".join(location))
        return f"{self.marker} " + " \u2014 ".join(parts)

    def to_markdown(self) -> str:
        return f"> **{self.label()}**"


class AgentResult(BaseModel):
    """Uniform envelope returned by every agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    agent: str = "unknown"
    intent: str = "unknown"
    answer: str = ""
    sources: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    grounding: list[GroundingLabel] = Field(default_factory=list)
    latency_ms: float = 0.0
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence_level", mode="before")
    @classmethod
    def _default_confidence_level(cls, value: Any, info: Any) -> Any:
        return value or ConfidenceLevel.from_score(float(info.data.get("confidence", 0.0)))

    def with_level(self) -> "AgentResult":
        """Recompute ``confidence_level`` from the numeric score."""
        self.confidence_level = ConfidenceLevel.from_score(self.confidence)
        return self

    def citation_markdown(self) -> str:
        if not self.sources:
            return ""
        lines = ["", "**Sources**"]
        for citation in self.sources:
            lines.append(f"- {citation.label()}")
        return "\n".join(lines)


def success_result(agent: str, answer: str, **kwargs: Any) -> AgentResult:
    """Terse constructor for a successful :class:`AgentResult`."""
    return AgentResult(success=True, agent=agent, answer=answer, **kwargs).with_level()


def failure_result(agent: str, message: str, **kwargs: Any) -> AgentResult:
    """Terse constructor for a failed :class:`AgentResult`."""
    kwargs.setdefault("warnings", [message])
    return AgentResult(
        success=False, agent=agent, answer=message, confidence=0.0, **kwargs
    ).with_level()