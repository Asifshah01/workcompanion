"""Flashcard schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from workcompanion.schemas.common import Citation

CardDifficulty = Literal["easy", "medium", "hard", "expert"]


class Flashcard(BaseModel):
    """A single question/answer pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    front: str
    back: str
    topic: str = "General"
    difficulty: CardDifficulty = "medium"
    hint: str | None = None
    tags: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)


class FlashcardDeck(BaseModel):
    """A generated deck plus its provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str = "Flashcard deck"
    subject: str = "General"
    cards: list[Flashcard] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.cards)

    def by_topic(self) -> dict[str, list[Flashcard]]:
        grouped: dict[str, list[Flashcard]] = {}
        for card in self.cards:
            grouped.setdefault(card.topic, []).append(card)
        return grouped