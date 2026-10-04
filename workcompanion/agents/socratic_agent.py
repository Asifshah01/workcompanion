"""Socratic Tutor Agent.

Never opens with the answer.  Instead it runs a scaffolded dialogue:

    guiding question -> evaluate -> hint -> harder question -> ... -> reveal

The scaffold level rises with each turn so the learner is walked to the answer
instead of being handed it.
"""

from __future__ import annotations

from collections.abc import Sequence

from workcompanion.agents.base import AgentBase
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.common import GroundingLabel
from workcompanion.schemas.tutor import TutorTurn
from workcompanion.utils.coerce import coerce_int
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Socratic Tutor inside WorkCompanion AI.

You guide learners to answers through questions. You must NOT give the answer \
immediately, no matter how the learner asks.

METHOD
1. Start from what the learner appears to know. Ask ONE focused, open question.
2. After each reply, evaluate: correct, partially correct, or a misconception.
3. If they are struggling, give a smaller, more concrete question - not the answer.
4. Give a HINT only when they are stuck. A hint narrows the search; it never \
closes it.
5. Escalate gradually: concrete -> relational -> abstract.
6. Only after the learner has genuinely worked through it (or explicitly asks \
twice, or says "just tell me") do you reveal the answer with a full explanation.

OUTPUT - return STRICT JSON:
{
  "message": "your turn: the question, plus brief feedback on their last answer",
  "hint": "a narrowing hint, or null",
  "reveal_answer": false,
  "scaffold_level": 1-5,
  "correctness": "correct | partial | incorrect | no_attempt_yet",
  "next_question": "the next single question to ask",
  "explanation": "the full answer - ONLY when reveal_answer is true",
  "misconception": "the specific misconception detected, or null"
}

Rules:
- Never write more than 120 words in "message".
- Never reveal the answer while reveal_answer is false.
- Be warm and curious, never condescending."""

REVEAL_SYSTEM = """You are the Socratic Tutor. The learner has asked for the answer \
directly after working through the dialogue, so reveal it now.

Return STRICT JSON:
{
  "message": "confirm what they had right, then give the clear explanation",
  "explanation": "the full correct explanation, preserving equations and units",
  "common_error": "the mistake most learners make here, or null",
  "follow_up": "one question to confirm the new understanding"
}"""


class SocraticAgent(AgentBase):
    """Question-led tutoring agent."""

    name = "socratic"
    description = "Guides the learner to answers with escalating questions and hints."

    MAX_SCAFFOLD = 5

    def __init__(
        self,
        rag_agent: RagAgent,
        llm=None,
        settings=None,
        cache=None,
    ):
        super().__init__(llm, settings, cache)
        self._rag = rag_agent

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def start(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        document_ids: list[str] | None = None,
        prior_scaffold: int = 0,
    ) -> TutorTurn:
        """Open the dialogue with a first guiding question."""
        retrieval = self._rag.retrieve(question, history=list(history or []), document_ids=document_ids)
        context = trim_to_tokens(retrieval.context, 4000) if retrieval.chunks else ""

        prompt = (
            f"TOPIC / QUESTION:\n{question}\n\n"
            + (f"REFERENCE MATERIAL (your context - do not quote it verbatim):\n{context}\n\n" if context else "")
            + "Open the Socratic dialogue. Ask the FIRST, easiest, most concrete question that "
            "moves the learner toward understanding. Do not answer."
        )
        return self._turn(
            prompt,
            fallback_question=self._first_probe(question),
            grounded=bool(context),
        )

    # ------------------------------------------------------------------
    def next_turn(
        self,
        question: str,
        learner_answer: str,
        *,
        scaffold_level: int = 0,
        history: Sequence[tuple[str, str]] | None = None,
        document_ids: list[str] | None = None,
        force_reveal: bool = False,
    ) -> TutorTurn:
        """Advance the dialogue given the learner's latest reply."""
        retrieval = self._rag.retrieve(question, history=list(history or []), document_ids=document_ids)
        context = trim_to_tokens(retrieval.context, 4000) if retrieval.chunks else ""
        level = min(scaffold_level + 1, self.MAX_SCAFFOLD)

        wants_answer = force_reveal or _asks_for_answer(learner_answer)
        system = REVEAL_SYSTEM if wants_answer else SYSTEM_PROMPT

        if wants_answer:
            prompt = (
                f"TOPIC:\n{question}\n\n"
                f"LEARNER'S ATTEMPT:\n{learner_answer[:1500]}\n\n"
                + (f"REFERENCE MATERIAL:\n{context}\n\n" if context else "")
                + "Reveal the answer now: confirm what they had right, then explain it clearly."
            )
        else:
            prompt = (
                f"TOPIC:\n{question}\n\n"
                f"LEARNER'S LATEST REPLY:\n{learner_answer[:1500]}\n\n"
                f"DIALOGUE SO FAR:\n{_render_dialogue(history)}\n\n"
                f"CURRENT SCAFFOLD LEVEL: {level} of {self.MAX_SCAFFOLD}\n\n"
                + (f"REFERENCE MATERIAL (your context):\n{context}\n\n" if context else "")
                + "Evaluate the reply, then ask the next single question. Never reveal the answer "
                "yet."
            )

        return self._turn(prompt, system=system, scaffold_level=level,
                          fallback_question=self._probe_for(question, level),
                          grounded=bool(context))

    # ------------------------------------------------------------------
    def _turn(
        self,
        prompt: str,
        *,
        system: str | None = None,
        scaffold_level: int = 1,
        fallback_question: str = "",
        grounded: bool = True,
    ) -> TutorTurn:
        request = self.build_request(
            "socratic_turn", system=system, user=prompt, temperature=0.5, max_tokens=800
        )
        payload = self.structured(request) or {}
        if not payload:
            logger.info("[socratic] falling back to a template turn")
            return TutorTurn(
                message=(
                    "Let's work it out together.\n\n"
                    f"{fallback_question}"
                    + (
                        "\n\n_Your documents did not contain a passage on this, so I'm guiding you "
                        "from first principles._"
                        if not grounded
                        else ""
                    )
                ),
                scaffold_level=scaffold_level,
                next_question=fallback_question or None,
            )

        message = str(payload.get("message") or fallback_question).strip()
        reveal = bool(payload.get("reveal_answer"))
        explanation = str(payload.get("explanation") or "")
        if explanation and explanation.lower() not in message.lower():
            message = f"{message}\n\n**The answer:** {explanation}"
            reveal = True

        return TutorTurn(
            message=message,
            hint=str(payload.get("hint") or "") or None,
            reveal_answer=reveal,
            scaffold_level=coerce_int(
                payload.get("scaffold_level"), scaffold_level, minimum=0, maximum=3
            ),
            checks_understanding=not reveal,
            next_question=str(payload.get("next_question") or "") or None,
            misconceptions_detected=(
                [str(payload["misconception"])] if payload.get("misconception") else []
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _first_probe(question: str) -> str:
        topic = question.strip().rstrip("?")
        return (
            f"Before we define anything: what do you already know about {topic} - even if it is "
            "just an intuition? Let's start from what you have."
        )

    @staticmethod
    def _probe_for(question: str, level: int) -> str:
        probes = [
            f"What is the simplest case of {question.strip().rstrip('?')} you can think of?",
            "Which quantity in that case changes, and by how much?",
            "Why would that change happen physically?",
            "What would happen if you doubled that quantity?",
            "Can you now state the general principle in one sentence?",
        ]
        return probes[min(max(level, 1), len(probes)) - 1]

    # ------------------------------------------------------------------
    def grounding_labels(self) -> list[GroundingLabel]:
        return [GroundingLabel.RETRIEVED_FACT, GroundingLabel.MODEL_REASONING]


def _asks_for_answer(text: str) -> bool:
    lowered = (text or "").lower()
    triggers = (
        "just tell me", "tell me the answer", "give me the answer", "what is the answer",
        "i give up", "no idea", "i don't know", "stop hinting", "answer now",
        "just answer", "reveal", "ok give me",
    )
    return any(trigger in lowered for trigger in triggers)


def _render_dialogue(history: Sequence[tuple[str, str]] | None, limit: int = 6) -> str:
    if not history:
        return "(this is the start of the dialogue)"
    return "\n".join(
        f"{'Learner' if role == 'user' else 'Tutor'}: {content.strip()[:320]}"
        for role, content in list(history)[-limit:]
    )