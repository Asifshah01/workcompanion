"""Tutor Agent.

Teaching, not answer-dumping.  The agent adapts level and style to the learner,
proactively targets recorded misconceptions, and can close the loop by asking a
understanding-check question.

Teaching philosophy implemented in the prompt:

    Explain -> Ask -> Check understanding -> Correct misconception -> Reinforce
"""

from __future__ import annotations

from collections.abc import Sequence

from workcompanion.agents.base import AgentBase
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.memory.learner_profile import LearnerProfileSnapshot
from workcompanion.rag.citations import append_sources, verify_grounding
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.tutor import TutorRequest, TutorTurn
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

BASE_SYSTEM = """You are the Tutor Agent inside WorkCompanion AI - a patient, \
expert study coach for university students, researchers and self-learners.

TEACHING PHILOSOPHY
1. Start from what the learner already knows and build up in small, clear steps.
2. Explain, then ASK a question to check understanding, then reinforce.
3. Prefer one strong, memorable analogy or worked example over a wall of text.
4. Actively probe for misconceptions. If the learner has a known confusion, \
address it explicitly and contrast the two ideas directly.
5. Never simply read the retrieved text back. Synthesise, structure and clarify.
6. If the learner's material disagrees with your explanation, say so.

ACADEMIC RIGOUR
- Preserve equations, symbols and units exactly; use Markdown and LaTeX.
- Define every technical term the first time you use it.
- State assumptions and limits of validity.
- Say "I am not sure" rather than guessing.

GROUNDING
- Use the CONTEXT block as your primary source. Cite markers like [1] for claims \
taken from it.
- Anything from your own background knowledge must be labelled \
"General knowledge (not from your documents)".
- Never invent a citation marker, page number or quotation."""

LEVEL_GUIDANCE = {
    "beginner": (
        "Learner level: BEGINNER (first year / no background assumed).\n"
        "- Use everyday analogies before technical vocabulary.\n"
        "- Define every term. Avoid jargon entirely.\n"
        "- Use short sentences and concrete numbers.\n"
        "- Anticipate confusion and pre-empt the two most likely mistakes."
    ),
    "undergraduate": (
        "Learner level: UNDERGRADUATE (has seen the topic in lectures/textbooks).\n"
        "- Use standard terminology freely, but define anything non-obvious.\n"
        "- Include the governing equations and the intuition behind them.\n"
        "- Connect to standard course content and typical exam expectations."
    ),
    "graduate": (
        "Learner level: GRADUATE (specialising in this area).\n"
        "- Be precise and quantitative. State assumptions and validity conditions.\n"
        "- Discuss limitations, edge cases and alternative formulations.\n"
        "- Point to the underlying theory and active research questions."
    ),
    "expert": (
        "Learner level: EXPERT / RESEARCHER.\n"
        "- Skip textbook exposition; engage with subtleties and open problems.\n"
        "- Discuss conventions, nomenclature, measurement issues and disputes.\n"
        "- Cite specific theories, papers or results where you know them, and clearly "
        "mark them as general knowledge rather than the learner's material."
    ),
}

STYLE_GUIDANCE = {
    "concise": "Be brief: at most 8 short lines. No preamble.",
    "detailed": (
        "Be thorough: a definition, the mechanism, a worked example, the key "
        "assumptions, then a check question."
    ),
    "socratic": (
        "Teach Socratically: answer in stages, asking a guiding question after each "
        "stage and waiting for the learner's reply. Never dump the full answer at once."
    ),
    "example_based": (
        "Anchor the explanation in at least two concrete worked examples with numbers "
        "where applicable."
    ),
    "mathematical": (
        "Show the mathematics explicitly: definitions, assumptions, derivations and "
        "units. Use LaTeX."
    ),
    "analogy": (
        "Explain with one vivid, accurate analogy and then state precisely where the "
        "analogy breaks down."
    ),
}


class TutorAgent(AgentBase):
    """Adaptive teaching agent."""

    name = "tutor"
    description = "Adaptive teaching across four levels with misconception targeting."

    def __init__(
        self,
        rag_agent: RagAgent,
        profile: LearnerProfileSnapshot | None = None,
        llm=None,
        settings=None,
        cache=None,
    ):
        super().__init__(llm, settings, cache)
        self._rag = rag_agent
        self._profile = profile or LearnerProfileSnapshot()

    # ------------------------------------------------------------------
    def set_profile(self, profile: LearnerProfileSnapshot) -> None:
        """Update the learner context used to personalise explanations."""
        self._profile = profile

    @property
    def system_prompt(self) -> str:
        return BASE_SYSTEM

    # ------------------------------------------------------------------
    def teach(
        self,
        request: TutorRequest,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
    ) -> AgentResult:
        """Teach a concept at the requested level and style."""
        level = request.level or self._profile.preferred_level
        style = request.style or self._profile.preferred_style

        retrieval = self._rag.retrieve(
            request.question,
            history=list(history or []),
            document_ids=document_ids,
            document_filter=document_filter,
        )

        profile_block = self._profile_block(request)
        context_block = (
            f"CONTEXT (from the learner's own material):\n{trim_to_tokens(retrieval.context, 8000)}"
            if retrieval.chunks
            else "CONTEXT:\n(nothing relevant was found in the learner's documents)"
        )

        from workcompanion.rag.query_transform import QueryTransformer

        prompt = (
            f"{QueryTransformer.conversation_prompt(request.question, list(history or []), limit=4)}\n\n"
            f"LEARNER PROFILE:\n{profile_block}\n\n"
            f"{LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE['undergraduate'])}\n"
            f"{STYLE_GUIDANCE.get(style, STYLE_GUIDANCE['detailed'])}\n\n"
            f"{context_block}\n\n"
            f"LEARNER WANTS TO UNDERSTAND:\n{request.question}\n\n"
            f"REQUIRED STRUCTURE:\n"
            f"1. **Hook** - one sentence that makes the idea matter.\n"
            f"2. **Core explanation** - build it up step by step at the stated level.\n"
            f"3. **Worked example or analogy** - make it concrete.\n"
            f"4. **Watch out** - the most common mistake here.\n"
            f"{'5. **Check your understanding** - end with ONE specific question and wait.' if request.follow_up_check else ''}\n\n"
            f"CITE with [n] markers when you use the CONTEXT. Label outside knowledge as "
            f"'General knowledge (not from your documents)'."
        )

        completion = self.build_request(
            "tutor_teach", user=prompt, temperature=0.4, max_tokens=2200
        )
        try:
            response = self.complete(completion)
            answer_text = response.text.strip()
        except Exception as exc:
            logger.warning("[tutor] LLM failed: %s", exc)
            return self._rag.answer(
                request.question,
                history=list(history or []),
                document_ids=document_ids,
                document_filter=document_filter,
                style=style,
            )

        citations, fabricated = verify_grounding(answer_text, retrieval.citations)
        warnings: list[str] = []
        if fabricated:
            warnings.append(
                "Some citation markers the model produced were not supported by the retrieved "
                "material and have been removed."
            )
        if not retrieval.chunks:
            warnings.append(
                "No matching passage was found in your documents, so this explanation comes "
                "from general knowledge."
            )
            grounding = [GroundingLabel.GENERAL_KNOWLEDGE]
            confidence = 0.35
        else:
            grounding = [GroundingLabel.RETRIEVED_FACT, GroundingLabel.MODEL_REASONING]
            confidence = max(retrieval.confidence * 0.95, 0.45)

        return self.result(
            append_sources(answer_text, citations),
            intent="teach",
            sources=citations,
            confidence=confidence,
            grounding=grounding,
            latency_ms=response.latency_ms + retrieval.stats.total_latency_ms,
            warnings=warnings,
            response=response,
            metadata={
                "level": level,
                "style": style,
                "evidence": "documents" if retrieval.chunks else "general_knowledge",
                "retrieval": retrieval.stats.model_dump(),
                "targeted_misconceptions": request.misconceptions,
            },
        )

    # ------------------------------------------------------------------
    def stream_teach(
        self,
        request: TutorRequest,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        document_ids: list[str] | None = None,
    ) -> tuple[AgentResult, object]:
        """Return a placeholder result plus an iterator of text chunks.

        Streaming keeps the UI responsive; citations are attached by the caller
        after the stream finishes via :meth:`finalise_stream`.
        """
        retrieval = self._rag.retrieve(
            request.question, history=list(history or []), document_ids=document_ids
        )
        prompt = (
            f"LEARNER PROFILE:\n{self._profile_block(request)}\n\n"
            f"{LEVEL_GUIDANCE.get(request.level, LEVEL_GUIDANCE['undergraduate'])}\n"
            f"{STYLE_GUIDANCE.get(request.style, STYLE_GUIDANCE['detailed'])}\n\n"
            f"CONTEXT:\n{trim_to_tokens(retrieval.context, 8000)}\n\n"
            f"LEARNER WANTS TO UNDERSTAND:\n{request.question}\n\n"
            "Teach it now. Use [n] markers for the CONTEXT. End with one check question."
        )
        request_obj = self.build_request("tutor_teach", user=prompt, temperature=0.4, max_tokens=2200)
        result = self.result(
            "",
            intent="teach",
            sources=retrieval.citations,
            confidence=retrieval.confidence,
            metadata={"streaming": True, "retrieval": retrieval.stats.model_dump()},
        )
        return result, self.stream(request_obj)

    def finalise_stream(self, result: AgentResult, text: str) -> AgentResult:
        """Attach citations/confidence to a streamed answer."""
        citations, _fabricated = verify_grounding(text, result.sources)
        result.answer = append_sources(text.strip(), citations)
        result.sources = citations
        result.confidence = max(result.confidence, 0.3)
        result.confidence_level = result.with_level().confidence_level
        result.grounding = [GroundingLabel.RETRIEVED_FACT, GroundingLabel.MODEL_REASONING]
        return result

    # ------------------------------------------------------------------
    def check_understanding(
        self, concept: str, learner_answer: str, *, expected: str = ""
    ) -> TutorTurn:
        """Evaluate a learner's explanation and decide the next teaching move."""
        prompt = (
            f"CONCEPT:\n{concept}\n\n"
            + (f"EXPECTED KEY POINTS:\n{expected}\n\n" if expected else "")
            + f"LEARNER'S ANSWER:\n{learner_answer}\n\n"
            "Assess this answer as a tutor:\n"
            "1. Is it correct? Judge substance, not wording.\n"
            "2. What is missing or wrong?\n"
            "3. Give ONE specific hint that moves them forward without giving the answer away.\n"
            "4. Write the next guiding question.\n\n"
            "Return STRICT JSON:\n"
            '{"is_correct": bool, "missing": ["..."], "misconception": "..." | null, '
            '"hint": "...", "next_question": "...", "praise": "..." | null}'
        )
        request = self.build_request("tutor_check", user=prompt, temperature=0.2, max_tokens=800)
        payload = self.structured(request) or {}
        is_correct = bool(payload.get("is_correct"))
        missing = [str(item) for item in (payload.get("missing") or [])]

        message_parts: list[str] = []
        praise = payload.get("praise")
        if is_correct:
            message_parts.append(str(praise or "Correct - well reasoned."))
            if missing:
                message_parts.append("A couple of refinements:")
                message_parts.extend(f"- {item}" for item in missing)
        else:
            message_parts.append("Not quite - let's tighten it up.")
            if missing:
                message_parts.append("What to revisit:")
                message_parts.extend(f"- {item}" for item in missing)

        return TutorTurn(
            message="\n".join(message_parts),
            hint=str(payload.get("hint") or "") or None,
            reveal_answer=False,
            scaffold_level=1 if is_correct else 2,
            checks_understanding=True,
            next_question=str(payload.get("next_question") or "") or None,
            misconceptions_detected=[str(payload["misconception"])] if payload.get("misconception") else [],
        )

    # ------------------------------------------------------------------
    def diagnose_misconception(
        self, question: str, learner_answer: str, *, expected: str
    ) -> TutorTurn:
        """Explain *why* an answer is wrong, targeting the specific error."""
        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"EXPECTED ANSWER:\n{expected}\n\n"
            f"LEARNER'S ANSWER:\n{learner_answer}\n\n"
            "The learner is wrong. Write a tutor's correction:\n"
            "1. Name the specific misconception (not 'you were wrong').\n"
            "2. Explain the correct reasoning in 2-4 sentences.\n"
            "3. Give a contrasting example that makes the distinction obvious.\n"
            "4. Ask one question that would reveal whether they now understand.\n"
            "Be supportive and specific. Do not repeat the question back.\n\n"
            'Return STRICT JSON: {"misconception": "...", "correction": "...", '
            '"contrast_example": "...", "check_question": "..."}'
        )
        request = self.build_request("tutor_diagnose", user=prompt, temperature=0.3, max_tokens=900)
        payload = self.structured(request) or {}

        parts = [
            f"**The misconception:** {payload.get('misconception', 'The answer does not match the material.')}\n",
            f"**Why it's wrong:** {payload.get('correction', expected)}",
        ]
        if payload.get("contrast_example"):
            parts.append(f"**Contrast:** {payload['contrast_example']}")
        if payload.get("check_question"):
            parts.append(f"**Check:** {payload['check_question']}")

        return TutorTurn(
            message="\n\n".join(parts),
            hint=None,
            reveal_answer=True,
            scaffold_level=3,
            checks_understanding=True,
            next_question=str(payload.get("check_question") or "") or None,
            misconceptions_detected=[str(payload["misconception"])] if payload.get("misconception") else [],
        )

    # ------------------------------------------------------------------
    def _profile_block(self, request: TutorRequest) -> str:
        """Blend the stored learner profile with this request's specifics."""
        lines = [self._profile.prompt_context() or "No prior learning history yet."]
        if request.misconceptions:
            lines.append(f"Address these known confusions: {'; '.join(request.misconceptions[:4])}")
        if request.subject:
            lines.append(f"Subject: {request.subject}")
        if request.history_summary:
            lines.append(f"Session summary: {request.history_summary}")
        return "\n".join(line for line in lines if line)