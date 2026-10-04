"""Knowledge (RAG) Agent.

Grounded question answering over the learner's own material.  Responsibilities:

* retrieve evidence through the advanced RAG pipeline,
* compose an answer that cites verifiable markers,
* refuse to answer when evidence is insufficient,
* distinguish retrieved fact / model reasoning / general knowledge.
"""

from __future__ import annotations

from collections.abc import Sequence

from workcompanion.agents.base import AgentBase
from workcompanion.config.logging_config import get_logger
from workcompanion.rag.citations import (
    annotate_grounding,
    append_sources,
    insufficient_evidence_result,
    verify_grounding,
)
from workcompanion.rag.pipeline import RAGPipeline
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.retrieval import RetrievalOutcome
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Knowledge Agent inside WorkCompanion AI, a study \
platform. You answer questions using ONLY the evidence supplied in the CONTEXT block.

ABSOLUTE RULES
1. Every factual claim must be supported by the CONTEXT. Cite the passage marker in \
square brackets, e.g. [1]. Place the marker immediately after the sentence it supports.
2. Never invent a citation marker, page number, document name or quotation. Only use \
markers that appear in the CONTEXT.
3. If the CONTEXT does not answer the question, say so plainly. Do not fill the gap \
from memory. You may then offer what general knowledge you have, clearly labelled \
"General knowledge (not from your documents)".
4. If sources disagree, report the disagreement instead of choosing silently.
5. Preserve equations, units and symbols exactly. Use Markdown and LaTeX ($...$ inline, \
$$...$$ displayed).
6. Be concise and academically precise. No filler, no praise, no restating the question.

STYLE
- Lead with the direct answer in the first sentence.
- Use short paragraphs or bullets.
- Be explicit about the limits of the evidence."""


class RagAgent(AgentBase):
    """Retrieval-grounded answering agent."""

    name = "knowledge"
    description = "Answers questions strictly from the learner's indexed material, with citations."

    def __init__(self, pipeline: RAGPipeline, llm=None, settings=None, cache=None):
        super().__init__(llm, settings, cache)
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    @property
    def store(self) -> object:
        """The underlying vector store (used by whole-document agents)."""
        return self._pipeline.store

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    # ------------------------------------------------------------------
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        include_context: bool = True,
    ) -> RetrievalOutcome:
        """Expose the retrieval stage (the router and UI use this directly)."""
        return self._pipeline.retrieve(
            question,
            history=list(history or []),
            top_k=top_k,
            document_ids=document_ids,
            document_filter=document_filter,
        )

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        style: str | None = None,
        allow_general_knowledge: bool = True,
        require_grounding: bool = True,
    ) -> AgentResult:
        """Answer a question from the learner's documents."""
        intent = "answer"
        if self._pipeline.is_empty:
            return insufficient_evidence_result(
                self.name,
                intent=intent,
                answer=(
                    "**No documents are indexed yet.**\n\n"
                    "Upload a PDF, DOCX, PPTX, TXT or Markdown file in **My Knowledge**, "
                    "or load the demo dataset from the Dashboard, then ask again."
                ),
            )

        outcome = self._pipeline.retrieve(
            question,
            history=list(history or []),
            top_k=top_k,
            document_ids=document_ids,
            document_filter=document_filter,
        )

        if not outcome.grounded_context_available or not outcome.chunks:
            return insufficient_evidence_result(self.name, intent=intent)

        prompt = self._build_prompt(
            question,
            outcome,
            history=list(history or []),
            style=style,
            allow_general_knowledge=allow_general_knowledge,
        )
        request = self.build_request("grounded_answer", user=prompt, temperature=0.25, max_tokens=1800)

        try:
            response = self.complete(request)
        except Exception as exc:
            logger.warning("[knowledge] LLM failed, falling back to extractive: %s", exc)
            return self._extractive_fallback(question, outcome, intent=intent)

        answer_text = response.text.strip()
        if not answer_text:
            return self._extractive_fallback(question, outcome, intent=intent)

        citations, fabricated = verify_grounding(
            answer_text, outcome.citations, strict=self._settings.citation_grounding_check
        )
        warnings: list[str] = []
        if fabricated:
            warnings.append(
                "The model referenced source markers that were not in the retrieved context; "
                "they have been removed."
            )
        if not citations and self._settings.citation_grounding_check:
            citations = outcome.citations[:3]

        outcome = self._pipeline.refresh_confidence(outcome, answer_text)
        grounding = annotate_grounding(answer_text, outcome.chunks, outcome.citations)
        if not outcome.chunks and allow_general_knowledge:
            grounding = [GroundingLabel.GENERAL_KNOWLEDGE]

        final_answer = append_sources(answer_text, citations)
        self.trace(
            f"answered with {len(outcome.chunks)} chunks",
            confidence=outcome.confidence,
            level=outcome.confidence_level.value,
        )

        return self.result(
            final_answer,
            intent=intent,
            sources=citations,
            confidence=outcome.confidence,
            grounding=grounding,
            latency_ms=outcome.stats.total_latency_ms + response.latency_ms,
            warnings=warnings,
            response=response,
            metadata={
                "retrieval": outcome.stats.model_dump(),
                "evidence": "documents",
                "documents": outcome.by_document(),
                "queries_used": outcome.query.all_queries(),
            },
        )

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        outcome: RetrievalOutcome,
        *,
        history: Sequence[tuple[str, str]],
        style: str | None,
        allow_general_knowledge: bool,
    ) -> str:
        from workcompanion.rag.query_transform import QueryTransformer

        conversation = QueryTransformer.conversation_prompt(question, list(history), limit=4)
        style_line = {
            "concise": "Answer in at most 6 short lines.",
            "detailed": "Give a structured, thorough answer with definitions and key points.",
            "socratic": "Answer, then pose one probing question to check understanding.",
            "example_based": "Include at least one worked example.",
            "mathematical": "Show the governing relations explicitly with symbols and units.",
            "analogy": "Include one clear analogy or mental model.",
        }.get(style or "", "Be clear and structured.")

        knowledge_rule = (
            "You MAY add clearly-labelled general knowledge after answering, but you must "
            "label it 'General knowledge (not from your documents)'."
            if allow_general_knowledge
            else "Do not add anything that is not in the CONTEXT."
        )

        budget_note = (
            f"\nRetrieved {len(outcome.chunks)} passages. Cite the strongest ones; you do not "
            "need to cite every passage."
        )
        return (
            f"{conversation}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{outcome.context}\n"
            f"{budget_note}\n\n"
            f"INSTRUCTIONS:\n"
            f"- {style_line}\n"
            f"- Cite with the bracket markers shown above, e.g. [1].\n"
            f"- {knowledge_rule}\n"
            f"- If the CONTEXT does not contain the answer, reply exactly:\n"
            f"  NOT_FOUND_IN_CONTEXT: <one sentence on what is missing>"
        )

    # ------------------------------------------------------------------
    def _extractive_fallback(
        self, question: str, outcome: RetrievalOutcome, *, intent: str = "answer"
    ) -> AgentResult:
        """Quote the best passages verbatim - always safe, never hallucinated."""
        passages: list[str] = []
        for citation, chunk in zip(outcome.citations, outcome.chunks, strict=False):
            passages.append(f"- {citation.label()}\n  > {chunk.chunk.preview(400)}")
        body = (
            "**Relevant passages from your material**\n\n"
            + "\n".join(passages)
            + "\n\n_The language model was unavailable, so these are the exact retrieved "
            "passages rather than a generated summary._"
        )
        return self.result(
            append_sources(body, outcome.citations),
            intent=intent,
            sources=outcome.citations,
            confidence=min(outcome.confidence, 0.5),
            grounding=[GroundingLabel.RETRIEVED_FACT],
            latency_ms=outcome.stats.total_latency_ms,
            warnings=["Generated answer unavailable - showing verbatim passages."],
            metadata={"evidence": "documents", "mode": "extractive"},
        )

    # ------------------------------------------------------------------
    def summarize(
        self,
        outcome: RetrievalOutcome,
        *,
        focus: str | None = None,
        style: str = "concise",
    ) -> AgentResult:
        """Summarise a retrieved body of text (document summary / section recap)."""
        if not outcome.chunks:
            return insufficient_evidence_result(self.name, intent="summarize")

        target = focus or "the supplied material"
        prompt = (
            f"Summarise {target} using ONLY the CONTEXT below.\n\n"
            f"CONTEXT:\n{trim_to_tokens(outcome.context, 9000)}\n\n"
            "Write a clear, well-structured summary. Preserve equations, definitions and "
            "important numbers. Cite markers like [1]. Do not add outside information."
        )
        request = self.build_request("summarize_document", user=prompt, temperature=0.2, max_tokens=1600)
        try:
            response = self.complete(request)
        except Exception as exc:
            logger.warning("[knowledge] summarisation failed: %s", exc)
            return self._extractive_fallback(target, outcome, intent="summarize")

        citations, _fabricated = verify_grounding(response.text, outcome.citations)
        outcome = self._pipeline.refresh_confidence(outcome, response.text)
        return self.result(
            append_sources(response.text.strip(), citations or outcome.citations[:3]),
            intent="summarize",
            sources=citations or outcome.citations[:3],
            confidence=outcome.confidence,
            grounding=annotate_grounding(response.text, outcome.chunks, outcome.citations),
            latency_ms=response.latency_ms,
            response=response,
            metadata={"evidence": "documents"},
        )

    # ------------------------------------------------------------------
    def compare(
        self,
        question: str,
        documents: Sequence[str],
        *,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> AgentResult:
        """Compare two or more documents / theories and surface contradictions."""
        per_document: dict[str, RetrievalOutcome] = {}
        blocks: list[str] = []
        citations = []

        for index, document_name in enumerate(documents, start=1):
            probe = f"{question} in {document_name}"
            outcome = self._pipeline.retrieve(
                probe,
                history=list(history or []),
                top_k=5,
                document_filter=document_name,
            )
            per_document[document_name] = outcome
            if outcome.chunks:
                blocks.append(f"=== SOURCE {index}: {document_name} ===\n{outcome.context}")
                citations.extend(outcome.citations)

        if not blocks:
            return insufficient_evidence_result(self.name, intent="compare")

        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"{chr(10).join(blocks)}\n\n"
            "INSTRUCTIONS:\n"
            "1. Summarise what each source says about the question.\n"
            "2. List the agreements between the sources.\n"
            "3. List any CONTRADICTIONS or inconsistencies explicitly, quoting both sides.\n"
            "4. State which claims are supported by which source, using the [n] markers.\n"
            "5. Do not invent disagreements that the text does not contain."
        )
        request = self.build_request("compare_sources", user=prompt, temperature=0.2, max_tokens=2000)
        try:
            response = self.complete(request)
        except Exception as exc:
            logger.warning("[knowledge] comparison failed: %s", exc)
            return self._extractive_fallback(question, per_document[documents[0]], intent="compare")

        used, fabricated = verify_grounding(response.text, citations)
        if fabricated:
            logger.info("Dropped %d ungrounded citation markers from the comparison.", len(fabricated))

        return self.result(
            append_sources(response.text.strip(), used or citations[:5]),
            intent="compare",
            sources=used or citations[:5],
            confidence=min(item.confidence for item in per_document.values()),
            grounding=annotate_grounding(response.text, list(per_document.values())[0].chunks, citations),
            latency_ms=response.latency_ms,
            response=response,
            metadata={
                "evidence": "documents",
                "compared": list(per_document),
                "conflicts_detected": len(per_document) > 1,
            },
        )