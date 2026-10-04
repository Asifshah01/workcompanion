"""Research Agent.

Two jobs, kept deliberately separate in the output so a reader can never
confuse a claim from the learner's own library with a claim from the open web:

* :meth:`ResearchAgent.analyze_paper` - structured analysis of one indexed
  paper (problem, method, data, results, limitations, gaps) plus a plain-language
  explanation at a chosen level.
* :meth:`ResearchAgent.research` - a synthesized answer to a research question
  from documents, web results, or both, with conflicting points surfaced instead
  of being smoothed over.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.rag.citations import (
    append_sources,
    build_citations,
    citations_markdown,
    verify_grounding,
)
from workcompanion.rag.compressor import ContextCompressor
from workcompanion.schemas.common import AgentResult, Citation, GroundingLabel, SourceKind
from workcompanion.schemas.research import PaperAnalysis, ResearchBrief, WebSource
from workcompanion.schemas.retrieval import RetrievalOutcome
from workcompanion.tools.web_search import WebSearchTool
from workcompanion.utils.text_utils import extract_key_terms
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

PAPER_SYSTEM_PROMPT = """You are the Research Agent inside WorkCompanion AI. You analyse \
academic papers so a student can understand them without reading all 12 pages.

ABSOLUTE RULES
1. Report ONLY what the TEXT shows. If a field is not stated (dataset, venue, \
sample size, metrics), write "Not stated in the paper" - never guess.
2. Keep the authors' own claims separate from your assessment. Limitations and \
research gaps are your critical reading; label them as such.
3. Cite the markers of the passages supporting each claim.
4. Preserve numbers, percentages and metric names exactly. Direction and \
magnitude matter more than adjectives.
5. Do not evaluate the paper's quality or recommend acceptance.

OUTPUT
- Be specific and skimmable: bullets over prose.
- `key_results` should be findings with numbers, not adjectives."""

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent inside WorkCompanion AI. You answer \
research questions and you are scrupulous about where each claim comes from.

ABSOLUTE RULES
1. DOCUMENT EVIDENCE and WEB EVIDENCE must never be blended.
   - Claims from the CONTEXT block are cited with the [n] markers they carry.
   - Claims from WEB RESULTS are cited with their [Wn] markers.
2. Never invent a source, an author, a year, a statistic or a DOI.
3. If the evidence disagrees, say so explicitly and give both sides.
4. If the evidence is insufficient, say what is missing instead of filling the gap.
5. Clearly separate what the sources claim from what you conclude.

STYLE
- Lead with the direct answer.
- Then evidence, then implications.
- Use Markdown bullets."""

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_LEVEL_INSTRUCTIONS = {
    "beginner": (
        "Explain as if to a first-year student: no jargon, everyday analogies, "
        "spell out every acronym."
    ),
    "undergraduate": (
        "Explain at final-year undergraduate level: use the field's standard "
        "terminology and show the key equation or procedure."
    ),
    "graduate": (
        "Explain at graduate level: focus on the technical contribution, the "
        "assumptions behind the method and the validity conditions."
    ),
    "expert": (
        "Explain at expert level: engage with the specific methodological choices, "
        "the novelty claim and how it relates to prior work."
    ),
}


class ResearchAgent(AgentBase):
    """Paper analysis and multi-source research synthesis."""

    name = "research"
    description = "Analyses papers and synthesises research from documents and the web."

    def __init__(
        self,
        rag_agent: RagAgent,
        web_search: WebSearchTool | None = None,
        llm=None,
        settings=None,
        cache=None,
    ):
        super().__init__(llm, settings, cache)
        self._rag = rag_agent
        self._web = web_search or WebSearchTool(settings)

    @property
    def system_prompt(self) -> str:
        return RESEARCH_SYSTEM_PROMPT

    @property
    def web_tool(self) -> WebSearchTool:
        """The web-search tool (exposed for the CrewAI tool wrappers)."""
        return self._web

    # ==================================================================
    # Paper analysis
    # ==================================================================
    def analyze_paper(
        self,
        document_id: str,
        *,
        document_name: str | None = None,
        level: str = "undergraduate",
        max_chunks: int = 24,
    ) -> PaperAnalysis:
        """Structured analysis of one indexed document.

        The whole document is sampled (front-loaded: title/abstract first, then a
        spread across the body) rather than query-retrieved, because a paper's
        contributions are not answerable by a single similarity search.
        """
        store = self._rag.store
        chunks = store.all_chunks(document_id) if hasattr(store, "all_chunks") else []
        if not chunks:
            chunks = self._rag.store.get_chunks([])  # keep the type checker happy
        if not chunks:
            return PaperAnalysis(
                document_id=document_id,
                document_name=document_name or "Unknown document",
                warnings=[
                    "This document is not in the index. Upload or re-index it before "
                    "requesting an analysis."
                ],
            )

        sample = self._sample_chunks(chunks, limit=max_chunks)
        context = self._render_document(sample)
        citations = build_citations(sample, limit=max_chunks)

        prompt = self._paper_prompt(context, level=level)
        request = self.build_request("paper_analysis", user=prompt, temperature=0.15, max_tokens=2600)

        payload: dict[str, Any] = {}
        try:
            payload = self.structured(request) or {}
        except Exception as exc:
            logger.warning("[research] paper analysis failed: %s", exc)

        if not payload:
            payload = {}

        name = document_name or chunks[0].metadata.document_name
        analysis = PaperAnalysis(
            document_id=document_id,
            document_name=name,
            title=str(payload.get("title") or chunks[0].metadata.section or name),
            authors=_as_list(payload.get("authors")),
            affiliation=_as_optional_str(payload.get("affiliation")),
            year=_as_year(payload.get("year")) or _guess_year(sample),
            venue=_as_optional_str(payload.get("venue")),
            abstract=str(payload.get("abstract") or "").strip(),
            keywords=_as_list(payload.get("keywords")),
            research_problem=str(payload.get("research_problem") or "").strip(),
            objectives=_as_list(payload.get("objectives")),
            research_questions=_as_list(payload.get("research_questions")),
            methodology=str(payload.get("methodology") or "").strip(),
            method_details=_as_list(payload.get("method_details")),
            dataset=str(payload.get("dataset") or "").strip(),
            experimental_setup=str(payload.get("experimental_setup") or "").strip(),
            models_or_algorithms=_as_list(payload.get("models_or_algorithms")),
            evaluation_metrics=_as_list(payload.get("evaluation_metrics")),
            key_results=_as_list(payload.get("key_results")),
            findings_summary=str(payload.get("findings_summary") or "").strip(),
            limitations=_as_list(payload.get("limitations")),
            future_work=_as_list(payload.get("future_work")),
            novelty=str(payload.get("novelty") or "").strip(),
            research_gaps=_as_list(payload.get("research_gaps")),
            contribution_summary=str(payload.get("contribution_summary") or "").strip(),
            confidence=_clamp(payload.get("confidence"), 0.0, 1.0),
            warnings=[],
        ).recompute_completeness()

        if not payload:
            analysis.warnings.append(
                "The analysis could not be generated by a language model, so only the "
                "document's own opening text is reported."
            )
            analysis.confidence = 0.15
        elif analysis.completeness < 0.5:
            analysis.warnings.append(
                f"This analysis only filled {analysis.completeness:.0%} of the expected "
                "fields - the paper may be a short note, or the text extraction incomplete."
            )

        analysis.explained = self._explain(analysis, citations, level=level)
        analysis.citations = citations[:8]
        return analysis

    def render_analysis(self, analysis: PaperAnalysis, *, level: str = "undergraduate") -> str:
        """Markdown rendering shown in the Research page."""
        lines = [f"### {analysis.title or analysis.document_name}", ""]
        header: list[str] = []
        if analysis.authors:
            header.append(", ".join(analysis.authors[:6]))
        if analysis.year:
            header.append(str(analysis.year))
        if analysis.venue:
            header.append(analysis.venue)
        if header:
            lines.append(" · ".join(header))
            lines.append("")

        def section(title: str, body: str) -> None:
            if body and body.strip():
                lines.extend([f"**{title}**", "", body.strip(), ""])

        def bullets(items: Sequence[str]) -> str:
            return "\n".join(f"- {item}" for item in items if item)

        section("Research problem", analysis.research_problem)
        section("Objectives", bullets(analysis.objectives))
        if analysis.research_questions:
            section("Research questions", bullets(analysis.research_questions))
        section("Methodology", analysis.methodology)
        if analysis.method_details:
            section("Method details", bullets(analysis.method_details))
        section("Dataset", analysis.dataset)
        section("Experimental setup", analysis.experimental_setup)
        if analysis.evaluation_metrics:
            section("Metrics", bullets(analysis.evaluation_metrics))
        section("Key results", bullets(analysis.key_results))
        section("Findings", analysis.findings_summary)
        section("Critical reading - limitations", bullets(analysis.limitations))
        section("Research gaps (my assessment)", bullets(analysis.research_gaps))
        section("Novelty", analysis.novelty)
        section("Future work", bullets(analysis.future_work))

        explanation = analysis.explained.get(level)
        if explanation:
            section(f"In plain terms ({level})", explanation)

        lines.append(
            f"_Analysis completeness: {analysis.completeness:.0%} · "
            f"confidence: {analysis.confidence:.0%}_"
        )
        for warning in analysis.warnings:
            lines.append(f"> {warning}")
        return "\n".join(lines).strip()

    # ==================================================================
    # Multi-source research
    # ==================================================================
    def research(
        self,
        question: str,
        *,
        use_web: bool = True,
        document_filter: str | None = None,
        document_ids: list[str] | None = None,
        history: Sequence[tuple[str, str]] | None = None,
        web_results: int | None = None,
    ) -> ResearchBrief:
        """Answer a research question from the document library and/or the web."""
        warnings: list[str] = []

        outcome = self._rag.retrieve(
            question,
            history=list(history or []),
            document_ids=document_ids,
            document_filter=document_filter,
            top_k=8,
        )

        web_sources: list[WebSource] = []
        if use_web and self._web.enabled:
            web_sources = self._web.search_many(
                self._web_queries(question, outcome),
                max_results=web_results,
            )
            if not web_sources and self._web.last_error:
                warnings.append(self._web.last_error)
        elif use_web:
            warnings.append("Web research is disabled in Settings, so only documents were used.")

        if (not outcome.chunks or outcome.confidence < self._settings.min_retrieval_confidence) and (
            not web_sources
        ):
            reason = (
                "Nothing in your indexed material matched closely enough to answer this."
                if outcome.chunks
                else "Nothing in your indexed material matched this question at all."
            )
            return ResearchBrief(
                question=question,
                answer=(
                    "**I found no usable evidence for this question.**\n\n"
                    f"{reason} Web research returned nothing, or is disabled.\n\n"
                    "- Upload material that discusses this topic, or\n"
                    "- Enable web research in Settings, or\n"
                    "- Rephrase the question using terms that appear in your documents."
                ),
                citations=[],
                confidence=0.0,
                used_web=False,
                warnings=warnings,
            )

        prompt, doc_citations, web_citations = self._research_prompt(
            question, outcome, web_sources, warnings
        )
        request = self.build_request(
            "research_synthesis", user=prompt, temperature=0.25, max_tokens=2600
        )

        answer_text = ""
        payload: dict[str, Any] = {}
        try:
            response = self.complete(request)
            answer_text = response.text.strip()
        except Exception as exc:
            logger.warning("[research] synthesis failed: %s", exc)

        if not answer_text and web_sources:
            # Degrade to a link list rather than losing the web evidence entirely.
            answer_text = "\n".join(
                f"- [{source.title}]({source.url}) - {source.snippet[:160]}"
                for source in web_sources[:6]
            )
            warnings.append("The model was unavailable, so web results are listed unranked.")

        doc_citations, fabricated = verify_grounding(answer_text, doc_citations, strict=False)
        web_citations, fabricated_web = verify_grounding(answer_text, web_citations, strict=False)
        if fabricated or fabricated_web:
            warnings.append(
                "Some source markers the model produced were not available and were removed."
            )

        brief = ResearchBrief(
            question=question,
            answer=append_sources(answer_text, [*doc_citations, *web_citations]),
            document_findings=[
                line
                for line in _extract_bullets(answer_text)
                if "**From your documents" in line or "DOCUMENT EVIDENCE" in line
            ],
            web_findings=[],
            reasoning=[],
            citations=[*doc_citations, *web_citations],
            conflicting_points=[],
            research_gaps=[],
            suggested_readings=web_sources,
            confidence=self._research_confidence(outcome, answer_text, web_sources),
            used_web=bool(web_citations or web_sources),
            warnings=warnings,
        )
        self.trace(
            f"researched {question[:60]}",
            documents=len(doc_citations),
            web=len(web_sources),
            confidence=brief.confidence,
        )
        return brief

    def run(
        self,
        question: str,
        *,
        use_web: bool = True,
        document_filter: str | None = None,
        document_ids: list[str] | None = None,
    ) -> AgentResult:
        """Envelope-returning entry point for chat / crew workflows."""
        brief = self.research(
            question,
            use_web=use_web,
            document_filter=document_filter,
            document_ids=document_ids,
        )
        grounding = [GroundingLabel.RETRIEVED_FACT]
        if any(c.source_kind is SourceKind.WEB for c in brief.citations):
            grounding.append(GroundingLabel.MODEL_REASONING)
        return self.result(
            brief.answer,
            intent="research",
            sources=brief.citations,
            confidence=brief.confidence,
            grounding=grounding,
            warnings=brief.warnings,
            metadata={
                "evidence": brief.evidence_kind,
                "used_web": brief.used_web,
                "document_sources": sum(
                    1 for c in brief.citations if c.source_kind is SourceKind.DOCUMENT
                ),
                "web_sources": sum(1 for c in brief.citations if c.source_kind is SourceKind.WEB),
                "offline": self.is_offline,
            },
        )

    # ==================================================================
    # Internals
    # ==================================================================
    def _explain(
        self, analysis: PaperAnalysis, citations: Sequence[Citation], *, level: str
    ) -> dict[str, str]:
        """Plain-language explanation at a chosen level."""
        prompt = (
            f"PAPER:\nTitle: {analysis.title}\n"
            f"Problem: {analysis.research_problem or 'not stated'}\n"
            f"Method: {analysis.methodology or 'not stated'}\n"
            f"Results: {'; '.join(analysis.key_results[:4]) or 'not stated'}\n\n"
            f"{_LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS['undergraduate'])}\n\n"
            "Explain what this paper did, why it mattered and what it found, in at most "
            "three short paragraphs. Use the CONTEXT markers where you rely on the text.\n\n"
            f"CONTEXT:\n{trim_to_tokens(citations_markdown(citations), 3000)}"
        )
        request = self.build_request("paper_explain", user=prompt, temperature=0.4, max_tokens=900)
        try:
            response = self.complete(request)
            return {level: response.text.strip()}
        except Exception as exc:
            logger.info("[research] plain-language explanation unavailable: %s", exc)
            return {}

    def _research_prompt(
        self,
        question: str,
        outcome: RetrievalOutcome,
        web_sources: Sequence[WebSource],
        warnings: list[str],
    ) -> tuple[str, list[Citation], list[Citation]]:
        """Build the synthesis prompt plus both citation sets."""
        doc_citations: list[Citation] = []
        sections: list[str] = []

        if outcome.chunks:
            doc_citations = outcome.citations
            sections.append(
                f"DOCUMENT EVIDENCE (your indexed material, cite as [1], [2], ...):\n"
                f"{trim_to_tokens(outcome.context, 7000)}"
            )

        web_citations: list[Citation] = []
        if web_sources:
            lines = []
            for index, source in enumerate(web_sources, start=1):
                marker = f"[W{index}]"
                web_citations.append(source.citation(marker))
                lines.append(f"{marker} {source.title}\n    URL: {source.url}\n    {source.snippet}")
            sections.append("WEB RESULTS (cite as [W1], [W2], ...):\n" + "\n".join(lines))
            warnings.append(
                "Web results are unverified search snippets, not peer-reviewed sources. "
                "Open the links before relying on them."
            )
        else:
            sections.append("WEB RESULTS: (none available)")

        prompt = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            + "\n\n".join(sections)
            + "\n\nREQUIRED STRUCTURE:\n"
            "1. **Answer** - the direct response, 2-4 sentences.\n"
            "2. **From your documents** - what the CONTEXT shows, with [n] markers.\n"
            "3. **From the web** - what the results indicate, with [Wn] markers.\n"
            "4. **Disagreements** - conflicts between sources, or 'None observed'.\n"
            "5. **What's still unknown** - the gaps a reader must close.\n\n"
            "If the CONTEXT does not address the question, say so in section 2 instead of "
            "filling the gap from memory."
        )
        return prompt, doc_citations, web_citations

    @staticmethod
    def _web_queries(question: str, outcome: RetrievalOutcome) -> list[str]:
        """Derive focused web queries (the question plus its key terms)."""
        queries = [question]
        terms = outcome.query.keywords or extract_key_terms(question, limit=6)
        if terms:
            queries.append(" ".join(terms[:4]))
        return queries[:2]

    @staticmethod
    def _research_confidence(
        outcome: RetrievalOutcome, answer: str, web_sources: Sequence[WebSource]
    ) -> float:
        """Confidence blends document retrieval quality with web corroboration."""
        base = outcome.confidence if outcome.chunks else 0.0
        if not web_sources:
            return round(base, 4)
        # Web snippets are weak evidence on their own: cap the boost.
        return round(min(0.75, base * 0.8 + 0.12), 4)

    # ------------------------------------------------------------------
    def _paper_prompt(self, context: str, *, level: str) -> str:
        return (
            "Analyse the following PAPER TEXT.\n\n"
            f"PAPER TEXT:\n{context}\n\n"
            "RULES:\n"
            "- Use only what this text states. For anything absent write "
            '"Not stated in the paper".\n'
            "- Attach [n] markers to the passages supporting each claim.\n"
            "- `limitations`, `research_gaps` and `novelty` are your critical reading: "
            "mark them as your assessment, not the authors' claims.\n"
            "- Keep every number, metric and comparison exactly as written.\n"
            f"- Audience level for the plain-language part: {level}.\n\n"
            'Return STRICT JSON:\n'
            '{"title": "...", "authors": ["..."], "affiliation": "..." | null, '
            '"year": 2024 | null, "venue": "..." | null, "abstract": "...", '
            '"keywords": ["..."], "research_problem": "...", "objectives": ["..."], '
            '"research_questions": ["..."], "methodology": "...", "method_details": ["..."], '
            '"dataset": "...", "experimental_setup": "...", "models_or_algorithms": ["..."], '
            '"evaluation_metrics": ["..."], "key_results": ["..."], '
            '"findings_summary": "...", "limitations": ["..."], "future_work": ["..."], '
            '"novelty": "...", "research_gaps": ["..."], "contribution_summary": "...", '
            '"confidence": 0.0-1.0}'
        )

    @staticmethod
    def _sample_chunks(chunks: Sequence[Any], *, limit: int) -> list[Any]:
        """Front-load the paper: head, then an even spread of the body."""
        if len(chunks) <= limit:
            return list(chunks)
        head_count = max(3, limit // 4)
        head = list(chunks[:head_count])
        remaining = list(chunks[head_count:])
        spread = max(0, limit - len(head))
        step = max(1, len(remaining) // spread) if spread else 1
        body = remaining[::step][:spread]
        return [*head, *body]

    @staticmethod
    def _render_document(chunks: Sequence[Any], *, max_chars: int = 24_000) -> str:
        """Render whole-document chunks with numbered, citable markers."""
        blocks: list[str] = []
        total = 0
        for index, chunk in enumerate(chunks, start=1):
            header = ContextCompressor.citation_header(chunk, marker=index)
            block = f"[{index}] {header}\n{chunk.text.strip()}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block) + 2
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[;\n]", value) if p.strip()]
        return parts
    return [str(item).strip() for item in value if str(item).strip()]


def _as_optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1500 <= year <= 2100 else None


def _guess_year(chunks: Sequence[Any]) -> int | None:
    """Fall back to a year mentioned in the opening text."""
    head = " ".join(chunk.text[:400] for chunk in chunks[:3])
    match = _YEAR_RE.search(head)
    return int(match.group(0)) if match else None


def _extract_bullets(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip().startswith("-")]


def _clamp(value: Any, low: float, high: float, *, default: float = 0.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default
