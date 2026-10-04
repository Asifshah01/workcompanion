"""Research / paper-analysis schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from workcompanion.schemas.common import Citation, SourceKind


class WebSource(BaseModel):
    """A web result, kept strictly separate from document evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    url: str
    snippet: str = ""
    provider: str = "duckduckgo"
    rank: int = 0

    def citation(self, marker: str) -> Citation:
        return Citation(
            marker=marker,
            document_name=self.title,
            url=self.url,
            source_kind=SourceKind.WEB,
            quote=self.snippet[:280] or None,
            relevance=0.5,
        )


class PaperAnalysis(BaseModel):
    """Structured analysis of a single research paper."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document_id: str
    document_name: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    affiliation: str | None = None
    year: int | None = None
    venue: str | None = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)

    research_problem: str = ""
    objectives: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)

    methodology: str = ""
    method_details: list[str] = Field(default_factory=list)
    dataset: str = ""
    experimental_setup: str = ""
    models_or_algorithms: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)

    key_results: list[str] = Field(default_factory=list)
    findings_summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)

    novelty: str = ""
    research_gaps: list[str] = Field(default_factory=list)
    contribution_summary: str = ""
    citation_count_hint: str | None = None

    explained: dict[str, str] = Field(default_factory=dict, description="level -> explanation")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    completeness: float = 0.0

    def recompute_completeness(self) -> "PaperAnalysis":
        """Fraction of the analytical fields the extractor actually filled."""
        checks = [
            self.title, self.abstract, self.research_problem, self.objectives,
            self.methodology, self.dataset, self.experimental_setup,
            self.key_results, self.limitations, self.future_work, self.novelty,
            self.research_gaps, self.findings_summary, self.contribution_summary,
        ]
        filled = sum(1 for value in checks if value and str(value).strip())
        list_fields = [
            self.authors, self.keywords, self.method_details, self.models_or_algorithms,
            self.evaluation_metrics, self.research_gaps, self.limitations,
        ]
        filled += sum(0.5 for value in list_fields if value)
        self.completeness = round(min(filled / (len(checks) + 0.5 * len(list_fields)), 1.0), 3)
        return self


class ResearchBrief(BaseModel):
    """A synthesized answer to a research question (documents and/or web)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    answer: str = ""
    document_findings: list[str] = Field(default_factory=list)
    web_findings: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    conflicting_points: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    suggested_readings: list[WebSource] = Field(default_factory=list)
    confidence: float = 0.0
    used_web: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def evidence_kind(self) -> Literal["documents", "web", "mixed", "none"]:
        has_docs = bool(self.document_findings)
        has_web = self.used_web and bool(self.web_findings or self.suggested_readings)
        if has_docs and has_web:
            return "mixed"
        if has_docs:
            return "documents"
        if has_web:
            return "web"
        return "none"