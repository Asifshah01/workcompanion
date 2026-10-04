"""CrewAI role definitions.

One CrewAI ``Agent`` per WorkCompanion specialist.  ``role``/``goal``/``backstory``
are derived from the agent's own identity (its ``name`` and ``description``) so
the crew never advertises a capability the underlying agent does not have.

Roles, not tools, are what make the multi-agent behaviour legible: each crew
member is given exactly the tools its WorkCompanion counterpart needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workcompanion.config.logging_config import get_logger

logger = get_logger(__name__)

#: Ground rules appended to every role's backstory so the crew keeps the same
#: evidence discipline the direct agent path enforces.
_SHARED_BACKSTORY = (
    " You work inside WorkCompanion AI. Two rules are non-negotiable: "
    "(1) never invent a fact, a quote or a citation - if the evidence is not in "
    "the passages or search results you were given, say so explicitly; "
    "(2) always cite the numbered source that supports a claim."
)


@dataclass(frozen=True)
class CrewRole:
    """Declarative description of one crew member."""

    key: str
    role: str
    goal: str
    backstory: str
    #: Tool names from :class:`~workcompanion.crew.tools.CrewToolbox`.
    tools: tuple[str, ...] = ()
    allow_delegation: bool = False

    def as_crew_agent(self, llm: Any, toolbox: dict[str, Any], *, max_iter: int) -> Any:
        """Materialise this role as a CrewAI ``Agent``."""
        from crewai import Agent

        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory + _SHARED_BACKSTORY,
            llm=llm,
            tools=[toolbox[name] for name in self.tools if name in toolbox],
            allow_delegation=self.allow_delegation,
            max_iter=max_iter,
            verbose=False,
        )


#: The canonical crew roster, keyed by the WorkCompanion agent name.
ROLES: dict[str, CrewRole] = {
    "research_lead": CrewRole(
        key="research_lead",
        role="Research Lead",
        goal=(
            "Break the learner's question into sub-questions, decide which sources "
            "are needed, and merge the findings into one cited answer."
        ),
        backstory=(
            "You are a research strategist. You plan the search before anyone reads a "
            "single paper, and you are the last word on what the evidence supports."
        ),
        tools=("search_notes", "search_web", "analyze_paper"),
        allow_delegation=True,
    ),
    "source_analyst": CrewRole(
        key="source_analyst",
        role="Source Analyst",
        goal=(
            "Extract the claims, method and limitations from the learner's documents "
            "and any web results, quoting the source of each one."
        ),
        backstory=(
            "You read closely and never paraphrase a finding into something stronger "
            "than the source states."
        ),
        tools=("search_notes", "search_web", "analyze_paper"),
    ),
    "teacher": CrewRole(
        key="teacher",
        role="Subject Tutor",
        goal=(
            "Explain the concept so the learner actually understands it, adapting the "
            "depth to their stated level and building on their weak areas."
        ),
        backstory=(
            "You have taught this subject for a decade. You start from what the "
            "learner already knows, use worked examples, and name the mistake they are "
            "most likely to make."
        ),
        tools=("search_notes", "teach_concept"),
    ),
    "assessment_designer": CrewRole(
        key="assessment_designer",
        role="Assessment Designer",
        goal=(
            "Write a quiz that tests real understanding of the retrieved material, "
            "with a correct answer, a rationale and a citation for every item."
        ),
        backstory=(
            "You write questions that a marker would defend: one unambiguous right "
            "answer, distractors that expose a real misconception, and a citation "
            "behind each item."
        ),
        tools=("search_notes", "generate_quiz"),
    ),
    "revision_specialist": CrewRole(
        key="revision_specialist",
        role="Revision Specialist",
        goal=(
            "Turn the material into atomic flashcards and a spaced-repetition schedule "
            "that fits the learner's remaining time."
        ),
        backstory=(
            "You optimise for retention: one idea per card, and revision booked on "
            "expanding intervals rather than crammed into the last night."
        ),
        tools=("search_notes", "generate_flashcards", "make_study_plan"),
    ),
    "quality_critic": CrewRole(
        key="quality_critic",
        role="Grounding Critic",
        goal=(
            "Check the draft answer against the retrieved evidence, remove anything "
            "unsupported, and insist the learner is told what is still unknown."
        ),
        backstory=(
            "You are the last gate before the learner sees an answer. Unsupported "
            "claims, vague confidence and missing citations are your speciality."
        ),
        tools=("search_notes",),
    ),
}

#: Workflow name -> ordered roles. A crew is built per workflow, not one global crew.
WORKFLOW_ROLES: dict[str, tuple[str, ...]] = {
    "deep_research": ("research_lead", "source_analyst", "teacher", "quality_critic"),
    "exam_prep": ("assessment_designer", "revision_specialist", "teacher"),
    "study_session": ("teacher", "assessment_designer"),
}

#: Role -> human label used in the UI trace panel.
ROLE_LABELS: dict[str, str] = {key: role.role for key, role in ROLES.items()}


def roles_for(workflow: str) -> list[CrewRole]:
    """Roles participating in a workflow, in execution order."""
    return [ROLES[key] for key in WORKFLOW_ROLES.get(workflow, ())]


def workflow_names() -> list[str]:
    return sorted(WORKFLOW_ROLES)


__all__ = [
    "CrewRole",
    "ROLES",
    "WORKFLOW_ROLES",
    "ROLE_LABELS",
    "roles_for",
    "workflow_names",
]