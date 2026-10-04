"""WorkCompanion agents.

Each agent owns one capability and returns an
:class:`~workcompanion.schemas.common.AgentResult` envelope (or a typed schema
for structured deliverables such as :class:`QuizSet`).

    NEXUS  -> routes every request and extracts its parameters
    Tutor  -> adaptive teaching across four levels
    Socratic -> scaffolded dialogue that never gives the answer away
    Knowledge (RAG) -> grounded, cited answers; refuses when evidence is missing
    Research -> paper analysis + document/web synthesis, kept separate
    Quiz -> grounded generation and grading
    Flashcards -> atomic, citation-backed decks
    Planner -> spaced-repetition schedules
"""

from __future__ import annotations

from workcompanion.agents.base import AgentBase
from workcompanion.agents.flashcard_agent import FlashcardAgent
from workcompanion.agents.nexus_agent import NexusAgent, WorkflowPlan, plan_workflow
from workcompanion.agents.planner_agent import PlannerAgent
from workcompanion.agents.quiz_agent import QuizAgent
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.agents.research_agent import ResearchAgent
from workcompanion.agents.socratic_agent import SocraticAgent
from workcompanion.agents.tutor_agent import TutorAgent

__all__ = [
    "AgentBase",
    "FlashcardAgent",
    "NexusAgent",
    "PlannerAgent",
    "QuizAgent",
    "RagAgent",
    "ResearchAgent",
    "SocraticAgent",
    "TutorAgent",
    "WorkflowPlan",
    "plan_workflow",
    "AGENT_REGISTRY",
    "agent_catalogue",
]

#: name -> (class, one-line responsibility) used by the UI and health checks.
AGENT_REGISTRY: dict[str, tuple[type[AgentBase], str]] = {
    NexusAgent.name: (NexusAgent, "Routes every request and extracts parameters"),
    TutorAgent.name: (TutorAgent, "Adaptive teaching across four explanation levels"),
    SocraticAgent.name: (SocraticAgent, "Scaffolded dialogue and misconception probing"),
    RagAgent.name: (RagAgent, "Cited, retrieval-grounded answers from your material"),
    ResearchAgent.name: (ResearchAgent, "Paper analysis and document/web research synthesis"),
    QuizAgent.name: (QuizAgent, "Grounded quiz generation, grading and weak-topic reports"),
    FlashcardAgent.name: (FlashcardAgent, "Atomic citation-backed flashcards for spaced repetition"),
    PlannerAgent.name: (PlannerAgent, "Spaced-repetition study plans from learner constraints"),
}


def agent_catalogue() -> list[dict[str, str]]:
    """Flat list of agents for the UI's agent-trace panel."""
    return [
        {"name": name, "class": agent.__name__, "description": description}
        for name, (agent, description) in AGENT_REGISTRY.items()
    ]
