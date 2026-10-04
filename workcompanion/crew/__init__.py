"""CrewAI orchestration for WorkCompanion AI.

Crews are the *optional* orchestration layer.  The default path calls agents
directly, which is faster and fully deterministic; a crew is used when the work
is genuinely multi-step and the extra planning step pays for itself.

    from workcompanion.crew import build_workflow, run_workflow

    workflow = build_workflow("deep_research", "What is the current state of ...")
    outcome = run_workflow(workflow, bundle)
    if outcome is None:          # crews unavailable -> direct agent path
        result = bundle.research.run("What is the current state of ...")

No import of :mod:`crewai` happens at module scope, so the application still
starts when the optional dependency is absent.
"""

from __future__ import annotations

from workcompanion.crew.llm import GROQ_BASE_URL, build_crew_llm, crew_enabled, crewai_available
from workcompanion.crew.roles import ROLES, ROLE_LABELS, WORKFLOW_ROLES, CrewRole, roles_for
from workcompanion.crew.tools import CrewToolbox
from workcompanion.crew.workflows import (
    CrewOutcome,
    CrewWorkflow,
    Step,
    build_workflow,
    run_workflow,
    workflow_names,
)

__all__ = [
    "CrewOutcome",
    "CrewRole",
    "CrewToolbox",
    "CrewWorkflow",
    "GROQ_BASE_URL",
    "ROLES",
    "ROLE_LABELS",
    "Step",
    "WORKFLOW_ROLES",
    "build_crew_llm",
    "build_workflow",
    "crew_enabled",
    "crewai_available",
    "roles_for",
    "run_workflow",
    "workflow_names",
]