"""CrewAI workflows.

A *workflow* is an ordered sequence of CrewAI tasks, each owned by one role, run
as a single sequential ``Crew``.  Crews add value where the work is genuinely
multi-step (research -> critique, assess -> revise) and are deliberately **not**
used for single-shot work: asking, grading and planning are faster and more
reliable through the direct agent path.

Every entry point returns ``None`` instead of raising when crews are unavailable
(no API key, ``ENABLE_CREWAI=false``, crewai not installed), so callers can fall
back with a single ``if result is None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.crew.llm import build_crew_llm, crew_enabled
from workcompanion.crew.roles import CrewRole, roles_for, workflow_names
from workcompanion.crew.tools import CrewToolbox

logger = get_logger(__name__)


@dataclass
class Step:
    """One task in a workflow."""

    key: str
    role: str
    description: str
    expected_output: str
    depends_on: tuple[str, ...] = ()

    def as_crew_task(self, role: CrewRole, context: dict[str, Any] | None = None) -> Any:
        """Materialise this step as a CrewAI ``Task``."""
        from crewai import Task

        return Task(
            description=self.description,
            expected_output=self.expected_output,
            agent=role.as_crew_agent(
                context["llm"], context["tools"], max_iter=context["max_iter"]
            )
            if context
            else None,
            context=context.get("outputs") if context else None,
            markdown=True,
        )


@dataclass
class CrewWorkflow:
    """A named, ordered set of steps plus its success criteria."""

    name: str
    question: str
    steps: list[Step]
    topic: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow builders
# ---------------------------------------------------------------------------
def _deep_research(question: str, **options: Any) -> CrewWorkflow:
    topic = options.get("topic")
    return CrewWorkflow(
        name="deep_research",
        question=question,
        topic=topic,
        options=options,
        steps=[
            Step(
                key="plan",
                role="research_lead",
                description=(
                    f"The learner asked: {question}\n\n"
                    "Break this into at most four sub-questions. For each, state whether "
                    "the learner's own documents are likely to answer it or whether the "
                    "web is needed. Do not answer the question yet."
                ),
                expected_output=(
                    "A numbered list of sub-questions, each tagged [docs] or [web], plus "
                    "the order you will tackle them in."
                ),
            ),
            Step(
                key="gather",
                role="source_analyst",
                description=(
                    "Using the sub-questions from the research plan, search the learner's "
                    "documents and the web for each one. Extract what each source actually "
                    "says, quoting it and keeping its [n] / [Wn] marker. If a sub-question "
                    "has no evidence, write 'no evidence found' - do not fill the gap."
                ),
                expected_output=(
                    "Per sub-question: the evidence with markers, or 'no evidence found'."
                ),
                depends_on=("plan",),
            ),
            Step(
                key="teach",
                role="teacher",
                description=(
                    "Using only the evidence above, write the learner's answer. Start with "
                    "the direct answer in 2-4 sentences, then the supporting evidence with "
                    "markers, then how the pieces fit together. Mark clearly anything the "
                    "evidence does not settle."
                ),
                expected_output="A structured, cited answer in Markdown.",
                depends_on=("gather",),
            ),
            Step(
                key="critique",
                role="quality_critic",
                description=(
                    "Audit the draft answer line by line against the gathered evidence. "
                    "Delete or flag every claim without a marker. Flag any over-confidence "
                    "and add a short 'What is still unknown' list."
                ),
                expected_output=(
                    "The corrected answer plus a short list of any removals you made."
                ),
                depends_on=("teach",),
            ),
        ],
    )


def _exam_prep(question: str, **options: Any) -> CrewWorkflow:
    topic = options.get("topic") or question
    count = int(options.get("count") or 8)
    days = int(options.get("days") or 7)
    hours = float(options.get("hours_per_day") or 2.0)
    return CrewWorkflow(
        name="exam_prep",
        question=question,
        topic=topic,
        options=options,
        steps=[
            Step(
                key="diagnose",
                role="assessment_designer",
                description=(
                    f"Topic: {topic}. Search the learner's material for it and list the "
                    "distinct sub-skills it covers. Then state which of them are most "
                    "likely to be misunderstood."
                ),
                expected_output="A list of sub-skills, hardest first.",
            ),
            Step(
                key="assess",
                role="assessment_designer",
                description=(
                    f"Write {count} exam questions on {topic} covering the sub-skills you "
                    "just listed. Each question needs one unambiguous correct answer, "
                    "distractors built from real misconceptions, and the source marker it "
                    "came from. Use the generate_quiz tool, then check every item against "
                    "the retrieved passages."
                ),
                expected_output=f"{count} numbered questions with answers and citations.",
                depends_on=("diagnose",),
            ),
            Step(
                key="schedule",
                role="revision_specialist",
                description=(
                    f"The learner has {days} day(s) left and can study {hours:g} h/day. "
                    f"Build a spaced-repetition plan for {topic} and produce flashcards for "
                    "the sub-skills that matter most. Use the make_study_plan and "
                    "generate_flashcards tools."
                ),
                expected_output=(
                    "A day-by-day schedule and a set of atomic flashcards."
                ),
                depends_on=("diagnose",),
            ),
            Step(
                key="teach_back",
                role="teacher",
                description=(
                    "Explain the two hardest sub-skills from the diagnosis, then give the "
                    "learner the three things most likely to cost them marks."
                ),
                expected_output="Two explanations and a short 'watch out for' list.",
                depends_on=("diagnose",),
            ),
        ],
    )


def _study_session(question: str, **options: Any) -> CrewWorkflow:
    level = options.get("level") or "undergraduate"
    return CrewWorkflow(
        name="study_session",
        question=question,
        topic=options.get("topic"),
        options=options,
        steps=[
            Step(
                key="teach",
                role="teacher",
                description=(
                    f"Teach {question} at the '{level}' level. Ground the explanation in "
                    "the learner's own documents first (use search_notes), then explain "
                    "step by step with a worked example and name the common mistake."
                ),
                expected_output="A cited, level-appropriate explanation in Markdown.",
            ),
            Step(
                key="check",
                role="assessment_designer",
                description=(
                    f"Write two questions that test whether {question} is really "
                    "understood - one recall, one application. Give the answers and the "
                    "citation behind each."
                ),
                expected_output="Two questions with answers and citations.",
                depends_on=("teach",),
            ),
        ],
    )


_BUILDERS = {
    "deep_research": _deep_research,
    "exam_prep": _exam_prep,
    "study_session": _study_session,
}


def build_workflow(name: str, question: str, **options: Any) -> CrewWorkflow:
    """Create the step list for a named workflow."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"Unknown crew workflow {name!r}. Available: {', '.join(workflow_names())}")
    return builder(question, **options)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@dataclass
class CrewOutcome:
    """Result of one crew run, including per-step output for the UI trace."""

    workflow: str
    question: str
    answer: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "engine": "crewai",
            "workflow": self.workflow,
            "steps": [key for key, _ in self.steps],
            "usage": self.usage,
        }


def run_workflow(
    workflow: CrewWorkflow,
    agents: Any,
    settings: Settings | None = None,
    *,
    llm: Any | None = None,
) -> CrewOutcome | None:
    """Execute a workflow as a sequential crew.

    Returns ``None`` when crews are unavailable, which is the signal for the
    caller to use the direct agent path instead.
    """
    settings = settings or get_settings()
    if not crew_enabled(settings):
        logger.info("[crew] '%s' skipped: crews are disabled.", workflow.name)
        return None

    llm = llm or build_crew_llm(settings)
    if llm is None:
        return None

    try:
        from crewai import Crew, Process
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("[crew] crewai unavailable (%s).", exc)
        return None

    toolbox = CrewToolbox(agents, settings)
    tools = toolbox.build()
    by_name = {getattr(tool, "name", f"tool_{index}"): tool for index, tool in enumerate(tools)}

    role_defs = roles_for(workflow.name)
    role_by_key = {role.key: role for role in role_defs}

    outputs: dict[str, Any] = {}
    steps: list[tuple[str, str]] = []
    tasks: list[Any] = []
    agents_by_role: dict[str, Any] = {}
    context: dict[str, Any] = {"llm": llm, "tools": by_name, "max_iter": settings.max_agent_steps}

    for step in workflow.steps:
        role = role_by_role(step.role)
        if role is None:
            logger.warning("[crew] step %s references unknown role %s", step.key, step.role)
            continue
        if step.key not in agents_by_role:
            agents_by_role[step.role] = role.as_crew_agent(llm, by_name, max_iter=settings.max_agent_steps)
        task = step.as_crew_task(role, {**context, "outputs": _render_context(outputs)})
        task.agent = agents_by_role[step.role]
        tasks.append(task)
        outputs[step.key] = task

    if not tasks:
        return None

    crew = Crew(
        agents=list(agents_by_role.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=settings.crewai_verbose,
    )

    started = now_ms()
    try:
        result = crew.kickoff()
    except Exception as exc:
        logger.warning("[crew] '%s' failed (%s); falling back to the direct path.", workflow.name, exc)
        return None

    for step in workflow.steps:
        task = outputs.get(step.key)
        if task is not None and getattr(task, "output", None):
            steps.append((step.key, str(task.output)))

    return CrewOutcome(
        workflow=workflow.name,
        question=workflow.question,
        answer=str(getattr(result, "raw", result) or "").strip(),
        steps=steps,
        usage={"latency_ms": round(now_ms() - started, 2), **(getattr(result, "token_usage", {}) or {})},
    )


def _render_context(outputs: dict[str, Any], *, limit: int = 6000) -> str:
    """Render completed task outputs as context for the next step."""
    rendered: list[str] = []
    for key, task in outputs.items():
        text = str(getattr(task, "output", "") or "").strip()
        if text:
            rendered.append(f"### {key}\n{text}")
    joined = "\n\n".join(reversed(rendered))
    return joined[-limit:]


def now_ms() -> float:
    """Monotonic milliseconds (kept local so this module stays import-light)."""
    import time

    return time.perf_counter() * 1000


__all__ = [
    "CrewWorkflow",
    "CrewOutcome",
    "Step",
    "build_workflow",
    "run_workflow",
    "workflow_names",
]