"""CrewAI tools that expose the WorkCompanion agents to a crew.

Each tool is a thin adapter: it receives plain strings (what an LLM can produce)
and returns a formatted string (what an LLM can read).  No agent logic lives
here - the tools only marshal arguments into the agent APIs so that a crew's
reasoning step and the direct agent path can never drift apart.
"""

from __future__ import annotations

import inspect
from typing import Any

from workcompanion.config.logging_config import get_logger
from workcompanion.config.settings import Settings, get_settings
from workcompanion.rag.citations import citations_markdown

logger = get_logger(__name__)

#: Cap the text a tool hands to the LLM, so one dense document cannot blow the
#: crew's context window.
MAX_TOOL_CHARS = 6000


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class CrewToolbox:
    """Builds the CrewAI tool set from an :class:`AgentBundle`."""

    def __init__(self, agents: Any, settings: Settings | None = None):
        self._agents = agents
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    def build(self) -> list[Any]:
        """Every tool the crews may use."""
        return [
            self.search_notes(),
            self.search_web(),
            self.teach_concept(),
            self.generate_quiz(),
            self.generate_flashcards(),
            self.make_study_plan(),
            self.analyze_paper(),
        ]

    # ------------------------------------------------------------------
    def search_notes(self) -> Any:
        """Search the learner's uploaded material for passages about a topic."""

        def search_notes(query: str, top_k: int = 5) -> str:
            """Search the learner's indexed documents and return cited passages.

            Args:
                query: What to look for, phrased as a natural-language question.
                top_k: How many passages to return (1-10).
            """
            rag = self._agents.rag
            outcome = rag.retrieve(query, top_k=max(1, min(int(top_k), 10)))
            if not outcome.chunks:
                return "No passage in the learner's material matched this query."
            passages = "\n\n".join(
                f"[{index}] {_clip(chunk.chunk.text, 900)}"
                for index, chunk in enumerate(outcome.chunks, start=1)
            )
            return _clip(f"{passages}\n\n{citations_markdown(outcome.citations)}")

        return self._as_tool(search_notes, name="search_notes")

    def search_web(self) -> Any:
        """Look up a research question on the public web."""

        def search_web(query: str, max_results: int = 5) -> str:
            """Search the web and return titles, URLs and snippets.

            Args:
                query: The research question.
                max_results: How many results to return (1-8).
            """
            if not self._settings.enable_web_research:
                return "Web research is disabled in Settings."
            tool = self._agents.research.web_tool
            if not tool.enabled:
                return f"Web search is unavailable: {tool.last_error or 'not configured'}"
            sources = tool.search(query, max_results=max(1, min(int(max_results), 8)))
            if not sources:
                return "No web results were returned."
            return _clip(
                "\n\n".join(
                    f"[W{index}] {source.title}\nURL: {source.url}\n{source.snippet}"
                    for index, source in enumerate(sources, start=1)
                )
            )

        return self._as_tool(search_web, name="search_web")

    def teach_concept(self) -> Any:
        """Explain a concept at a chosen level, grounded in the learner's notes."""

        def teach_concept(
            question: str, level: str = "undergraduate", style: str = "detailed"
        ) -> str:
            """Explain a concept to the learner at the requested level.

            Args:
                question: The concept or question to explain.
                level: beginner | undergraduate | graduate | expert.
                style: concise | detailed | socratic | example_based | mathematical | analogy.
            """
            from workcompanion.schemas.tutor import TutorRequest

            result = self._agents.tutor.teach(
                TutorRequest(
                    question=question,
                    level=level,  # type: ignore[arg-type]
                    style=style,  # type: ignore[arg-type]
                )
            )
            return _clip(result.answer)

        return self._as_tool(teach_concept, name="teach_concept")

    def generate_quiz(self) -> Any:
        """Build a grounded quiz on a topic."""

        def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> str:
            """Generate a grounded quiz with an answer key.

            Args:
                topic: The topic to test.
                num_questions: How many questions (1-20).
                difficulty: easy | medium | hard | expert.
            """
            quiz = self._agents.quiz.generate_quiz(
                topic,
                num_questions=max(1, min(int(num_questions), 20)),
                difficulty=difficulty,  # type: ignore[arg-type]
            )
            lines = [f"# {quiz.title}"]
            for index, question in enumerate(quiz.questions, start=1):
                lines.append(f"\n**{index}. {question.question}**")
                for option in question.options:
                    lines.append(f"- {option}")
                lines.append(f"- Answer: {question.correct_answer}")
            return _clip("\n".join(lines))

        return self._as_tool(generate_quiz, name="generate_quiz")

    def generate_flashcards(self) -> Any:
        """Build a revision deck."""

        def generate_flashcards(topic: str, count: int = 10) -> str:
            """Generate atomic flashcards for spaced repetition.

            Args:
                topic: The topic to cover.
                count: How many cards (1-40).
            """
            deck = self._agents.flashcards.generate_deck(
                topic, count=max(1, min(int(count), 40))
            )
            return _clip(
                "\n".join(f"- **{card.front}** -> {card.back}" for card in deck.cards)
            )

        return self._as_tool(generate_flashcards, name="generate_flashcards")

    def make_study_plan(self) -> Any:
        """Turn learner constraints into a spaced-repetition schedule."""

        def make_study_plan(
            subject: str, days_available: int = 7, hours_per_day: float = 2.0,
            weak_areas: str = "", exam_date: str = "",
        ) -> str:
            """Build a day-by-day study plan.

            Args:
                subject: What is being studied.
                days_available: How many days remain before the exam.
                hours_per_day: Maximum hours per day.
                weak_areas: Comma-separated topics the learner is weakest at.
                exam_date: Exam date as YYYY-MM-DD, or empty if unknown.
            """
            from datetime import date as date_type

            from workcompanion.schemas.planner import StudyPlanInput

            plan = self._agents.planner.create_plan(
                StudyPlanInput(
                    subject=subject,
                    days_available=max(1, min(int(days_available), 60)),
                    hours_per_day=max(0.25, min(float(hours_per_day), 12.0)),
                    weak_areas=[part.strip() for part in weak_areas.split(",") if part.strip()],
                    exam_date=_parse_date(exam_date),
                )
            )
            return _clip(self._agents.planner.render_plan(plan))

        return self._as_tool(make_study_plan, name="make_study_plan")

    def analyze_paper(self) -> Any:
        """Run a structured analysis over one indexed document."""

        def analyze_paper(document_id: str, level: str = "undergraduate") -> str:
            """Analyse an indexed document: problem, method, results, limitations.

            Args:
                document_id: The document identifier from the knowledge base.
                level: beginner | undergraduate | graduate | expert.
            """
            analysis = self._agents.research.analyze_paper(document_id, level=level)
            return _clip(self._agents.research.render_analysis(analysis))

        return self._as_tool(analyze_paper, name="analyze_paper")

    # ------------------------------------------------------------------
    def _as_tool(self, func: Any, *, name: str) -> Any:
        """Wrap a plain function in a CrewAI ``BaseTool`` (with a stub fallback)."""
        try:
            from crewai.tools import BaseTool
        except Exception:  # pragma: no cover - optional dependency
            return _FallbackTool(name=name, func=func)

        # Bound to locals first: a class body cannot read an enclosing name that
        # it assigns to (``name: str = name`` raises NameError).
        tool_name = name
        tool_description = (func.__doc__ or tool_name).strip()
        tool_schema = build_args_model(func)

        class _Tool(BaseTool):  # type: ignore[misc]
            name: str = tool_name
            description: str = tool_description
            args_schema: type = tool_schema

            def _run(self, **kwargs: Any) -> str:  # type: ignore[override]
                return func(**kwargs)

        _Tool.__name__ = f"{tool_name}_tool"
        return _Tool()


class _FallbackTool:
    """Stand-in used when crewai is not installed, so callers need no branches."""

    def __init__(self, *, name: str, func: Any):
        self.name = name
        self.description = (func.__doc__ or name).strip()
        self._func = func

    def run(self, **kwargs: Any) -> str:
        return self._func(**kwargs)


def _parse_date(value: str):
    from datetime import date, datetime

    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def build_args_model(func: Any) -> Any:
    """Build a pydantic args schema from a function's signature and docstring.

    CrewAI needs a model describing each tool's parameters; the Google-style
    ``Args:`` block in the tool docstring becomes the per-field description, so
    the schema never drifts from the documentation the model already reads.
    """
    from pydantic import Field, create_model

    arg_docs: dict[str, str] = {}
    in_args = False
    for line in (func.__doc__ or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if not in_args:
            continue
        if not stripped:
            break
        name, _, description = stripped.partition(":")
        arg_docs[name.strip().split("(")[0]] = description.strip()

    properties: dict[str, Any] = {}
    for name, parameter in inspect.signature(func).parameters.items():
        annotation = str if parameter.annotation is inspect.Parameter.empty else parameter.annotation
        default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
        properties[name] = (annotation, Field(default, description=arg_docs.get(name, "")))

    model_name = "".join(part.title() for part in func.__name__.split("_")) + "Args"
    return create_model(model_name, **properties)


__all__ = ["CrewToolbox", "build_args_model", "MAX_TOOL_CHARS"]