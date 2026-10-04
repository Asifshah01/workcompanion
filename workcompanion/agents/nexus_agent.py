"""NEXUS - the routing / orchestration agent.

NEXUS is the only agent the learner talks to in Auto mode.  It:

1. classifies the request into an :class:`~workcompanion.schemas.tutor.Intent`,
2. extracts the parameters the downstream agent needs (counts, difficulty,
   level, document scope) so the UI never has to re-ask,
3. checks that the prerequisites exist (documents indexed, web research enabled)
   and says so plainly when they do not,
4. decides whether the request is single-shot or a **workflow** (for example
   "teach me thermodynamics then quiz me" is a two-stage workflow).

Routing is LLM-assisted with a deterministic rule-based fallback, so Auto mode
never dead-ends.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.tutor import (
    ExplanationLevel,
    Intent,
    RouteDecision,
    TeachingStyle,
)
from workcompanion.utils.coerce import coerce_int
from workcompanion.utils.text_utils import matches_keywords

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are NEXUS, the router inside WorkCompanion AI - a multi-agent \
study and research platform. You classify what the learner wants and hand it to \
the right specialist agent.

You DO NOT answer the question yourself. You only decide who should handle it and \
with which parameters.

INTENT CHOICES (pick the single best one)
- explain / teach  : learner wants a concept explained or taught
- answer           : a factual question about their material
- summarize        : condense material into a shorter form
- compare          : contrast documents, theories or sources
- quiz             : generate a quiz / test me / exam questions
- exam             : a full timed exam paper
- flashcards       : revision cards / spaced repetition
- plan             : build or adjust a study schedule
- paper_analysis   : analyse a research paper (methodology, gaps)
- research         : research a question, possibly with web sources
- socratic         : guide me without telling me the answer
- search           : find passages, not an answer
- progress         : how am I doing / my weak areas / my streak
- small_talk       : greetings and meta questions about the app
- unclear          : genuinely cannot tell

PARAMETERS
- Set needs_documents=true when the intent is answerable from the learner's files.
- Set use_web=true only for research / latest-news style requests.
- Extract counts ("make 8 questions"), difficulty ("hard"), level ("explain like I'm a \
beginner") and topic whenever the learner states them; leave null otherwise.
- Set interactive=true for quiz/exam intents and false otherwise.

If the request implies several steps (e.g. "teach me X then test me"), still pick the \
primary intent but describe the extra step in "rationale".

Return STRICT JSON only:
{"intent": "explain", "confidence": 0.9, "needs_documents": true, "use_web": false,
 "explanation_level": "beginner" | null, "teaching_style": "detailed" | null,
 "difficulty": "easy|medium|hard|expert" | null, "question_count": 8 | null,
 "flashcard_count": null, "topic": "..." | null, "document_filter": null,
 "interactive": false, "timed": false, "rationale": "one sentence",
 "follow_ups": ["up to 3 suggested replies the learner can tap"]}"""

#: Keyword table used by the offline / fallback router, most specific first.
#: Order matters: "research" contains "search", so the research rules must win.
_RULES: tuple[tuple[tuple[str, ...], Intent], ...] = (
    (("study plan", "revision schedule", "schedule my", "plan my", "timetable"), Intent.PLAN),
    (("exam paper", "mock exam", "exam mode", "full exam", "past paper"), Intent.EXAM),
    (("quiz", "test me", "mcq", "practice questions", "viva", "question paper"), Intent.QUIZ),
    (("flashcard", "flash card", "revision card", "spaced repetition", "anki"), Intent.FLASHCARDS),
    (
        ("analyse this paper", "analyze this paper", "paper analysis", "methodology of this",
         "research gap", "critical analysis", "summarise this paper", "summarize this paper"),
        Intent.PAPER_ANALYSIS,
    ),
    (("compare", "versus", " vs ", "vs.", "difference between", "contradict", "conflict"), Intent.COMPARE),
    (("socratic", "guide me", "hint me", "don't tell me", "dont tell me"), Intent.SOCRATIC),
    (("summarise", "summarize", "tldr", "tl;dr", "in short", "condense"), Intent.SUMMARIZE),
    (("how am i", "my progress", "my score", "weak area", "streak", "improve me"), Intent.PROGRESS),
    (
        ("latest", "news", "on the internet", "online", "recent paper", "who is working on",
         "research", "find papers", "literature", "state of the art", "survey the"),
        Intent.RESEARCH,
    ),
    (("search for", "search my", "find where", "look up", "locate", "search"), Intent.SEARCH),
    (
        ("explain", "teach me", "how does", "what is", "why does", "walk me through",
         "i don't understand", "help me understand", "meaning of"),
        Intent.EXPLAIN,
    ),
    (("hi", "hello", "hey", "thanks", "thank you", "who are you", "what can you do"), Intent.SMALL_TALK),
)

#: Intents that are fully answerable without the learner's documents.
_NO_DOCUMENTS = frozenset(
    {Intent.SMALL_TALK, Intent.PROGRESS, Intent.UNCLEAR, Intent.RESEARCH}
)

_COUNT_RE = re.compile(
    r"\b(\d{1,3})\s*(?:\w+[- ]){0,2}?"
    r"(?:questions?|mcqs?|items?|cards?|flashcards?|problems?)\b"
)
_DAYS_RE = re.compile(r"\b(\d{1,3})\s*(?:days?|weeks?)\b")
_MINUTES_RE = re.compile(r"\b(\d{1,3}(?:\.\d)?)\s*(?:minutes?|mins?|hours?|hrs?)\b")
_WEEK = 7


class NexusAgent(AgentBase):
    """Intent router and workflow planner."""

    name = "nexus"
    description = "Routes every request to the right agent and extracts its parameters."

    def __init__(self, llm=None, settings=None, cache=None, *, has_documents: bool | None = None):
        super().__init__(llm, settings, cache)
        #: ``None`` means "not known yet" - the workflow checks at run time.
        self._has_documents = has_documents

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def set_document_state(self, has_documents: bool) -> None:
        self._has_documents = bool(has_documents)

    # ------------------------------------------------------------------
    def route(
        self,
        message: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        has_documents: bool | None = None,
        defaults: dict[str, Any] | None = None,
        learner_context: str | None = None,
    ) -> RouteDecision:
        """Classify a learner message and extract execution parameters."""
        text = (message or "").strip()
        if not text:
            return RouteDecision(
                intent=Intent.UNCLEAR,
                confidence=0.0,
                needs_documents=False,
                rationale="Empty message.",
                follow_ups=["Ask a question about your material", "Generate a quiz"],
            )

        from workcompanion.rag.query_transform import QueryTransformer

        conversation = QueryTransformer.conversation_prompt(text, list(history or []), limit=3)
        learner_block = f"LEARNER CONTEXT:\n{learner_context}\n\n" if learner_context else ""
        prompt = (
            f"{conversation}\n"
            f"{learner_block}"
            f"LEARNER MESSAGE:\n{text}\n\n"
            "Classify this message. Return STRICT JSON as specified."
        )
        request = self.build_request("route", user=prompt, temperature=0.1, max_tokens=700)

        payload: dict[str, Any] = {}
        try:
            payload = self.structured(request) or {}
        except Exception as exc:
            logger.info("[nexus] LLM routing unavailable, using rules: %s", exc)

        decision = self._from_payload(payload, text) if payload else None
        if decision is None:
            decision = self._rule_based(text)

        self._apply_defaults(decision, defaults or {})
        self._check_prerequisites(decision, text, has_documents)
        logger.info(
            "[nexus] %s -> %s (conf %.2f, docs=%s, web=%s)",
            text[:60], decision.intent.value, decision.confidence,
            decision.needs_documents, decision.use_web,
        )
        return decision

    def run(
        self,
        message: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        has_documents: bool | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Envelope describing the routing decision (shown in the agent trace)."""
        decision = self.route(
            message, history=history, has_documents=has_documents, defaults=defaults
        )
        answer = f"**Routed to:** `{decision.intent.value}`\n\n_{decision.rationale}_"
        if decision.follow_ups:
            answer += "\n\n" + "\n".join(f"- {item}" for item in decision.follow_ups)
        return self.result(
            answer,
            intent="route",
            confidence=decision.confidence,
            grounding=[GroundingLabel.MODEL_REASONING],
            metadata={
                "evidence": "user_input",
                "routed_intent": decision.intent.value,
                "needs_documents": decision.needs_documents,
                "use_web": decision.use_web,
                "workflow": plan_workflow(decision).name,
                "parameters": decision.model_dump(mode="json"),
            },
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _from_payload(self, payload: dict[str, Any], text: str) -> RouteDecision | None:
        """Build a decision from model JSON, or ``None`` if it is unusable."""
        raw_intent = str(payload.get("intent") or "").strip().lower()
        try:
            intent = Intent(raw_intent)
        except ValueError:
            logger.info("[nexus] unknown intent %r from model; using rules.", raw_intent)
            return None

        decision = RouteDecision(
            intent=intent,
            confidence=_clamp(payload.get("confidence"), 0.0, 1.0, default=0.7),
            needs_documents=bool(payload.get("needs_documents", True)),
            use_web=bool(payload.get("use_web", False)),
            explanation_level=_as_level(payload.get("explanation_level")),
            teaching_style=_as_style(payload.get("teaching_style")),
            difficulty=_as_difficulty(payload.get("difficulty")),
            question_count=_as_int(payload.get("question_count")),
            flashcard_count=_as_int(payload.get("flashcard_count")),
            topic=_as_str(payload.get("topic")),
            document_filter=_as_str(payload.get("document_filter")),
            interactive=bool(payload.get("interactive", intent in (Intent.QUIZ, Intent.EXAM))),
            timed=bool(payload.get("timed", intent is Intent.EXAM)),
            rationale=str(payload.get("rationale") or "").strip(),
            follow_ups=[str(item).strip() for item in (payload.get("follow_ups") or []) if str(item).strip()][:3],
        )
        # Explicit learner wording always wins over a model that omitted (or
        # guessed) the parameter - "5 hard questions" must survive routing.
        lowered = text.lower()
        typed = _typed_count(text)
        if typed:
            if decision.intent in (Intent.QUIZ, Intent.EXAM):
                decision.question_count = typed
            elif decision.intent is Intent.FLASHCARDS:
                decision.flashcard_count = typed
        if not decision.explanation_level:
            decision.explanation_level = _typed_level(lowered)
        if not decision.teaching_style:
            decision.teaching_style = _typed_style(lowered)
        if not decision.difficulty:
            decision.difficulty = _typed_difficulty(lowered)
        if not decision.topic:
            decision.topic = _typed_topic(text)
        return decision

    def _rule_based(self, text: str) -> RouteDecision:
        """Deterministic routing (offline mode, LLM failure, or nonsense output)."""
        lowered = text.lower()
        intent = Intent.UNCLEAR
        for keywords, candidate in _RULES:
            if matches_keywords(lowered, keywords):
                intent = candidate
                break

        count = _typed_count(text)
        decision = RouteDecision(
            intent=intent,
            confidence=0.62 if intent is not Intent.UNCLEAR else 0.3,
            needs_documents=intent not in _NO_DOCUMENTS,
            use_web=intent is Intent.RESEARCH and bool(self._settings.enable_web_research),
            explanation_level=_typed_level(lowered),
            teaching_style=_typed_style(lowered),
            difficulty=_typed_difficulty(lowered),
            question_count=count if intent in (Intent.QUIZ, Intent.EXAM) else None,
            flashcard_count=count if intent is Intent.FLASHCARDS else None,
            topic=_typed_topic(text),
            interactive=intent in (Intent.QUIZ, Intent.EXAM),
            timed=intent is Intent.EXAM,
            rationale="Rule-based routing (no language model decision available).",
        )
        if intent is Intent.UNCLEAR:
            decision.follow_ups = [
                "Explain a concept from my notes",
                "Quiz me on my weakest topic",
                "Summarise this document",
            ]
        return decision

    # ------------------------------------------------------------------
    def _apply_defaults(self, decision: RouteDecision, defaults: dict[str, Any]) -> None:
        """Fill unspecified parameters from the UI's current settings."""
        if not decision.explanation_level:
            decision.explanation_level = _as_level(
                defaults.get("level") or self._settings.default_explanation_level
            )
        if not decision.teaching_style:
            decision.teaching_style = _as_style(
                defaults.get("style") or self._settings.default_teaching_style
            )
        if not decision.difficulty:
            decision.difficulty = _as_difficulty(defaults.get("difficulty"))
        if decision.intent in (Intent.QUIZ, Intent.EXAM) and not decision.question_count:
            decision.question_count = int(defaults.get("question_count") or 6)
        if decision.intent is Intent.FLASHCARDS and not decision.flashcard_count:
            decision.flashcard_count = int(defaults.get("flashcard_count") or 15)
        if not decision.topic:
            decision.topic = _as_str(defaults.get("topic"))

    def _check_prerequisites(
        self, decision: RouteDecision, text: str, has_documents: bool | None
    ) -> None:
        """Warn (in the follow-ups) when a prerequisite is missing."""
        documents = self._has_documents if has_documents is None else has_documents

        if decision.needs_documents and documents is False:
            decision.rationale += (
                " The knowledge base is empty, so the learner must upload material first."
            )
            decision.follow_ups = [
                "Open My Knowledge to upload material",
                *decision.follow_ups,
            ][:3]

        if decision.use_web and not self._settings.enable_web_research:
            decision.use_web = False
            decision.rationale += " Web research is disabled in Settings."

        # A multi-stage request: teach me X then quiz me.
        if decision.intent in (Intent.EXPLAIN, Intent.TEACH) and _requests_second_step(text):
            decision.follow_ups = [
                f"Now quiz me on {decision.topic or 'this'}",
                *decision.follow_ups,
            ][:3]

    # ------------------------------------------------------------------
    @staticmethod
    def clarify(decision: RouteDecision) -> str:
        """Human-readable clarification prompt when routing failed."""
        if decision.intent is not Intent.UNCLEAR:
            return ""
        return (
            "I am not sure what you want me to do with that. Try one of these:"
            + "\n".join(f"- {item}" for item in decision.follow_ups[:3])
        )


# ---------------------------------------------------------------------------
# Workflow planning
# ---------------------------------------------------------------------------
#: Ordered stages a routed intent should run through.
_WORKFLOW_STAGES: dict[Intent, tuple[str, ...]] = {
    Intent.EXPLAIN: ("retrieve", "tutor", "verify"),
    Intent.TEACH: ("retrieve", "tutor", "verify"),
    Intent.ANSWER: ("retrieve", "knowledge", "verify"),
    Intent.SUMMARIZE: ("retrieve", "compress", "knowledge"),
    Intent.COMPARE: ("retrieve_per_document", "knowledge", "conflict_check"),
    Intent.QUIZ: ("retrieve", "quiz", "validate_grounding"),
    Intent.EXAM: ("retrieve_per_topic", "quiz", "validate_grounding", "grade"),
    Intent.FLASHCARDS: ("retrieve", "flashcards", "validate_grounding"),
    Intent.PLAN: ("progress_lookup", "planner"),
    Intent.PAPER_ANALYSIS: ("load_document", "research", "critique"),
    Intent.RESEARCH: ("retrieve", "web_search", "research", "synthesise"),
    Intent.SOCRATIC: ("retrieve", "socratic", "evaluate"),
    Intent.SEARCH: ("retrieve", "rank"),
    Intent.PROGRESS: ("progress_lookup",),
    Intent.SMALL_TALK: ("reply",),
    Intent.UNCLEAR: ("clarify",),
}

_WORKFLOW_NAMES = {
    Intent.EXAM: "exam_pipeline",
    Intent.QUIZ: "quiz_pipeline",
    Intent.PLAN: "planning_pipeline",
    Intent.RESEARCH: "research_pipeline",
    Intent.PAPER_ANALYSIS: "paper_analysis_pipeline",
}


class WorkflowPlan:
    """Ordered stages for one routed intent (metadata for the UI trace panel)."""

    __slots__ = ("intent", "stages")

    def __init__(self, intent: Intent, stages: Sequence[str]):
        self.intent = intent
        self.stages = list(stages)

    @property
    def name(self) -> str:
        return _WORKFLOW_NAMES.get(self.intent, f"{self.intent.value}_pipeline")

    def requires(self, stage: str) -> bool:
        return stage in self.stages

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "intent": self.intent.value, "stages": self.stages}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"WorkflowPlan(name={self.name!r}, stages={self.stages!r})"


def plan_workflow(decision: RouteDecision) -> WorkflowPlan:
    """Map a routing decision onto its execution stages."""
    return WorkflowPlan(decision.intent, _WORKFLOW_STAGES.get(decision.intent, ("reply",)))


def default_question_count(decision: RouteDecision) -> int:
    """Sensible question count when the learner did not state one."""
    if decision.question_count:
        return max(1, min(decision.question_count, 30))
    if decision.intent is Intent.EXAM:
        return 20
    if decision.difficulty == "expert" or decision.difficulty == "hard":
        return 8
    return 6


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
def _typed_count(text: str) -> int | None:
    match = _COUNT_RE.search(text.lower())
    return int(match.group(1)) if match else None


def typed_days(text: str) -> int | None:
    """Extract an exam horizon ('in 3 weeks' -> 21 days)."""
    lowered = text.lower()
    match = _DAYS_RE.search(lowered)
    if not match:
        return None
    amount = int(match.group(1))
    return amount * _WEEK if "week" in lowered[match.start() : match.end() + 4] else amount


def typed_hours_per_day(text: str) -> float | None:
    """Extract a daily study budget in **hours** ('3 hours a day' -> 3.0).

    ``StudyPlanInput.hours_per_day`` is expressed in hours, so minute figures are
    converted (``90 minutes`` -> ``1.5``) rather than passed through.
    """
    match = _MINUTES_RE.search(text.lower())
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(0)
    hours = amount if ("hour" in unit or "hr" in unit) else amount / 60.0
    return round(min(hours, 16.0), 2)


def _typed_level(lowered: str) -> ExplanationLevel | None:
    if "like i'm a beginner" in lowered or "like i am a beginner" in lowered or "beginner" in lowered:
        return "beginner"
    if "simple terms" in lowered or "layman" in lowered or "in plain english" in lowered:
        return "beginner"
    if "undergrad" in lowered or "undergraduate" in lowered:
        return "undergraduate"
    if "postgrad" in lowered or "postgraduate" in lowered or "graduate level" in lowered:
        return "graduate"
    if "expert" in lowered or "at a high level" in lowered or "rigorous" in lowered:
        return "expert"
    return None


def _typed_style(lowered: str) -> TeachingStyle | None:
    if "analogy" in lowered or "metaphor" in lowered:
        return "analogy"
    if "socratic" in lowered or "guide me" in lowered:
        return "socratic"
    if "short" in lowered or "brief" in lowered or "quick" in lowered:
        return "concise"
    if "example" in lowered:
        return "example_based"
    if "equation" in lowered or "mathematic" in lowered or "derivation" in lowered:
        return "mathematical"
    if "detailed" in lowered or "in depth" in lowered or "thorough" in lowered:
        return "detailed"
    return None


def _typed_difficulty(lowered: str) -> str | None:
    if "easy" in lowered or "beginner quiz" in lowered:
        return "easy"
    if "hard" in lowered or "difficult" in lowered:
        return "hard"
    if "expert" in lowered or "olympiad" in lowered or "tricky" in lowered:
        return "expert"
    if "medium" in lowered or "moderate" in lowered:
        return "medium"
    return None


#: Words that end a topic phrase: they introduce the *next* clause, not the topic.
_TOPIC_BREAKS = frozenset(
    {"with", "using", "please", "then", "quickly", "briefly", "afterwards", "step", "steps"}
)
#: Extractions too vague to be a usable retrieval filter.
_TOPIC_REJECT = frozenset(
    {"this", "that", "it", "them", "these", "those", "notes", "material", "everything",
     "my notes", "the notes", "my material", "the material", "all", "anything"}
)


#: Level/style boilerplate that precedes the real topic ("explain like I'm a
#: beginner how entropy works").
_TOPIC_LEAD_NOISE = re.compile(
    r"\b(?:like\s+(?:i\s*(?:am|'m)\s+)?(?:a\s+)?(?:beginner|high\s+school|undergrad\w*|expert)"
    r"|in\s+(?:simple|layman|plain)\s+(?:terms|english)"
    r"|as\s+a\s+(?:beginner|student)|for\s+a\s+beginner)\b"
)
#: Trailing function words left behind by truncation.
_TOPIC_TAIL_NOISE = re.compile(
    r"(?:\s+(?:in|on|of|for|and|with|to|a|an|the|is|are|how|why|what|that|it))*[\s]*$"
)


def _typed_topic(text: str) -> str | None:
    """Best-effort topic extraction.

    Tries the longest prepositional patterns first ('quiz me on X') and stops at
    the first clause break, so "quiz me on entropy with 5 hard questions" yields
    ``entropy`` rather than the whole sentence.
    """
    cleaned = _TOPIC_LEAD_NOISE.sub(" ", (text or "").lower())
    match = re.search(
        r"\b(?:quiz me on|test me on|teach me|flashcards? (?:on|about|for)|explain|"
        r"about|regarding|concerning|on|for|of)\s+(?:the\s+)?(?P<topic>.+)",
        cleaned,
    )
    if not match:
        return None
    kept: list[str] = []
    for word in re.split(r"[\s,]+", match.group("topic"))[:14]:
        bare = word.strip(".,;:?!'\"()")
        if not bare or bare.isdigit():
            break
        if kept and bare in _TOPIC_BREAKS:
            break
        kept.append(bare)
    topic = _TOPIC_TAIL_NOISE.sub("", " ".join(kept).strip()).strip()
    return None if topic in _TOPIC_REJECT or len(topic) < 3 else topic


def _requests_second_step(text: str) -> bool:
    return bool(
        re.search(r"\b(then|after that|afterwards|followed by|and then)\b", text, re.IGNORECASE)
    )


def _as_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_int(value: Any) -> int | None:
    """A plausible question/flashcard count from model output."""
    return coerce_int(value, minimum=1, maximum=200)


def _as_level(value: Any) -> ExplanationLevel | None:
    text = str(value or "").strip().lower()
    return text if text in ("beginner", "undergraduate", "graduate", "expert") else None


def _as_style(value: Any) -> TeachingStyle | None:
    text = str(value or "").strip().lower()
    valid = ("concise", "detailed", "socratic", "example_based", "mathematical", "analogy")
    return text if text in valid else None


def _as_difficulty(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in ("easy", "medium", "hard", "expert") else None


def _clamp(value: Any, low: float, high: float, *, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default
