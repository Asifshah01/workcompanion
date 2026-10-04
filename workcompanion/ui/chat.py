"""The conversational core: NEXUS routes, a specialist answers.

One dispatch path serves every chat page.  Keeping routing in one place is what
lets the Dashboard, the Tutor page and the Research page behave identically -
and it means the router's decisions are exercised no matter which page is open.

``handle_message`` returns an :class:`AgentOutcome` describing what happened, so
pages can render their own extras (a quiz, a plan, retrieved passages) without
re-running the agent.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from workcompanion.agents.nexus_agent import Intent, default_question_count, plan_workflow
from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.flashcards import FlashcardDeck
from workcompanion.schemas.planner import StudyPlan, StudyPlanInput
from workcompanion.schemas.quiz import QuizSet
from workcompanion.schemas.research import PaperAnalysis, ResearchBrief
from workcompanion.schemas.retrieval import RetrievalOutcome
from workcompanion.ui import state, theme

logger = get_logger(__name__)

#: The tutor speaks in a four-step scale; :class:`StudyPlanInput` in three.
PLAN_LEVELS_BY_LEVEL = {
    "beginner": "beginner",
    "undergraduate": "intermediate",
    "graduate": "advanced",
    "expert": "advanced",
}

#: Human label per intent, shown in the trace panel.
INTENT_LABELS = {
    Intent.EXPLAIN: "Teaching explanation",
    Intent.ANSWER: "Grounded answer",
    Intent.SUMMARIZE: "Summary",
    Intent.COMPARE: "Comparison",
    Intent.QUIZ: "Quiz",
    Intent.EXAM: "Timed exam",
    Intent.FLASHCARDS: "Flashcards",
    Intent.PLAN: "Study plan",
    Intent.PAPER_ANALYSIS: "Paper analysis",
    Intent.RESEARCH: "Research synthesis",
    Intent.SOCRATIC: "Socratic dialogue",
    Intent.SEARCH: "Passage search",
    Intent.PROGRESS: "Progress review",
    Intent.SMALL_TALK: "Small talk",
    Intent.UNCLEAR: "Clarification needed",
}


@dataclass
class AgentOutcome:
    """What one routed turn produced."""

    intent: Intent
    text: str
    result: AgentResult | None = None
    outcome: RetrievalOutcome | None = None
    quiz: QuizSet | None = None
    deck: FlashcardDeck | None = None
    plan: StudyPlan | None = None
    brief: ResearchBrief | None = None
    analysis: PaperAnalysis | None = None
    socratic: Any = None
    warnings: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def citations(self) -> list[Any]:
        """Citations from whichever deliverable this turn produced."""
        for source in (self.quiz, self.deck, self.brief, self.analysis):
            citations = getattr(source, "citations", None)
            if citations:
                return list(citations)
        if self.result is not None and self.result.sources:
            return list(self.result.sources)
        return list(self.outcome.citations) if self.outcome else []


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def handle_message(
    context: state.AppContext,
    message: str,
    *,
    topic: str | None = None,
    level: str | None = None,
    style: str | None = None,
    difficulty: str = "medium",
    count: int | None = None,
    use_web: bool | None = None,
    document_filter: str | None = None,
    days: int | None = None,
    hours_per_day: float | None = None,
    enable_crew: bool = False,
) -> AgentOutcome:
    """Route ``message`` to the right specialist and return what it produced."""
    if not message or not message.strip():
        return AgentOutcome(intent=Intent.UNCLEAR, text="Ask me anything about your material.")

    started = time.perf_counter()
    with context.learner() as (bundle, _repositories, memory):
        conversation = memory.context(context.conversation_id)
        history = conversation.as_pairs()
        learner_context = conversation.render() if conversation.turns else ""
        decision = bundle.nexus.route(
            message,
            history=history,
            has_documents=not bundle.pipeline.is_empty,
            learner_context=learner_context,
        )
        plan = plan_workflow(decision)
        stages = [plan.name, *plan.stages]
        context.log_action(
            "route",
            message=message[:80],
            intent=decision.intent.value,
            confidence=decision.confidence,
            stages=stages,
        )

        subject = topic or decision.topic
        # Handlers take ``(context, bundle, decision)`` positionally, so the
        # keyword bag must not repeat them.
        args: dict[str, Any] = {
            "message": message,
            "history": history,
            "subject": subject,
            "level": str(level or decision.explanation_level or context.settings.default_explanation_level),
            "style": str(style or decision.teaching_style or context.settings.default_teaching_style),
            "difficulty": str(decision.difficulty or difficulty),
            "count": int(count or decision.question_count or default_question_count(decision)),
            "use_web": decision.use_web if use_web is None else use_web,
            "document_filter": document_filter or decision.document_filter,
            "days": int(days or 7),
            "hours_per_day": float(hours_per_day or 2.0),
        }
        if enable_crew:
            args["crew"] = _try_crew(context, bundle, decision, message, subject)

        handler = _HANDLERS.get(decision.intent, _run_fallback)
        try:
            payload = handler(context, bundle, decision, **args)
        except Exception as exc:  # pragma: no cover - surfaced to the learner
            logger.exception("[ui] handler failed for intent %s", decision.intent)
            payload = _failure(context, decision, exc)

    payload.stages = stages
    payload.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if not payload.follow_ups:
        payload.follow_ups = list(decision.follow_ups) or default_follow_ups(
            decision.intent, subject
        )
    _push(context, decision, message, payload)
    return payload


def default_follow_ups(intent: Intent, topic: str | None) -> list[str]:
    """Suggestions shown under an answer when the router supplied none."""
    subject = topic or "this topic"
    table = {
        Intent.EXPLAIN: [
            f"Quiz me on {subject}",
            f"Make 10 flashcards for {subject}",
            f"Why does {subject} matter?",
        ],
        Intent.ANSWER: [
            f"Explain {subject} simply",
            f"What else does my material say about {subject}?",
            f"Test me on {subject}",
        ],
        Intent.QUIZ: ["Make it harder", "Explain why question 1 has that answer", "Turn this into flashcards"],
        Intent.FLASHCARDS: [f"Quiz me on {subject}", f"Plan revision for {subject}"],
        Intent.RESEARCH: [
            f"Summarise {subject} from my documents only",
            f"What is still unknown about {subject}?",
        ],
        Intent.PLAN: ["Quiz me on my weakest topic", "Halve the time in this plan"],
        Intent.SOCRATIC: ["Reveal the answer", "Give me a hint", "Explain it properly instead"],
        Intent.PAPER_ANALYSIS: ["What are the research gaps?", "Explain the methodology"],
    }
    return table.get(intent, [f"Explain {subject}", "What am I weak at?"])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _run_tutor(context, bundle, decision, **kwargs) -> AgentOutcome:
    from workcompanion.schemas.tutor import TutorRequest

    result = bundle.tutor.teach(
        TutorRequest(
            question=kwargs["message"],
            level=kwargs["level"],
            style=kwargs["style"],
            subject=kwargs["subject"],
        ),
        history=kwargs["history"],
        document_filter=kwargs["document_filter"],
    )
    return _from_result(result, decision)


def _run_rag(context, bundle, decision, **kwargs) -> AgentOutcome:
    result = bundle.rag.answer(
        kwargs["message"],
        history=kwargs["history"],
        style=None if kwargs["style"] == "socratic" else kwargs["style"],
        document_filter=kwargs["document_filter"],
    )
    return _from_result(result, decision)


def _run_summarize(context, bundle, decision, **kwargs) -> AgentOutcome:
    outcome = bundle.rag.retrieve(
        kwargs["message"],
        history=kwargs["history"],
        document_filter=kwargs["document_filter"],
    )
    result = bundle.rag.summarize(outcome, style="concise")
    payload = _from_result(result, decision)
    payload.outcome = outcome
    return payload


def _run_compare(context, bundle, decision, **kwargs) -> AgentOutcome:
    documents = _document_names(bundle, kwargs["document_filter"])
    result = bundle.rag.compare(kwargs["message"], documents or ["document A", "document B"])
    return _from_result(result, decision)


def _run_search(context, bundle, decision, **kwargs) -> AgentOutcome:
    outcome = bundle.rag.retrieve(
        kwargs["message"],
        history=kwargs["history"],
        document_filter=kwargs["document_filter"],
    )
    if not outcome.chunks:
        return AgentOutcome(
            intent=decision.intent,
            text=(
                "Nothing in your indexed material matched that. Try different keywords, "
                "or upload a document that covers the topic."
            ),
            outcome=outcome,
            warnings=list(outcome.warnings),
        )
    lines = [f"**{len(outcome.chunks)} passage(s) found**", ""]
    for index, chunk in enumerate(outcome.chunks, start=1):
        meta = chunk.metadata
        location = " · ".join(
            bit for bit in (meta.document_name, f"p. {meta.page}" if meta.page else "") if bit
        )
        lines += [f"**[{index}]** {location}", "", chunk.chunk.text.strip(), ""]
    from workcompanion.ui.components import sources_markdown

    lines.append(sources_markdown(outcome.citations))
    return AgentOutcome(
        intent=decision.intent,
        text="\n".join(lines).strip(),
        outcome=outcome,
        warnings=list(outcome.warnings),
    )


def _run_quiz(context, bundle, decision, **kwargs) -> AgentOutcome:
    topic = kwargs["subject"] or kwargs["message"]
    quiz = bundle.quiz.generate_quiz(
        topic,
        num_questions=kwargs["count"],
        difficulty=kwargs["difficulty"],
        document_filter=kwargs["document_filter"],
    )
    return AgentOutcome(
        intent=decision.intent,
        text=(
            f"Generated **{len(quiz.questions)} question(s)** on **{quiz.title}** "
            f"at {quiz.difficulty} level."
        ),
        quiz=quiz,
        warnings=list(quiz.warnings),
    )


def _run_exam(context, bundle, decision, **kwargs) -> AgentOutcome:
    topic = kwargs["subject"] or kwargs["message"]
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        weak = [
            stat.topic
            for stat in ProgressTracker(repositories, repositories.session).weak_topics()
        ]
    exam = bundle.quiz.build_exam(
        topic,
        num_questions=kwargs["count"],
        difficulty=kwargs["difficulty"],
        weak_topics=weak or None,
        duration_minutes=30,
    )
    if not weak:
        exam.warnings.append(
            "No quiz history yet, so the exam covers the topic evenly."
        )
    return AgentOutcome(
        intent=decision.intent,
        text=(
            f"Exam ready: **{len(exam.questions)} questions**, {exam.total_points} "
            f"points, {exam.duration_minutes or 30} minutes."
        ),
        quiz=exam,
        warnings=list(exam.warnings),
    )


def _run_flashcards(context, bundle, decision, **kwargs) -> AgentOutcome:
    topic = kwargs["subject"] or kwargs["message"]
    deck = bundle.flashcards.generate_deck(
        topic,
        count=kwargs["count"],
        difficulty=kwargs["difficulty"],
        document_filter=kwargs["document_filter"],
    )
    return AgentOutcome(
        intent=decision.intent,
        text=f"Built **{len(deck.cards)} card(s)** on **{deck.title}**.",
        deck=deck,
        warnings=list(deck.warnings),
    )


def _run_socratic(context, bundle, decision, **kwargs) -> AgentOutcome:
    """Drive the question-led dialogue; the UI owns the dialogue state."""
    session = dict(st.session_state.get(state.KEY_TURN) or {})
    question = session.get("question") or kwargs["message"]
    history = list(session.get("history") or kwargs["history"])
    level = int(session.get("scaffold_level", 0))

    if session:
        turn = bundle.socratic.next_turn(
            question,
            kwargs["message"],
            scaffold_level=level,
            history=history,
            force_reveal=_asks_for_answer(kwargs["message"]),
        )
        history.append(("user", kwargs["message"]))
    else:
        turn = bundle.socratic.start(
            kwargs["message"],
            history=kwargs["history"],
        )
        history.append(("user", kwargs["message"]))
        question = kwargs["message"]

    history.append(("assistant", turn.message))
    st.session_state[state.KEY_TURN] = {
        "question": question,
        "history": history[-8:],
        "scaffold_level": int(turn.scaffold_level),
    }
    if turn.reveal_answer:
        # The answer is out, so the dialogue is over: drop the state.
        st.session_state[state.KEY_TURN] = None

    result = bundle.socratic.result(
        turn.message,
        intent=decision.intent.value,
        grounding=[
            GroundingLabel.MODEL_REASONING,
            *([GroundingLabel.RETRIEVED_FACT] if not turn.reveal_answer else []),
        ],
        metadata={"scaffold_level": int(turn.scaffold_level), "revealed": bool(turn.reveal_answer)},
    )
    payload = _from_result(result, decision)
    payload.socratic = turn
    if turn.hint:
        theme.hint(f"**Hint:** {turn.hint}")
    if turn.misconceptions_detected:
        theme.warning("Possible misconception: " + ", ".join(turn.misconceptions_detected))
    return payload


def _run_plan(context, bundle, decision, **kwargs) -> AgentOutcome:
    with context.session() as repositories:
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repositories, repositories.session)
        weak = [stat.topic for stat in tracker.weak_topics()]
        strong = [stat.topic for stat in tracker.strong_topics()]
    plan = bundle.planner.create_plan(
        StudyPlanInput(
            subject=kwargs["subject"] or kwargs["message"][:60],
            days_available=kwargs["days"],
            hours_per_day=kwargs["hours_per_day"],
            current_level=PLAN_LEVELS_BY_LEVEL.get(kwargs["level"], "intermediate"),
            weak_areas=weak,
            strong_areas=strong,
            topics=[kwargs["subject"]] if kwargs["subject"] else [],
        )
    )
    text = (
        f"Built a **{plan.total_days}-day** plan ({plan.total_hours:.1f} h) for "
        f"**{plan.subject}**. Weak areas are scheduled first."
    )
    if plan.warnings:
        text += f"\n\n{plan.warnings[0]}"
    return AgentOutcome(intent=decision.intent, text=text, plan=plan, warnings=list(plan.warnings))


def _run_research(context, bundle, decision, **kwargs) -> AgentOutcome:
    brief = bundle.research.research(
        kwargs["message"],
        document_filter=kwargs["document_filter"],
        use_web=kwargs["use_web"],
    )
    return AgentOutcome(
        intent=decision.intent, text=brief.answer, brief=brief, warnings=list(brief.warnings)
    )


def _run_paper(context, bundle, decision, **kwargs) -> AgentOutcome:
    document_id = st.session_state.get("wc_active_document") or _pick_document(
        bundle, kwargs["message"]
    )
    if not document_id:
        return AgentOutcome(
            intent=decision.intent,
            text=(
                "I could not tell which document you mean. Choose it in **My Knowledge**, "
                "or upload the paper first."
            ),
        )
    analysis = bundle.research.analyze_paper(document_id, level=kwargs["level"])
    st.session_state["wc_active_document"] = document_id
    return AgentOutcome(
        intent=decision.intent,
        text=bundle.research.render_analysis(analysis),
        analysis=analysis,
        warnings=list(analysis.warnings),
    )


def _run_progress(context, bundle, decision, **kwargs) -> AgentOutcome:
    progress = context.progress_snapshot()
    profile = context.profile_snapshot()
    text = _render_progress(progress, profile)
    result = bundle.nexus.result(
        text,
        intent=decision.intent.value,
        confidence=1.0,
        grounding=[GroundingLabel.MODEL_REASONING],
    )
    return _from_result(result, decision)


def _run_small_talk(context, bundle, decision, **kwargs) -> AgentOutcome:
    return _from_result(bundle.nexus.run(kwargs["message"], history=kwargs["history"]), decision)


def _run_fallback(context, bundle, decision, **kwargs) -> AgentOutcome:
    """Unrecognised requests ask for clarification; everything else answers."""
    if decision.intent is Intent.UNCLEAR:
        text = bundle.nexus.clarify(decision)
        return AgentOutcome(
            intent=decision.intent,
            text=text,
            result=bundle.nexus.result(text, intent=decision.intent.value),
            follow_ups=list(decision.follow_ups)
            or [
                "Explain the first law of thermodynamics",
                "Quiz me on entropy",
                "Plan my revision for the next 7 days",
            ],
        )
    return _run_rag(context, bundle, decision, **kwargs)


_HANDLERS = {
    Intent.EXPLAIN: _run_tutor,
    Intent.ANSWER: _run_rag,
    Intent.SUMMARIZE: _run_summarize,
    Intent.COMPARE: _run_compare,
    Intent.SEARCH: _run_search,
    Intent.QUIZ: _run_quiz,
    Intent.EXAM: _run_exam,
    Intent.FLASHCARDS: _run_flashcards,
    Intent.SOCRATIC: _run_socratic,
    Intent.PLAN: _run_plan,
    Intent.RESEARCH: _run_research,
    Intent.PAPER_ANALYSIS: _run_paper,
    Intent.PROGRESS: _run_progress,
    Intent.SMALL_TALK: _run_small_talk,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _from_result(result: AgentResult, decision: Any) -> AgentOutcome:
    return AgentOutcome(
        intent=decision.intent,
        text=result.answer,
        result=result,
        warnings=[w for w in result.warnings if w],
    )


def _failure(context: state.AppContext, decision: Any, exc: Exception) -> AgentOutcome:
    detail = str(exc) if context.settings.debug else "an internal error"
    result = AgentResult(
        success=False,
        agent="nexus",
        intent=decision.intent.value,
        answer=(
            f"Something went wrong handling that request ({detail}). Nothing was saved. "
            "Try rephrasing the request, or check the logs."
        ),
        confidence=0.0,
        confidence_level="low",
        warnings=[detail] if context.settings.debug else [],
    )
    return AgentOutcome(intent=decision.intent, text=result.answer, result=result)


def _asks_for_answer(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        trigger in lowered
        for trigger in (
            "just tell me",
            "tell me the answer",
            "give me the answer",
            "i give up",
            "no idea",
            "reveal",
            "just answer",
        )
    )


def _document_names(bundle: Any, document_filter: str | None) -> list[str]:
    """Document names matching a filter, for the comparison agent."""
    names: list[str] = []
    try:
        from workcompanion.database.database import session_scope
        from workcompanion.database.repositories import get_repositories

        with session_scope(bundle.settings if hasattr(bundle, "settings") else None) as session:
            for record in get_repositories(session).documents.list_all(session, status="indexed"):
                if not document_filter or document_filter.lower() in (record.name or "").lower():
                    names.append(record.name or "")
    except Exception as exc:  # pragma: no cover - comparison is best-effort
        logger.info("[ui] could not list document names for comparison: %s", exc)
    return [name for name in names if name][:6]


def _pick_document(bundle: Any, message: str) -> str | None:
    """Best-effort document match for a paper-analysis request."""
    try:
        ids = list(bundle.pipeline.store.document_ids())
    except Exception:  # pragma: no cover
        return None
    if not ids:
        return None
    lowered = (message or "").lower()
    for document_id in ids:
        if document_id.lower() in lowered:
            return document_id
    return ids[0] if len(ids) == 1 else None


def _render_progress(progress: Any, profile: Any) -> str:
    """Markdown progress report (the Progress page shows the same numbers)."""
    lines = ["## Your progress", ""]
    stats = [
        ("Documents indexed", f"{progress.documents_indexed}/{progress.documents_total}"),
        ("Chunks in the index", f"{progress.total_chunks:,}"),
        ("Questions answered", f"{progress.questions_answered}"),
        ("Quiz accuracy", f"{progress.average_quiz_score:.0f}%" if progress.average_quiz_score else "-"),
        ("Overall mastery", f"{progress.overall_mastery:.0f}%"),
        ("Study time", f"{progress.study_minutes:.0f} min"),
        ("Sessions", f"{progress.sessions_count}"),
    ]
    lines += [f"- **{label}:** {value}" for label, value in stats]
    if progress.recommendation:
        lines += ["", f"**Next:** {progress.recommendation}"]
    if progress.recommendation_topics:
        lines.append("Focus areas: " + ", ".join(progress.recommendation_topics[:5]))
    weak = profile.weak_topics or []
    strong = profile.strong_topics or []
    if weak:
        lines += ["", "**Weak areas:** " + ", ".join(weak[:6])]
    if strong:
        lines.append("**Strong areas:** " + ", ".join(strong[:6]))
    if not progress.questions_answered:
        lines += [
            "",
            "_Nothing recorded yet. Take a quiz or ask a question and this page fills in._",
        ]
    return "\n".join(lines)


def _try_crew(
    context: state.AppContext,
    bundle: Any,
    decision: Any,
    message: str,
    subject: str | None,
) -> Any:
    """Optional CrewAI pass; ``None`` means "use the direct agent path"."""
    from workcompanion.crew import build_workflow, run_workflow

    workflow_name = {
        Intent.RESEARCH: "deep_research",
        Intent.EXAM: "exam_prep",
        Intent.EXPLAIN: "study_session",
    }.get(decision.intent)
    if workflow_name is None:
        return None
    try:
        plan = build_workflow(
            workflow_name,
            message,
            topic=subject,
            count=decision.question_count,
            level=decision.explanation_level,
        )
        return run_workflow(plan, bundle, context.settings)
    except Exception as exc:
        logger.warning("[ui] crew workflow failed (%s); using the direct agent path.", exc)
        return None


def _push(context: state.AppContext, decision: Any, message: str, payload: AgentOutcome) -> None:
    """Record the turn in the transcript and (once) in persistent memory."""
    state.push_turn("user", message, intent=decision.intent.value)
    state.push_turn(
        "assistant",
        payload.text,
        intent=decision.intent.value,
        agent=str(payload.result.agent) if payload.result else decision.intent.value,
        confidence=float(payload.result.confidence) if payload.result else None,
        sources=payload.citations,
    )
    state.record_turn(
        context,
        payload.result
        or AgentResult(
            success=True,
            agent=decision.intent.value,
            intent=decision.intent.value,
            answer=payload.text,
            confidence=0.0,
            confidence_level="low",
        ),
        question=message,
        mode=decision.intent.value,
        answer=payload.text,
        citations=payload.citations,
        warnings=payload.warnings,
        subject=payload.result.metadata.get("topic") if payload.result else None,
    )
    stages, notes = state.current_trace()
    plan = plan_workflow(decision)
    state.set_trace(
        [plan.name, *plan.stages],
        {
            plan.name: decision.rationale or f"routed as {decision.intent.value}",
            **{stage: "completed" for stage in plan.stages},
        },
    )


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
def render_outcome(context: state.AppContext, payload: AgentOutcome) -> None:
    """Render whatever the handler produced, with the right widget per intent."""
    from workcompanion.ui import components

    if payload.result is not None:
        components.render_answer(payload.result)
    else:
        st.markdown(payload.text or "_Nothing to show._")

    if payload.quiz is not None:
        components.render_quiz(payload.quiz, key_prefix=f"chat_{payload.intent.value}")
    if payload.deck is not None:
        components.render_deck(payload.deck, key_prefix=f"chat_deck_{payload.intent.value}")
    if payload.plan is not None:
        components.render_plan(payload.plan)
    if payload.intent is Intent.SEARCH and payload.outcome is not None:
        components.render_retrieval(payload.outcome, show_chunks=False)
    if payload.socratic is not None:
        turn = payload.socratic
        theme.badge_row(
            [
                (f"scaffold {turn.scaffold_level}/{5}", "neutral", "🪜"),
                ("answer revealed", "low", "🔓") if turn.reveal_answer else ("guiding", "neutral", "🔒"),
            ]
        )
    if payload.result is None and payload.warnings:
        components.render_warnings(payload.warnings)


def render_trace_panel(payload: AgentOutcome | None = None) -> None:
    """Show the routing decision and pipeline stages for the last request."""
    stages, notes = state.current_trace()
    if not stages:
        return
    with st.expander("Agent trace", expanded=False):
        theme.trace(stages, notes)
        if payload is not None:
            theme.hint(
                f"Intent: {payload.intent.value} · {payload.elapsed_ms:.0f} ms"
                + (f" · deliverable: {payload.intent.value}" if payload.quiz or payload.deck or payload.plan else "")
            )


def chat(
    context: state.AppContext,
    *,
    placeholder: str = "Ask anything about your material…",
    topic: str | None = None,
    level: str | None = None,
    style: str | None = None,
    difficulty: str = "medium",
    use_web: bool | None = None,
    enable_crew: bool = False,
    key: str = "chat",
) -> None:
    """A full chat surface: transcript, input, follow-ups and trace panel.

    A follow-up button queues the next message and reruns, so the learner types
    one word instead of repeating a long prompt.
    """
    from workcompanion.ui import components

    entries = state.history()
    if entries:
        components.render_transcript(entries)

    prompt = st.chat_input(placeholder, key=f"{key}_input")
    prompt = st.session_state.pop(f"{key}_queued", None) or prompt
    if not prompt:
        if st.session_state.get(f"{key}_quit"):
            st.session_state.pop(f"{key}_quit", None)
            state.clear_chat()
            st.rerun()
        render_trace_panel()
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    payload = handle_message(
        context,
        prompt,
        topic=topic,
        level=level,
        style=style,
        difficulty=difficulty,
        use_web=use_web,
        enable_crew=enable_crew,
    )
    with st.chat_message("assistant"):
        render_outcome(context, payload)
        chosen = components.follow_up_buttons(payload.follow_ups, key=f"{key}_fu")
        if chosen:
            st.session_state[f"{key}_queued"] = chosen
        if st.button("Clear conversation", key=f"{key}_clear"):
            st.session_state[f"{key}_quit"] = True
            st.rerun()
    render_trace_panel(payload)


__all__ = [
    "AgentOutcome",
    "INTENT_LABELS",
    "chat",
    "default_follow_ups",
    "handle_message",
    "render_outcome",
    "render_trace_panel",
]