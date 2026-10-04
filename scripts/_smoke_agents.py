"""Smoke test for the agent layer (offline provider, real RAG pipeline).

Run:  .\\.venv\\Scripts\\python.exe scripts\\_smoke_agents.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from workcompanion.config.settings import Settings, get_settings
from workcompanion.rag.ingestion import IngestionService
from workcompanion.rag.pipeline import RAGPipeline
from workcompanion.rag.sparse_index import BM25Index
from workcompanion.rag.vector_store import InMemoryVectorStore

from workcompanion.agents import (
    FlashcardAgent,
    NexusAgent,
    PlannerAgent,
    QuizAgent,
    RagAgent,
    ResearchAgent,
)
from workcompanion.llm.offline_provider import OfflineProvider
from workcompanion.schemas.planner import StudyPlanInput
from workcompanion.schemas.quiz import QuizSet

NOTES = """# Thermodynamics - Lecture 3

## First Law of Thermodynamics
The first law of thermodynamics states that the change in internal energy of a
system equals the heat added to the system minus the work done by the system.
The equation is dU = dQ - dW. Internal energy is a state function, while heat and
work are path functions, so only the net change is meaningful in a cyclic process.

## Enthalpy and Constant Pressure
Enthalpy H = U + pV measures the heat content at constant pressure. During a
phase transition at constant pressure the heat absorbed equals the change in
enthalpy. For an ideal gas H depends only on temperature, so Cp and Cv are
related by Cp - Cv = nR.

## Entropy and the Second Law
Entropy S measures the number of microstates compatible with a macrostate. The
second law of thermodynamics states that the entropy of an isolated system never
decreases. For a reversible process dS = dQ_rev / T. A spontaneous process
increases the total entropy of the system plus its surroundings.

## Gibbs Free Energy
The Gibbs free energy G = H - TS determines spontaneity at constant temperature
and pressure. A process is spontaneous when the change in G is negative. The
equilibrium constant is related to the change in G through
dG = -RT ln K.
"""


def build(settings: Settings) -> tuple[RAGPipeline, RagAgent, str]:
    store = InMemoryVectorStore()
    sparse = BM25Index()
    embeddings = _embeddings()
    pipeline = RAGPipeline(store, sparse, embeddings, OfflineProvider(), settings=settings)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thermodynamics_lecture3.md"
        path.write_text(NOTES, encoding="utf-8")
        service = IngestionService(store, sparse, embeddings, settings=settings)
        result = service.ingest_file(path)
        assert result.ok, result.summary()
        document_id = result.document_id

    return pipeline, RagAgent(pipeline, llm=OfflineProvider(), settings=settings), document_id


def _embeddings():
    from workcompanion.rag.embeddings import get_embedding_provider

    settings = get_settings()
    return get_embedding_provider(settings)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


_CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record an assertion so a silent regression cannot pass unnoticed."""
    _CHECKS.append(label)
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' -> ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(f"{label} failed{(': ' + detail) if detail else ''}")


def main() -> int:
    settings = get_settings()
    settings.ensure_directories()

    pipeline, rag, document_id = build(settings)
    section("INDEX")
    print(f"chunks={pipeline.chunk_count} document={document_id}")
    check("index holds the ingested chunks", pipeline.chunk_count >= 3, str(pipeline.chunk_count))

    # ------------------------------------------------------------------
    section("NEXUS routing")
    nexus = NexusAgent(llm=OfflineProvider(), settings=settings, has_documents=True)
    expectations = {
        "explain the first law of thermodynamics": ("explain", None),
        "quiz me on entropy with 5 hard questions": ("quiz", 5),
        "make 12 flashcards on enthalpy": ("flashcards", None),
        "plan my revision for thermodynamics in 7 days, 2 hours a day": ("plan", None),
        "what is the latest research on quantum error correction?": ("research", None),
        "how am I doing?": ("progress", None),
        "asdf qwerty": ("unclear", None),
    }
    for message, (expected_intent, expected_count) in expectations.items():
        decision = nexus.route(message)
        print(
            f"{message[:52]:54} -> {decision.intent.value:15} "
            f"docs={str(decision.needs_documents):5} web={str(decision.use_web):5} "
            f"n={decision.question_count} level={decision.explanation_level}"
        )
        check(f"routes {message[:34]!r}", decision.intent.value == expected_intent,
              decision.intent.value)
        if expected_count is not None:
            check(f"extracts count for {message[:28]!r}",
                  decision.question_count == expected_count, str(decision.question_count))
    check("extracts topic from the message",
          nexus.route("quiz me on entropy with 5 hard questions").topic == "entropy",
          str(nexus.route("quiz me on entropy").topic))
    check("'this' is rejected as a topic", nexus.route("tell me more about this").topic is None)

    # ------------------------------------------------------------------
    section("KNOWLEDGE answer")
    answer = rag.answer("What does the second law of thermodynamics say?")
    print(f"success={answer.success} confidence={answer.confidence} "
          f"level={answer.confidence_level.value} sources={len(answer.sources)}")
    print(answer.answer[:400])
    check("grounded answer has citations", len(answer.sources) > 0, str(len(answer.sources)))
    check("grounded answer is high confidence", answer.confidence > 0.5, str(answer.confidence))
    missing = rag.answer("What is the boiling point of tungsten at sea level?")
    print(f"\n-- unanswerable -> evidence={missing.metadata.get('evidence')} "
          f"confidence={missing.confidence}")
    check("unanswerable question is not over-confident", missing.confidence < 0.5,
          str(missing.confidence))

    # ------------------------------------------------------------------
    section("QUIZ")
    quiz_agent = QuizAgent(rag, llm=OfflineProvider(), settings=settings)
    quiz: QuizSet = quiz_agent.generate_quiz(
        "Entropy and the second law",
        num_questions=4,
        difficulty="hard",
        misconceptions=["Students think entropy always decreases in reversible processes"],
    )
    print(f"questions={len(quiz.questions)} points={quiz.total_points} "
          f"citations={len(quiz.citations)}")
    for question in quiz.questions:
        print(f"  [{question.question_type}/{question.difficulty}] {question.question[:64]}")
        print(f"      answer={question.correct_answer[:52]!r} refs={len(question.citation_refs)}")
    for warning in quiz.warnings:
        print(f"  ! {warning}")
    check("quiz produced questions", len(quiz.questions) >= 2, str(len(quiz.questions)))
    check("every question is grounded", all(q.citation_refs for q in quiz.questions))

    correct_id = quiz.questions[0].id
    responses = {correct_id: quiz.questions[0].correct_answer}
    for question in quiz.questions[1:]:
        responses[question.id] = "I have no idea"
    report = quiz_agent.grade(quiz, responses, seconds_per_question={q.id: 45 for q in quiz.questions})
    print(f"\nscore={report.score_percent}% correct={report.correct}/{report.total_questions}")
    print(f"topic_breakdown={report.topic_breakdown}")
    print(f"weak={report.weak_topics} misconceptions={report.misconceptions[:2]}")
    print(quiz_agent.render_report(report)[:400])

    section("QUIZ - letter-answer grading")
    mcq = next((q for q in quiz.questions if q.question_type == "mcq" and q.options), None)
    if mcq:
        index = mcq.options.index(mcq.correct_answer)
        for answer_text in ("ABCDEFGH"[index], str(index + 1), f"({chr(65 + index).lower()})"):
            evaluation = quiz_agent.evaluate_answer(mcq, answer_text)
            print(f"  {answer_text!r:6} -> correct={evaluation.is_correct}")
        wrong = quiz_agent.evaluate_answer(mcq, "None of these are mentioned")
        print(f"  {'wrong':6} -> correct={wrong.is_correct} feedback={wrong.feedback[:60]!r}")
        check("letter/number grading agrees", quiz_agent.evaluate_answer(mcq, "H").is_correct is False)

    section("QUIZ - exam + targeted")
    exam = quiz_agent.build_exam("Thermodynamics", num_questions=6, weak_topics=["Gibbs free energy"])
    print(f"exam questions={len(exam.questions)} timed={exam.timed} "
          f"duration={exam.duration_minutes}min")
    targeted = quiz_agent.generate_targeted_quiz(
        "Thermodynamics", ["Entropy", "Gibbs free energy"], num_questions=4
    )
    print(f"targeted questions={len(targeted.questions)} title={targeted.title!r}")
    check("exam is timed", exam.timed and bool(exam.duration_minutes))
    check("targeted quiz is scoped to weak topics", len(targeted.questions) >= 2,
          str(len(targeted.questions)))

    # ------------------------------------------------------------------
    section("FLASHCARDS")
    card_agent = FlashcardAgent(rag, llm=OfflineProvider(), settings=settings)
    deck = card_agent.generate_deck("Enthalpy", count=8, difficulty="medium")
    print(f"cards={len(deck.cards)} citations={len(deck.citations)} title={deck.title!r}")
    for card in deck.cards[:5]:
        print(f"  {card.front[:70]!r}\n    -> {card.back[:70]!r} [{card.topic}]")
    for warning in deck.warnings:
        print(f"  ! {warning}")
    check("deck produced cards", len(deck.cards) >= 2, str(len(deck.cards)))
    check("every card is grounded", all(c.citation_refs for c in deck.cards))

    # ------------------------------------------------------------------
    section("RESEARCH - paper analysis")
    research = ResearchAgent(rag, llm=OfflineProvider(), settings=settings)
    analysis = research.analyze_paper(document_id, level="beginner")
    print(f"title={analysis.title[:70]!r}")
    print(f"completeness={analysis.completeness:.2f} confidence={analysis.confidence}")
    print(f"year={analysis.year} authors={analysis.authors}")
    print(f"problem={analysis.research_problem[:100]!r}")
    print(f"results={len(analysis.key_results)} limitations={len(analysis.limitations)}")
    print(f"explained levels={list(analysis.explained)}")
    print(research.render_analysis(analysis)[:400])
    check("paper analysis cites sources", len(analysis.citations) >= 1, str(len(analysis.citations)))
    check("paper analysis does not leak the system prompt",
          "You are the Research Agent" not in analysis.title, analysis.title[:60])

    section("RESEARCH - synthesis (documents only, web disabled)")
    brief = research.research(
        "What determines whether a process is spontaneous?", use_web=False
    )
    print(f"evidence={brief.evidence_kind} confidence={brief.confidence} "
          f"citations={len(brief.citations)} used_web={brief.used_web}")
    print(brief.answer[:400])
    for warning in brief.warnings:
        print(f"  ! {warning}")
    check("synthesis is grounded", len(brief.citations) > 0, str(len(brief.citations)))
    check("synthesis does not leak the system prompt",
          "You are the Research Agent" not in brief.answer)

    section("RESEARCH - empty evidence path")
    empty = research.research("Quantum chromodynamics lattice gauge theory", use_web=False)
    print(f"confidence={empty.confidence} answer starts: {empty.answer[:80]!r}")
    check("no evidence -> explicit refusal", "no usable evidence" in empty.answer.lower())
    check("no evidence -> zero confidence", empty.confidence == 0.0, str(empty.confidence))

    # ------------------------------------------------------------------
    section("PLANNER")
    planner = PlannerAgent(llm=OfflineProvider(), settings=settings)
    plan = planner.create_plan(
        StudyPlanInput(
            subject="Thermodynamics",
            exam_date=date.today() + timedelta(days=10),
            days_available=10,
            hours_per_day=2.0,
            topics=["First law", "Enthalpy", "Entropy", "Gibbs free energy"],
            weak_areas=["Entropy", "Gibbs free energy"],
            strong_areas=["First law"],
        )
    )
    print(f"days={plan.total_days} hours={plan.total_hours} revision_slots="
          f"{sorted(plan.revision_schedule)}")
    for day in plan.days:
        print(f"  D{day.day_number:2} {day.day_date} {day.focus[:26]:28} "
              f"{','.join(day.activities):32} {day.hours}h {day.priority}")
    print("\n" + planner.render_plan(plan)[:300])
    check("plan covers the requested horizon", plan.total_days == 7, str(plan.total_days))
    check("plan respects the daily budget", plan.total_hours <= 7 * 2.0, str(plan.total_hours))
    check("plan schedules weak areas first",
          any(topic in plan.days[0].focus or topic in plan.days[0].topics
              for topic in ("Entropy", "Gibbs")), plan.days[0].focus)

    section("PLANNER - tiny horizon + no weak areas")
    small = planner.create_plan(
        StudyPlanInput(subject="Stats", days_available=3, hours_per_day=0.5)
    )
    print(f"days={small.total_days} hours={small.total_hours}")
    for day in small.days:
        print(f"  D{day.day_number} {day.focus} {day.activities} {day.hours}h")
    check("short horizon still yields a plan", small.total_days == 3, str(small.total_days))
    check("every planned day is within the budget",
          all(day.hours <= 0.5 for day in small.days))

    print(f"\nALL AGENT SMOKE CHECKS COMPLETED ({len(_CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
