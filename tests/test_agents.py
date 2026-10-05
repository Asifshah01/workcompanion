"""Agent behaviour, using the deterministic offline provider.

The offline provider is the point of these tests: every assertion must hold
with no network, no API key and no rate limit, so a green suite means the
*orchestration* is correct rather than the model happened to be helpful today.

Note that quiz and flashcard generation are grounded by design - they refuse
rather than inventing questions when nothing relevant is indexed. That is the
hallucination control working, so those tests run against ``seeded``.
"""

from __future__ import annotations

import pytest

from workcompanion.schemas.tutor import Intent
from workcompanion.schemas.common import GroundingLabel
from workcompanion.schemas import (
    QuizReport,
    QuizSet,
    RouteDecision,
    StudyPlan,
    StudyPlanInput,
    TutorRequest,
    TutorTurn,
)


def tutor_request(question: str, **kw) -> TutorRequest:
    return TutorRequest(question=question, **kw)


def plan_input(**kw) -> StudyPlanInput:
    payload = {
        "subject": "Physics",
        "days_available": 5,
        "hours_per_day": 1.5,
        "current_level": "beginner",
        "topics": ["Entropy", "Gibbs free energy"],
    }
    payload.update(kw)
    return StudyPlanInput(**payload)


# ---------------------------------------------------------------------------
# The shared AgentResult envelope
# ---------------------------------------------------------------------------
class TestAgentEnvelope:
    def test_agents_returning_an_envelope_share_its_shape(self, seeded):
        """Q&A-style agents all speak AgentResult, so the UI can render uniformly."""
        from workcompanion.schemas.common import AgentResult

        results = [
            seeded.tutor.teach(tutor_request("Explain entropy", subject="Physics")),
            seeded.rag.answer("What is the second law?"),
        ]

        for result in results:
            assert isinstance(
                result, AgentResult
            ), f"{type(result).__name__} is not an AgentResult"
            assert result.agent, "every result must name the agent that produced it"
            assert isinstance(result.warnings, list)
            assert isinstance(result.latency_ms, (int, float))
            assert isinstance(result.metadata, dict)
            assert 0.0 <= result.confidence <= 1.0

    def test_artifact_agents_return_typed_domain_objects(self, seeded):
        """Generation agents return the artifact itself, not an envelope.

        A StudyPlan is far more useful to the planner page than a string, and a
        QuizSet needs its questions intact to be gradeable - so these
        deliberately do not funnel through AgentResult.
        """
        from workcompanion.schemas import FlashcardDeck, QuizSet, StudyPlan

        assert isinstance(seeded.quiz.generate_quiz("Entropy", num_questions=2), QuizSet)
        assert isinstance(seeded.flashcards.generate_deck("Entropy", count=2), FlashcardDeck)
        assert isinstance(seeded.planner.create_plan(plan_input()), StudyPlan)

    def test_a_successful_answer_has_text(self, seeded):
        result = seeded.tutor.teach(tutor_request("Explain entropy", subject="Physics"))
        assert result.success
        assert result.answer.strip()

    def test_offline_results_are_labelled(self, bundle):
        """A user must never mistake offline filler for a real model answer."""
        result = bundle.tutor.teach(tutor_request("Explain entropy"))

        blob = f"{result.answer} {result.warnings} {result.metadata}".lower()
        assert "offline" in blob

    def test_confidence_is_always_in_range(self, bundle):
        for result in (
            bundle.tutor.teach(tutor_request("Explain entropy")),
            bundle.rag.answer("Explain entropy"),
        ):
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tutor
# ---------------------------------------------------------------------------
class TestTutor:
    def test_returns_a_teaching_turn(self, seeded):
        result = seeded.tutor.teach(
            tutor_request("What is entropy?", subject="Physics", level="beginner")
        )
        assert result.success
        assert result.answer.strip()

    @pytest.mark.parametrize("level", ["beginner", "undergraduate", "graduate", "expert"])
    def test_every_explanation_level_is_accepted(self, seeded, level: str):
        result = seeded.tutor.teach(
            tutor_request("What is entropy?", subject="Physics", level=level)
        )
        assert result.success
        assert result.answer.strip()

    def test_misconception_diagnosis_addresses_the_error(self, seeded):
        turn = seeded.tutor.diagnose_misconception(
            "How does entropy change?",
            "Entropy always decreases in every process.",
            expected="Entropy of an isolated system never decreases.",
        )
        assert isinstance(turn, TutorTurn)
        assert turn.message.strip()

    def test_understanding_check_replies(self, seeded):
        turn = seeded.tutor.check_understanding(
            "Entropy", "It is a measure of disorder.", expected="It measures disorder."
        )
        assert turn.message.strip()

    def test_understanding_check_can_reveal_the_answer(self, seeded):
        turn = seeded.tutor.check_understanding(
            "Entropy", "no idea", expected="A measure of disorder."
        )
        assert turn.message.strip()


# ---------------------------------------------------------------------------
# RAG / knowledge
# ---------------------------------------------------------------------------
class TestRagAgent:
    def test_grounded_answer_cites_the_source(self, seeded):
        result = seeded.rag.answer(
            "What is the Gibbs free energy criterion for spontaneity?"
        )
        assert result.success, result.warnings
        assert result.sources, "a grounded answer must carry sources"
        assert any("thermodynamics" in c.document_name for c in result.sources)

    def test_refuses_when_the_corpus_cannot_help(self, seeded):
        result = seeded.rag.answer(
            "Who won the 1998 FIFA World Cup final?",
            allow_general_knowledge=False,
        )
        assert not result.success
        assert not result.sources
        assert result.warnings

    def test_general_knowledge_is_opt_in(self, seeded):
        strict = seeded.rag.answer(
            "Who won the 1998 FIFA World Cup final?", allow_general_knowledge=False
        )
        relaxed = seeded.rag.answer(
            "Who won the 1998 FIFA World Cup final?", allow_general_knowledge=True
        )
        assert strict.success is False
        assert relaxed.success is True

    def test_general_knowledge_is_never_disguised_as_sourced(self, seeded):
        """The whole point: an unsourced answer must not look like it is sourced.

        This flag used to be threaded into the prompt but never consulted at the
        refusal gate, so `allow_general_knowledge=True` refused anyway - and the
        day that got "fixed" naively, the answer would have come back looking
        exactly like a grounded one. Both halves matter.
        """
        result = seeded.rag.answer(
            "Who won the 1998 FIFA World Cup final?", allow_general_knowledge=True
        )

        assert result.success, "the caller explicitly permitted an answer"
        assert not result.sources, "general knowledge must carry no citations"
        assert GroundingLabel.GENERAL_KNOWLEDGE in result.grounding
        assert result.confidence <= 0.35, (
            "an answer grounded in nothing the learner owns cannot be confident, "
            f"no matter how good it is (got {result.confidence})"
        )
        assert result.warnings, "the learner must be told the source is the model"

    def test_the_permission_changes_nothing_about_the_retrieval(self, seeded):
        """Permitting general knowledge must not weaken grounded answering."""
        grounded = seeded.rag.answer(
            "What is the Gibbs free energy criterion for spontaneity?",
            allow_general_knowledge=True,
        )
        assert grounded.sources, "a genuinely covered question is still cited"
        assert GroundingLabel.GENERAL_KNOWLEDGE not in grounded.grounding

    def test_empty_knowledge_base_still_answers_safely(self, bundle):
        result = bundle.rag.answer("What is entropy?")
        assert isinstance(result.warnings, list)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------
class TestQuiz:
    def test_generates_the_requested_number_of_questions(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=4)
        assert isinstance(quiz, QuizSet)
        assert len(quiz.questions) == 4

    def test_questions_have_unique_ids(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=5)
        ids = [q.id for q in quiz.questions]
        assert len(ids) == len(set(ids))

    def test_questions_carry_options_for_multiple_choice(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=4)
        mcqs = [q for q in quiz.questions if str(q.question_type) == "mcq"]
        assert mcqs, "at least one multiple-choice question"
        for question in mcqs:
            assert len(question.options) >= 2
            assert question.correct_answer.strip()

    def test_grading_returns_a_percentage(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=3)
        report = seeded.quiz.grade(
            quiz, {q.id: q.correct_answer for q in quiz.questions}
        )
        assert isinstance(report, QuizReport)
        assert 0.0 <= report.score_percent <= 100.0

    def test_wrong_answers_score_lower_than_right_ones(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=3)

        perfect = seeded.quiz.grade(
            quiz, {q.id: q.correct_answer for q in quiz.questions}
        )
        wrong = seeded.quiz.grade(
            quiz, {q.id: "definitely not it" for q in quiz.questions}
        )

        assert perfect.score_percent > wrong.score_percent

    def test_per_question_evaluations_are_returned(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=2)
        report = seeded.quiz.grade(
            quiz, {q.id: q.correct_answer for q in quiz.questions}
        )
        assert len(report.evaluations) == len(quiz.questions)

    def test_a_missing_answer_is_unanswered_not_correct(self, seeded):
        quiz = seeded.quiz.generate_quiz("Thermodynamics", num_questions=2)
        report = seeded.quiz.grade(quiz, {})
        assert report.score_percent == 0.0

    def test_refuses_to_invent_questions_without_evidence(self, bundle):
        """No material on the topic means no quiz, not a fabricated one."""
        quiz = bundle.quiz.generate_quiz("Mongolian throat singing", num_questions=4)
        assert quiz.questions == []
        assert quiz.warnings


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------
class TestFlashcards:
    def test_deck_respects_the_requested_size(self, seeded):
        deck = seeded.flashcards.generate_deck("Gibbs free energy", count=5)
        assert 0 < len(deck.cards) <= 5

    def test_cards_have_front_and_back(self, seeded):
        deck = seeded.flashcards.generate_deck("Gibbs free energy", count=3)
        for card in deck.cards:
            assert card.front.strip()
            assert card.back.strip()

    def test_ids_are_unique(self, seeded):
        deck = seeded.flashcards.generate_deck("Gibbs free energy", count=6)
        ids = [c.id for c in deck.cards]
        assert len(ids) == len(set(ids))

    def test_refuses_to_build_a_deck_without_evidence(self, bundle):
        deck = bundle.flashcards.generate_deck("Mongolian throat singing", count=5)
        assert deck.cards == []
        assert deck.warnings


# ---------------------------------------------------------------------------
# Socratic
# ---------------------------------------------------------------------------
class TestSocratic:
    def test_opening_turn_is_a_question_not_an_answer(self, seeded):
        opening = seeded.socratic.start("What is entropy?")

        assert isinstance(opening, TutorTurn)
        assert opening.message.strip()
        assert "?" in opening.message, "a Socratic tutor must pose a question"

    def test_conversation_advances(self, seeded):
        seeded.socratic.start("What is entropy?")
        second = seeded.socratic.next_turn(
            "What is entropy?", "I think entropy means disorder."
        )
        assert second.message.strip()

    def test_history_is_accepted_without_error(self, seeded):
        turn = seeded.socratic.start(
            "Why does ice melt?",
            history=[("user", "Why does ice melt?"), ("assistant", "Think about density.")],
        )
        assert turn.message.strip()

    def test_scaffold_level_survives_nonsense_from_the_model(self, bundle):
        """The offline provider echoes a range like '1-5'; that must not crash."""
        turn = bundle.socratic.start("What is entropy?")
        assert 0 <= turn.scaffold_level <= 3


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
class TestResearch:
    def test_offline_research_degrades_instead_of_raising(self, bundle):
        """Web research is disabled in tests; it must not blow up."""
        brief = bundle.research.research(
            "state of quantum error correction", use_web=False
        )
        assert brief is not None
        assert brief.answer.strip()

    def test_research_reports_whether_web_was_used(self, bundle):
        brief = bundle.research.research("quantum error correction", use_web=False)
        assert brief.used_web is False

    def test_renders_a_readable_report(self, bundle):
        from workcompanion.schemas import PaperAnalysis

        analysis = PaperAnalysis(
            document_id="doc1",
            document_name="decoherence.pdf",
            title="On decoherence",
            summary="A short summary.",
        )
        assert bundle.research.render_analysis(analysis).strip()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class TestPlanner:
    def test_plan_covers_the_requested_number_of_days(self, seeded):
        plan = seeded.planner.create_plan(
            plan_input(days_available=5, topics=["Entropy", "Enthalpy"])
        )
        assert isinstance(plan, StudyPlan)
        assert len(plan.days) == 5

    def test_plan_stays_inside_the_time_budget(self, seeded):
        plan = seeded.planner.create_plan(
            plan_input(days_available=4, hours_per_day=1.0)
        )
        for day in plan.days:
            assert day.hours <= 1.0 + 1e-6, f"day {day.day_number} overruns the budget"

    def test_total_hours_are_plausible(self, seeded):
        plan = seeded.planner.create_plan(
            plan_input(days_available=5, hours_per_day=1.5)
        )
        total = sum(day.hours for day in plan.days)
        assert 0 < total <= 5 * 1.5 + 1e-6

    def test_empty_topic_list_is_handled(self, seeded):
        plan = seeded.planner.create_plan(plan_input(topics=[]))
        assert isinstance(plan.days, list)

    def test_days_are_chronological(self, seeded):
        plan = seeded.planner.create_plan(plan_input(days_available=3))
        dates = [day.day_date for day in plan.days]
        assert dates == sorted(dates), "study days must be in chronological order"

    def test_renders_as_text(self, seeded):
        plan = seeded.planner.create_plan(plan_input())
        assert seeded.planner.render_plan(plan).strip()


# ---------------------------------------------------------------------------
# Nexus router
# ---------------------------------------------------------------------------
class TestNexusRouter:
    @pytest.mark.parametrize(
        ("message", "intent"),
        [
            ("Quiz me on thermodynamics", "quiz"),
            ("Make me flashcards about entropy", "flashcards"),
            ("Plan my study week for physics", "plan"),
            ("What is the second law?", Intent.EXPLAIN),
        ],
    )
    def test_routes_by_intent(self, bundle, message: str, intent: str):
        decision = bundle.nexus.route(message)
        assert isinstance(decision, RouteDecision)
        assert decision.intent == intent, f"{message!r} -> {decision.intent}"

    def test_every_decision_carries_a_rationale(self, bundle):
        assert bundle.nexus.route("Quiz me on thermodynamics").rationale

    def test_quiz_route_carries_its_parameters(self, bundle):
        decision = bundle.nexus.route("Quiz me on thermodynamics")
        assert decision.question_count and decision.question_count > 0

    def test_vague_input_does_not_raise(self, bundle):
        decision = bundle.nexus.route("tell me more")
        assert decision.intent
        assert 0.0 <= decision.confidence <= 1.0

    def test_survives_a_nonsense_message(self, bundle):
        decision = bundle.nexus.route("??? 123 !!! \x00")
        assert decision.intent