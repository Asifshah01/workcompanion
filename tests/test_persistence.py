"""Persistence and memory: repositories, learner profile, progress tracking.

These tests guard the thing that makes the app feel like a companion rather
than a search box - it has to remember. Every SQLAlchemy session is opened and
closed inside a context manager here, which is also the rule the UI follows.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def repos(settings):
    """A RepositoryBundle on a real (temporary) SQLite database."""
    from workcompanion.database.database import init_db, session_scope
    from workcompanion.database.repositories import RepositoryBundle

    init_db(settings)
    with session_scope(settings) as session:
        yield RepositoryBundle(session)


def parsed_document(document_id: str = "doc-1", name: str = "notes.pdf") -> object:
    from workcompanion.schemas import ParsedDocument, RawPage

    return ParsedDocument(
        document_id=document_id,
        document_name=name,
        source_type="pdf",
        pages=[RawPage(page_number=1, text="Entropy never decreases.")],
        full_text="Entropy never decreases.",
    )


def quiz_set(question_ids: list[str] = ("q1", "q2")):
    from workcompanion.schemas import QuizQuestion, QuizSet

    return QuizSet(
        id="quiz-1",
        title="Entropy",
        subject="Physics",
        questions=[
            QuizQuestion(
                id=qid,
                question=f"Question {qid}?",
                question_type="short_answer",
                correct_answer="Because entropy never decreases",
                topic="Entropy",
            )
            for qid in question_ids
        ],
    )


# ---------------------------------------------------------------------------
# Users and preferences
# ---------------------------------------------------------------------------
class TestUsers:
    def test_get_or_create_is_idempotent(self, repos):
        first = repos.users.get_or_create(repos.session)
        second = repos.users.get_or_create(repos.session)

        assert first.id == second.id, "a second call must not create a duplicate learner"

    def test_merge_preserves_unrelated_keys(self, repos):
        repos.users.merge_preferences(repos.session, {"subject": "Physics"})
        repos.users.merge_preferences(repos.session, {"level": 2})
        repos.session.commit()

        prefs = repos.users.get_or_create(repos.session).preferences
        assert prefs.get("subject") == "Physics"
        assert prefs.get("level") == 2


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class TestDocuments:
    def _register(self, repos, document_id="doc-1", name="notes.pdf", sha="a" * 64):
        document = parsed_document(document_id, name)
        return repos.documents.register(
            repos.session,
            document,
            sha256=sha,
            size_bytes=1024,
            subject="Physics",
        )

    def test_register_then_mark_indexed(self, repos):
        self._register(repos)
        repos.documents.mark_indexed(repos.session, "doc-1", chunk_count=12)
        repos.session.commit()

        document = repos.documents.get(repos.session, "doc-1")
        assert document is not None
        assert document.status == "indexed"
        assert document.chunk_count == 12

    def test_failure_is_recorded(self, repos):
        self._register(repos)
        repos.documents.mark_failed(repos.session, "doc-1", "scanned document")
        repos.session.commit()

        assert repos.documents.get(repos.session, "doc-1").status == "failed"

    def test_find_by_hash_detects_a_reupload(self, repos):
        self._register(repos, sha="b" * 64)
        repos.session.commit()

        assert repos.documents.find_by_hash(repos.session, "b" * 64, 1024) is not None
        assert repos.documents.find_by_hash(repos.session, "c" * 64, 1024) is None

    def test_subjects_are_listed_without_duplicates(self, repos):
        # Distinct content hashes: (sha256, size_bytes) is unique by design.
        self._register(repos, document_id="d1", name="a.pdf", sha="1" * 64)
        self._register(repos, document_id="d2", name="b.pdf", sha="2" * 64)
        repos.session.commit()

        assert repos.documents.subjects(repos.session).count("Physics") == 1

    def test_counts_track_indexed_and_total(self, repos):
        self._register(repos, document_id="d1", name="a.pdf", sha="1" * 64)
        self._register(repos, document_id="d2", name="b.pdf", sha="2" * 64)
        repos.documents.mark_indexed(repos.session, "d1", chunk_count=3)
        repos.session.commit()

        total, indexed = repos.documents.counts(repos.session)
        assert (total, indexed) == (2, 1)

    def test_identical_content_is_rejected_by_the_database(self, repos):
        """The unique (sha256, size) constraint is the last line of defence."""
        from sqlalchemy.exc import IntegrityError

        self._register(repos, document_id="d1", name="a.pdf", sha="9" * 64)
        repos.session.commit()

        with pytest.raises(IntegrityError):
            self._register(repos, document_id="d2", name="copy.pdf", sha="9" * 64)
            repos.session.commit()
        repos.session.rollback()

    def test_delete_removes_the_row(self, repos):
        self._register(repos)
        repos.session.commit()

        assert repos.documents.delete(repos.session, "doc-1") is True
        assert repos.documents.get(repos.session, "doc-1") is None


# ---------------------------------------------------------------------------
# Quiz attempts
# ---------------------------------------------------------------------------
class TestQuizzes:
    def test_attempt_lifecycle(self, repos):
        from workcompanion.schemas import AnswerEvaluation

        quiz = quiz_set(["q1", "q2"])
        attempt = repos.quizzes.start_attempt(repos.session, quiz)

        evaluations = [
            AnswerEvaluation(
                question_id=q.id,
                is_correct=True,
                score=1.0,
                max_score=1.0,
                feedback="correct",
                expected_answer=q.correct_answer,
                learner_answer=q.correct_answer,
                topic="Entropy",
            )
            for q in quiz.questions
        ]

        from workcompanion.schemas import QuizReport

        report = QuizReport(
            quiz_id=quiz.id,
            title=quiz.title,
            total_questions=2,
            attempted=2,
            correct=2,
            score_percent=100.0,
            points_earned=2.0,
            points_possible=2.0,
            evaluations=evaluations,
        )

        for question, evaluation in zip(quiz.questions, evaluations, strict=True):
            repos.quizzes.record_answer(
                repos.session, attempt, question, evaluation, seconds_taken=4.0
            )
        repos.quizzes.complete_attempt(repos.session, attempt, report)
        repos.session.commit()

        stats = repos.quizzes.stats(repos.session)
        assert stats.get("attempts", 0) >= 1

    def test_empty_history_does_not_divide_by_zero(self, repos):
        """A brand-new learner has no attempts; stats must not crash."""
        assert isinstance(repos.quizzes.stats(repos.session), dict)


# ---------------------------------------------------------------------------
# Topic progress and weak-area detection
# ---------------------------------------------------------------------------
class TestTopicProgress:
    @staticmethod
    def _answer(repos, topic: str, correct: bool, subject: str = "Physics") -> None:
        repos.topics.update(
            repos.session, topic, correct=correct, subject=subject, seconds=5.0
        )

    def test_accuracy_reflects_answers(self, repos):
        for correct in (True, True, False, False):
            self._answer(repos, "Entropy", correct)
        repos.session.commit()

        assert repos.topics.accuracy(repos.session, "Entropy", "Physics") == 50.0

    def test_weak_topics_ranks_lowest_mastery_first(self, repos):
        for topic, correct in (("Entropy", True), ("Gibbs", False), ("Enthalpy", False)):
            for _ in range(4):
                self._answer(repos, topic, correct)
        repos.session.commit()

        weak = repos.topics.weak_topics(repos.session, limit=2)
        assert weak, "there must be weak topics"
        assert weak[0].topic in {"Gibbs", "Enthalpy"}

    def test_no_data_returns_no_weak_topics(self, repos):
        assert repos.topics.weak_topics(repos.session) == []

    def test_accuracy_of_an_unseen_topic_is_zero(self, repos):
        assert repos.topics.accuracy(repos.session, "NeverStudied") == 0.0

    def test_subject_scopes_progress(self, repos):
        self._answer(repos, "Entropy", True, subject="Physics")
        self._answer(repos, "Entropy", False, subject="Chemistry")
        repos.session.commit()

        assert repos.topics.accuracy(repos.session, "Entropy", "Physics") == 100.0
        assert repos.topics.accuracy(repos.session, "Entropy", "Chemistry") == 0.0


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
class TestConversations:
    def test_turns_are_stored_in_order(self, repos):
        conv = repos.conversations.get_or_create(repos.session)
        repos.conversations.add_message(repos.session, conv, "user", "What is entropy?")
        repos.conversations.add_message(
            repos.session, conv, "assistant", "A measure of disorder."
        )
        repos.session.commit()

        turns = repos.conversations.recent_turns(repos.session, conv.id)
        assert [role for role, _ in turns] == ["user", "assistant"]

    def test_get_or_create_reuses_an_existing_thread(self, repos):
        first = repos.conversations.get_or_create(repos.session)
        again = repos.conversations.get_or_create(repos.session, conversation_id=first.id)
        assert first.id == again.id

    def test_clear_empties_the_thread(self, repos):
        conv = repos.conversations.get_or_create(repos.session)
        repos.conversations.add_message(repos.session, conv, "user", "hi")
        repos.session.commit()

        repos.conversations.clear(repos.session, conv.id)
        repos.session.commit()
        assert repos.conversations.messages(repos.session, conv.id) == []

    def test_summary_is_empty_until_something_writes_one(self, repos):
        """The repository stores the summary; the memory layer generates it."""
        conv = repos.conversations.get_or_create(repos.session)
        assert repos.conversations.summarize_if_needed(repos.session, conv.id) is None

        conv.summary = "So far: entropy, Gibbs free energy."
        repos.session.commit()

        assert "Gibbs" in repos.conversations.summarize_if_needed(repos.session, conv.id)

    def test_recent_turns_is_a_bounded_window(self, repos):
        """Only the tail of a thread belongs in a prompt - not all of it."""
        conv = repos.conversations.get_or_create(repos.session)
        for i in range(20):
            repos.conversations.add_message(
                repos.session,
                conv,
                "user",
                f"Question number {i} about entropy and thermodynamics.",
            )
        conv.summary = "Discussing entropy."
        repos.session.commit()

        turns = repos.conversations.recent_turns(repos.session, conv.id)
        assert 0 < len(turns) < 20, "the window must be bounded"
        assert turns[-1][1].endswith("number 19 about entropy and thermodynamics.")
        assert repos.conversations.summarize_if_needed(repos.session, conv.id)


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------
class TestFlashcards:
    @staticmethod
    def _deck(deck_id: str = "deck-1") -> object:
        from workcompanion.schemas import Flashcard, FlashcardDeck

        return FlashcardDeck(
            id=deck_id,
            title="Entropy",
            subject="Physics",
            cards=[
                Flashcard(id="c1", front="What is entropy?", back="Disorder."),
                Flashcard(id="c2", front="2nd law?", back="Entropy never decreases."),
            ],
        )

    def test_deck_and_cards_round_trip(self, repos):
        repos.flashcards.save_deck(repos.session, self._deck())
        repos.session.commit()

        assert repos.flashcards.total_cards(repos.session) == 2
        assert repos.flashcards.get_deck(repos.session, "deck-1") is not None
        assert len(repos.flashcards.cards(repos.session, "deck-1")) == 2

    def test_review_updates_state(self, repos):
        repos.flashcards.save_deck(repos.session, self._deck("deck-2"))
        repos.session.commit()

        repos.flashcards.review(repos.session, "c1", correct=True)
        repos.session.commit()

        cards = {c.id: c for c in repos.flashcards.cards(repos.session, "deck-2")}
        assert cards["c1"].times_reviewed >= 1
        assert cards["c1"].times_correct >= 1
        assert cards["c2"].times_reviewed == 0, "reviewing one card must not touch another"


# ---------------------------------------------------------------------------
# Learner profile
# ---------------------------------------------------------------------------
class TestLearnerProfile:
    def test_snapshot_of_a_new_learner_is_empty(self, repos):
        from workcompanion.memory.learner_profile import LearnerProfile

        assert LearnerProfile(repos).snapshot().is_empty()

    def test_preferences_appear_in_the_prompt_context(self, repos):
        from workcompanion.memory.learner_profile import LearnerProfile

        profile = LearnerProfile(repos)
        profile.set_preferences(subject="Physics", level="undergraduate")
        repos.session.commit()

        assert "Physics" in profile.snapshot().prompt_context()

    def test_misconceptions_are_recorded_and_cleared(self, repos):
        from workcompanion.memory.learner_profile import LearnerProfile

        profile = LearnerProfile(repos)
        profile.record_misconception("Entropy always decreases")
        repos.session.commit()
        assert profile.get_misconceptions()

        profile.clear_misconceptions()
        repos.session.commit()
        assert profile.get_misconceptions() == []


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------
class TestProgressTracker:
    def test_mastery_of_a_new_learner_is_zero(self, repos):
        from workcompanion.memory.progress_tracker import ProgressTracker

        assert ProgressTracker(repos, repos.session).overall_mastery() == 0.0

    def test_answers_move_mastery_and_count(self, repos):
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repos, repos.session)
        for correct in (True, True, False, False):
            tracker.record_answer(topic="Entropy", subject="Physics", correct=correct)
        repos.session.commit()

        snapshot = tracker.snapshot()
        assert 0.0 <= snapshot.overall_mastery <= 1.0
        assert snapshot.subjects == ["Physics"]

        entropy = next(t for t in snapshot.topics if t.topic == "Entropy")
        assert entropy.attempts == 4
        assert entropy.correct == 2

    def test_quiz_totals_are_counted_separately(self, repos):
        """Quiz attempts and topic drills are different things."""
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repos, repos.session)
        for correct in (True, True, False, False):
            tracker.record_answer(topic="Entropy", subject="Physics", correct=correct)
        repos.session.commit()

        snapshot = ProgressTracker(repos, repos.session).snapshot()
        assert snapshot.questions_answered == 0, "no quiz was taken"
        assert snapshot.quiz_attempts == 0

    def test_snapshot_survives_a_reload(self, repos):
        from workcompanion.memory.progress_tracker import ProgressTracker

        ProgressTracker(repos, repos.session).record_answer(
            topic="Entropy", subject="Physics", correct=True
        )
        repos.session.commit()

        # A fresh tracker over the same rows is what a page reload looks like.
        reloaded = ProgressTracker(repos, repos.session).snapshot()
        assert [t.attempts for t in reloaded.topics if t.topic == "Entropy"] == [1]

    def test_weak_topics_are_detected(self, repos):
        from workcompanion.memory.progress_tracker import ProgressTracker

        tracker = ProgressTracker(repos, repos.session)
        for correct in (False, False, False, True):
            tracker.record_answer(topic="Relativity", subject="Physics", correct=correct)
        repos.session.commit()

        weak = tracker.weak_topics()
        assert weak, "a topic answered wrongly three times is weak"
        assert weak[0].topic == "Relativity"

    def test_recommendation_is_always_produced(self, repos):
        from workcompanion.memory.progress_tracker import ProgressTracker

        recommendation = ProgressTracker(repos, repos.session).recommend()
        assert recommendation.headline
        assert isinstance(recommendation.topics, list)