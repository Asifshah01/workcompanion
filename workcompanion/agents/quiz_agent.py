"""Quiz / Assessment Agent.

Responsibilities:

* build grounded assessment items from the learner's own material,
* grade objective items deterministically and subjective items with an LLM judge
  (plus a dependency-free lexical fallback),
* aggregate an attempt into a :class:`QuizReport` that feeds the adaptive loop.

Every question carries ``citation_refs`` (the chunk ids it came from) and a
``misconception_targeted`` label, so weak areas can be tracked topic by topic.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.agents.rag_agent import RagAgent
from workcompanion.config.logging_config import get_logger
from workcompanion.rag.citations import citations_markdown
from workcompanion.schemas.common import AgentResult, Citation, GroundingLabel
from workcompanion.schemas.quiz import (
    AnswerEvaluation,
    Difficulty,
    QuestionType,
    QuizQuestion,
    QuizReport,
    QuizSet,
)
from workcompanion.schemas.retrieval import RetrievalOutcome
from workcompanion.utils.text_utils import extract_key_terms, sentences_from_text, similarity_ratio
from workcompanion.utils.token_utils import trim_to_tokens

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Quiz Agent inside WorkCompanion AI. You write assessment \
items that are strictly grounded in the learner's own study material.

ABSOLUTE RULES
1. Every question must be answerable from the CONTEXT block. Never test outside \
knowledge and never invent facts.
2. Attach the citation markers of the passages each question came from in \
"citation_refs" (e.g. ["[1]", "[3]"]). A question with no refs is not usable.
3. Distractors must be plausible to a learner who half-remembers the topic: use \
other concepts from the SAME material, never silly options like "None of the above".
4. The correct option must be unambiguous from the CONTEXT alone.
5. Write "explanation" as 1-2 sentences that state WHY the answer is right, \
quoting the supporting passage. "hint" must scaffold without giving the answer.
6. Never write a question whose answer is merely the absence of information.

PEDAGOGY
- Cover different topics/sections; do not ask five variations of one fact.
- Vary difficulty within the requested band and label each item honestly.
- Prefer "why" and "how" items over rote recall once the learner is competent.
- When a misconception is supplied, write an item that would reveal it."""

_QUESTION_TYPES: dict[QuestionType, str] = {
    "mcq": "single-answer multiple choice with exactly 4 distinct options",
    "true_false": "a true/false statement (options will be generated for you)",
    "short_answer": "one phrase or short sentence",
    "numerical": "a single number or quantity with units if relevant",
    "conceptual": "a 'why / how does it work' question needing an explanation",
    "long_answer": "a paragraph-length explanation",
    "viva": "an oral-exam style question",
}

_TRUE_FALSE = ("true", "false")
_LETTERS = "ABCDEFGH"
_ARTICLES_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TRAILING_RE = re.compile(r"[\s.,;:!?]+$")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalise_answer(text: str) -> str:
    """Lowercase, strip articles and punctuation for tolerant comparison."""
    value = _TRAILING_RE.sub("", (text or "").strip().lower())
    value = _ARTICLES_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,:;")


def _numeric_value(text: str) -> float | None:
    match = _NUMBER_RE.search((text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - regex guarantees a float
        return None


def _matches_any(learner: str, candidates: Sequence[str]) -> bool:
    """Tolerant answer matching: exact, numeric, containment or high similarity."""
    given = normalise_answer(learner)
    if not given:
        return False
    given_number = _numeric_value(given)

    for candidate in candidates:
        expected = normalise_answer(candidate)
        if not expected:
            continue
        if given == expected:
            return True
        if given_number is not None and _numeric_value(expected) is not None:
            if abs(given_number - float(_numeric_value(expected) or 0.0)) < 1e-6:
                return True
        if len(given) > 3 and (given in expected or expected in given):
            return True
        if similarity_ratio(given, expected) >= 0.86:
            return True
    return False


def _resolve_option_index(learner_answer: str, options: Sequence[str]) -> int | None:
    """Map 'B', '(b)', '2', 'option 3' onto a zero-based option index."""
    given = (learner_answer or "").strip().lower()
    if not given or not options:
        return None
    candidate = re.sub(r"^[\(\[]?([a-h])[\)\].:]?$", r"\1", given)

    digits = re.fullmatch(r"(?:option\s*)?(\d{1,2})", candidate)
    if digits:
        number = int(digits.group(1))
        # Out of range means "not an option reference" - fall through to text
        # matching rather than guessing or raising.
        return number - 1 if 1 <= number <= len(options) else None

    if re.fullmatch(r"[a-h]", candidate):
        index = _LETTERS.find(candidate)
        return index if 0 <= index < len(options) else None
    return None


def evidence_map(outcome: RetrievalOutcome) -> dict[str, str]:
    """Map a context marker such as ``[2]`` to the chunk id behind it."""
    return {citation.marker: (citation.chunk_id or "") for citation in outcome.citations}


# ---------------------------------------------------------------------------
class QuizAgent(AgentBase):
    """Grounded quiz generation and grading."""

    name = "quiz"
    description = "Builds grounded quizzes, grades answers and reports weak topics."

    def __init__(
        self,
        rag_agent: RagAgent,
        llm=None,
        settings=None,
        cache=None,
    ):
        super().__init__(llm, settings, cache)
        self._rag = rag_agent

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_quiz(
        self,
        topic: str,
        *,
        num_questions: int = 6,
        difficulty: Difficulty = "medium",
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        question_types: Sequence[QuestionType] | None = None,
        misconceptions: Sequence[str] | None = None,
        title: str | None = None,
        interactive: bool = True,
        timed: bool = False,
        duration_minutes: int | None = None,
        per_question_topics: Mapping[str, RetrievalOutcome] | None = None,
    ) -> QuizSet:
        """Generate a grounded quiz for one topic.

        Args:
            topic: the subject of the quiz (also used as the retrieval query).
            per_question_topics: optional pre-computed retrieval per sub-topic;
                when supplied the quiz draws on several focused retrievals
                instead of a single broad one (used by targeted/weak-area mode).
        """
        warnings: list[str] = []
        wanted = max(1, min(int(num_questions), 30))
        types = list(question_types or ["mcq", "short_answer", "conceptual"])
        retrievals: dict[str, RetrievalOutcome] = dict(per_question_topics or {})

        if per_question_topics:
            outcome = list(per_question_topics.values())[0]
            retrieval_note = (
                f"{len(per_question_topics)} focused retrievals were combined into this quiz."
            )
        else:
            outcome = self._rag.retrieve(
                topic,
                document_ids=document_ids,
                document_filter=document_filter,
                top_k=max(wanted + 4, 8),
            )
            retrievals = {topic: outcome}
            retrieval_note = ""

        if outcome is None or not outcome.chunks:
            return QuizSet(
                id=uuid.uuid4().hex[:12],
                title=title or f"Quiz: {topic}",
                subject=topic,
                difficulty=difficulty,
                questions=[],
                warnings=[
                    "No passages matching this topic were found in your material, so no "
                    "grounded quiz could be created. Upload relevant notes or widen the topic."
                ],
            ).recompute_points()

        warnings.extend(outcome.warnings)
        if retrieval_note:
            warnings.append(retrieval_note)

        blocks = self._evidence_blocks(retrievals)
        prompt = self._build_prompt(
            topic,
            blocks,
            wanted=wanted,
            difficulty=difficulty,
            types=types,
            misconceptions=list(misconceptions or []),
        )
        request = self.build_request("quiz", user=prompt, temperature=0.35, max_tokens=3000)

        questions: list[QuizQuestion] = []
        raw_questions: list[dict[str, Any]] = []
        try:
            payload = self.structured(request) or {}
            raw_questions = [q for q in (payload.get("questions") or []) if isinstance(q, dict)]
            quiz_title = str(payload.get("title") or "").strip()
        except Exception as exc:
            logger.warning("[quiz] generation failed, using extractive items: %s", exc)
            quiz_title = ""

        mapping = evidence_map(outcome)
        for extra in retrievals.values():
            mapping.update(evidence_map(extra))
        for spec in raw_questions:
            question = self._build_question(spec, mapping, difficulty, types, topic)
            if question is not None:
                questions.append(question)

        dropped = len(raw_questions) - len(questions)
        if dropped > 0:
            warnings.append(
                f"{dropped} generated item(s) were discarded because they were not "
                "answerable from your material."
            )

        if len(questions) < max(1, wanted // 2):
            filler = self._extractive_questions(
                blocks, needed=max(0, wanted - len(questions)), topic=topic, difficulty=difficulty
            )
            if filler:
                questions.extend(filler)
                warnings.append(
                    "Some items were completed extractively from your material because the "
                    "model could not produce enough usable questions."
                )

        if not questions:
            return QuizSet(
                id=uuid.uuid4().hex[:12],
                title=title or quiz_title or f"Quiz: {topic}",
                subject=topic,
                difficulty=difficulty,
                warnings=warnings
                + ["The model did not return a usable question set for this topic."],
            ).recompute_points()

        if len(questions) > wanted:
            questions = questions[:wanted]

        used_markers = {ref for q in questions for ref in q.citation_refs}
        citations = [c for c in outcome.citations if c.chunk_id in used_markers]
        if not citations:
            citations = outcome.citations[:5]

        return QuizSet(
            id=uuid.uuid4().hex[:12],
            title=title or quiz_title or f"Quiz: {topic}",
            subject=topic,
            difficulty=difficulty,
            questions=questions,
            citations=citations,
            interactive=interactive,
            timed=timed,
            duration_minutes=duration_minutes
            if duration_minutes is not None
            else (max(5, wanted * 2) if timed else None),
            warnings=warnings,
        ).recompute_points()

    # ------------------------------------------------------------------
    def build_exam(
        self,
        subject: str,
        *,
        num_questions: int = 20,
        difficulty: Difficulty = "hard",
        document_ids: list[str] | None = None,
        weak_topics: Sequence[str] | None = None,
        duration_minutes: int | None = None,
    ) -> QuizSet:
        """Exam-mode paper: timed, harder, with extra weight on weak topics."""
        warnings: list[str] = []
        weak = [t for t in (weak_topics or []) if t.strip()]
        questions: list[QuizQuestion] = []
        citations = []

        # Weak areas first, then broad coverage of the subject.
        if weak:
            weak_set = self.generate_quiz(
                f"{subject}: {', '.join(weak[:3])}",
                num_questions=max(2, num_questions // 3),
                difficulty=difficulty,
                document_ids=document_ids,
            )
            if weak_set.questions:
                questions.extend(weak_set.questions)
                citations.extend(weak_set.citations)
                warnings.extend(weak_set.warnings)

        remaining = max(0, num_questions - len(questions))
        if remaining:
            main = self.generate_quiz(
                subject,
                num_questions=remaining,
                difficulty=difficulty,
                document_ids=document_ids,
                question_types=["mcq", "numerical", "conceptual", "long_answer"],
                misconceptions=weak,
            )
            questions.extend(main.questions)
            citations.extend(main.citations)
            warnings.extend(main.warnings)

        # Drop near-duplicate items that the weak-area and broad passes overlapped on.
        seen_questions: set[str] = set()
        unique_questions: list[QuizQuestion] = []
        for question in questions:
            key = normalise_answer(question.question)[:90]
            if key and key in seen_questions:
                continue
            seen_questions.add(key)
            unique_questions.append(question)
        questions = unique_questions[:num_questions]

        if not questions:
            return QuizSet(
                id=uuid.uuid4().hex[:12],
                title=f"Exam: {subject}",
                subject=subject,
                difficulty=difficulty,
                warnings=warnings
                + ["Not enough grounded material to build an exam paper. Add more documents."],
            ).recompute_points()

        deduped: list[Citation] = []
        seen_markers = set()
        for citation in citations:
            key = (citation.chunk_id, citation.marker)
            if key not in seen_markers:
                seen_markers.add(key)
                deduped.append(citation)

        return QuizSet(
            id=uuid.uuid4().hex[:12],
            title=f"Exam: {subject}",
            subject=subject,
            difficulty=difficulty,
            questions=questions,
            citations=deduped,
            interactive=True,
            timed=True,
            duration_minutes=duration_minutes or max(15, len(questions) * 2),
            warnings=warnings,
        ).recompute_points()

    # ------------------------------------------------------------------
    def generate_targeted_quiz(
        self,
        subject: str,
        topics: Sequence[str],
        *,
        num_questions: int = 8,
        difficulty: Difficulty = "medium",
        document_ids: list[str] | None = None,
        misconceptions: Sequence[str] | None = None,
    ) -> QuizSet:
        """Weak-area drill: one focused retrieval per topic, merged into a quiz."""
        wanted = max(1, num_questions)
        topic_list = [t.strip() for t in topics if t.strip()][:6] or [subject]
        per_topic: dict[str, RetrievalOutcome] = {}
        for topic in topic_list:
            outcome = self._rag.retrieve(
                f"{subject}: {topic}",
                document_ids=document_ids,
                top_k=max(4, wanted // max(1, len(topic_list)) + 3),
            )
            if outcome.chunks:
                per_topic[topic] = outcome

        if not per_topic:
            return QuizSet(
                id=uuid.uuid4().hex[:12],
                title=f"Targeted drill: {', '.join(topic_list[:3])}",
                subject=subject,
                difficulty=difficulty,
                warnings=[
                    "None of these weak topics could be matched to your indexed material."
                ],
            ).recompute_points()

        return self.generate_quiz(
            subject,
            num_questions=wanted,
            difficulty=difficulty,
            document_ids=document_ids,
            misconceptions=misconceptions,
            title=f"Targeted drill: {', '.join(topic_list[:3])}",
            per_question_topics=per_topic,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        topic: str,
        *,
        num_questions: int = 6,
        difficulty: Difficulty = "medium",
        document_ids: list[str] | None = None,
        document_filter: str | None = None,
        question_types: Sequence[QuestionType] | None = None,
        misconceptions: Sequence[str] | None = None,
        timed: bool = False,
    ) -> AgentResult:
        """Envelope-returning entry point (chat / crew workflows use this)."""
        quiz = self.generate_quiz(
            topic,
            num_questions=num_questions,
            difficulty=difficulty,
            document_ids=document_ids,
            document_filter=document_filter,
            question_types=question_types,
            misconceptions=misconceptions,
            timed=timed,
        )
        if not quiz.questions:
            return AgentResult(
                success=False,
                agent=self.name,
                intent="quiz",
                answer="\n\n".join(quiz.warnings)
                or "I could not build a grounded quiz from your material.",
                confidence=0.0,
                metadata={"evidence": "insufficient", "quiz_id": quiz.id},
            )

        markdown = self.render_quiz(quiz, reveal=False)
        sources = citations_markdown(quiz.citations)
        return self.result(
            markdown + sources,
            intent="quiz",
            sources=quiz.citations,
            confidence=self._quiz_confidence(quiz),
            grounding=[GroundingLabel.RETRIEVED_FACT, GroundingLabel.MODEL_REASONING],
            warnings=quiz.warnings,
            metadata={
                "evidence": "documents",
                "quiz_id": quiz.id,
                "title": quiz.title,
                "question_count": len(quiz.questions),
                "total_points": quiz.total_points,
                "difficulty": quiz.difficulty,
                "timed": quiz.timed,
                "duration_minutes": quiz.duration_minutes,
                "offline": self.is_offline,
            },
        )

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    def evaluate_answer(
        self,
        question: QuizQuestion,
        learner_answer: str,
        *,
        seconds: float = 0.0,
    ) -> AnswerEvaluation:
        """Grade one response, choosing a deterministic or model-based path."""
        learner_answer = (learner_answer or "").strip()

        if not learner_answer:
            return AnswerEvaluation(
                question_id=question.id,
                is_correct=False,
                score=0.0,
                max_score=question.points,
                feedback="No answer was given. Try to commit to an answer - guessing is free.",
                expected_answer=question.correct_answer,
                learner_answer="",
                topic=question.topic,
                next_hint=question.hint or "Re-read the source passage before retrying.",
            )

        if question.question_type in ("mcq", "true_false"):
            return self._grade_objective(question, learner_answer, seconds=seconds)
        return self._grade_subjective(question, learner_answer, seconds=seconds)

    def _grade_objective(
        self, question: QuizQuestion, learner_answer: str, *, seconds: float
    ) -> AnswerEvaluation:
        options = question.options
        index = _resolve_option_index(learner_answer, options)
        given = options[index] if index is not None else learner_answer

        candidates = question.acceptable()
        # Accept the option TEXT as well as the stored answer string.
        if index is not None and question.correct_answer:
            candidates = [*candidates, learner_answer]
        if question.correct_answer.isdigit() and options:
            numeric_index = int(question.correct_answer)
            if 1 <= numeric_index <= len(options):
                candidates.append(options[numeric_index - 1])

        is_correct = _matches_any(given, candidates)
        return AnswerEvaluation(
            question_id=question.id,
            is_correct=is_correct,
            score=question.points if is_correct else 0.0,
            max_score=question.points,
            feedback=(
                f"Correct. {question.explanation}".strip()
                if is_correct
                else f"Not quite. {question.explanation or f'The answer is: {question.correct_answer}'}".strip()
            ),
            misconception=None if is_correct else (question.misconception_targeted or "Key detail missed"),
            expected_answer=question.correct_answer,
            learner_answer=learner_answer,
            topic=question.topic,
            next_hint=None if is_correct else (question.hint or None),
        )

    def _grade_subjective(
        self, question: QuizQuestion, learner_answer: str, *, seconds: float
    ) -> AnswerEvaluation:
        prompt = (
            f"QUESTION:\n{question.question}\n\n"
            f"EXPECTED ANSWER:\n{question.correct_answer}\n\n"
            f"ACCEPTED ALTERNATIVES:\n{', '.join(question.accepted_answers) or '(none)'}\n\n"
            f"CONTEXT (the material the question came from):\n"
            f"{trim_to_tokens(question.explanation or question.correct_answer, 1200)}\n\n"
            f"LEARNER ANSWER:\n{learner_answer}\n\n"
            "Grade this answer as a fair but rigorous marker:\n"
            "1. Judge substance, not wording; allow equivalent phrasing.\n"
            "2. Partial credit is allowed - score between 0 and 1.\n"
            "3. Name the specific misconception if one is present, otherwise null.\n"
            "4. Give feedback the learner can act on, then one follow-up question.\n\n"
            'Return STRICT JSON: {"is_correct": bool, "score": 0.0-1.0, "feedback": "...", '
            '"misconception": "..." | null, "severity": "none|minor|major", '
            '"corrected_explanation": "...", "follow_up_question": "..." | null}'
        )
        request = self.build_request("evaluate_answer", user=prompt, temperature=0.15, max_tokens=900)
        payload: dict[str, Any] = {}
        try:
            payload = self.structured(request) or {}
        except Exception as exc:
            logger.warning("[quiz] subjective grading fell back to lexical overlap: %s", exc)

        if not payload:
            payload = self._lexical_grade(question, learner_answer)

        score = _clamp(payload.get("score"), 0.0, 1.0)
        if score > 0.55:
            payload.setdefault("is_correct", True)
        is_correct = bool(payload.get("is_correct")) or score >= 0.7
        severity = str(payload.get("severity") or ("none" if is_correct else "minor"))
        if severity not in ("none", "minor", "major"):
            severity = "minor"

        return AnswerEvaluation(
            question_id=question.id,
            is_correct=is_correct,
            score=round(score * question.points, 3),
            max_score=question.points,
            feedback=str(payload.get("feedback") or "").strip()
            or ("Correct." if is_correct else "Not quite - review the source passage."),
            misconception=str(payload["misconception"]).strip()
            if payload.get("misconception")
            else None,
            expected_answer=question.correct_answer,
            learner_answer=learner_answer,
            topic=question.topic,
            next_hint=str(payload["follow_up_question"]).strip()
            if payload.get("follow_up_question")
            else (question.hint or None),
        )

    def _lexical_grade(self, question: QuizQuestion, learner_answer: str) -> dict[str, Any]:
        """Deterministic partial-credit grading (no LLM required)."""
        expected_terms = set(extract_key_terms(question.correct_answer, limit=12))
        given_terms = set(extract_key_terms(learner_answer, limit=12))
        if not expected_terms:
            score = 0.0
        else:
            overlap = len(expected_terms & given_terms) / len(expected_terms)
            score = round(min(1.0, overlap), 3)
        is_correct = score >= 0.7
        missing = sorted(expected_terms - given_terms)[:5]
        return {
            "is_correct": is_correct,
            "score": score,
            "feedback": (
                "Your answer covers the key points in the material."
                if is_correct
                else "Key ideas from the source passage are missing: "
                + (", ".join(missing) if missing else "restate the definition precisely.")
            ),
            "misconception": None if is_correct else "Key definition not recalled accurately",
            "severity": "none" if is_correct else "minor",
            "corrected_explanation": question.correct_answer,
            "follow_up_question": None
            if is_correct
            else "Can you restate the key idea in your own words?",
        }

    # ------------------------------------------------------------------
    def grade(
        self,
        quiz: QuizSet,
        responses: Mapping[str, str],
        *,
        seconds_per_question: Mapping[str, float] | None = None,
        duration_seconds: float | None = None,
    ) -> QuizReport:
        """Grade a whole attempt and build the aggregate report.

        Args:
            responses: ``{question_id: learner answer}``; missing ids count as
                unanswered rather than silently vanishing from the statistics.
        """
        evaluations: list[AnswerEvaluation] = []
        for question in quiz.questions:
            learner_answer = responses.get(question.id, "")
            evaluations.append(
                self.evaluate_answer(
                    question,
                    learner_answer,
                    seconds=float((seconds_per_question or {}).get(question.id, 0.0) or 0.0),
                )
            )

        attempted = [e for e, q in zip(evaluations, quiz.questions, strict=False) if e.learner_answer]
        correct = [e for e in attempted if e.is_correct]
        earned = sum(e.score for e in evaluations)
        possible = sum(e.max_score for e in evaluations) or 1.0

        breakdown: dict[str, list[float]] = {}
        for evaluation, question in zip(evaluations, quiz.questions, strict=False):
            bucket = breakdown.setdefault(question.topic or "General", [])
            bucket.append(1.0 if evaluation.is_correct else 0.0)
        topic_breakdown = {
            topic: round(100.0 * sum(scores) / len(scores), 2) for topic, scores in breakdown.items()
        }

        weak = sorted(topic_breakdown.items(), key=lambda item: item[1])[:3]
        strong = sorted(topic_breakdown.items(), key=lambda item: -item[1])[:3]
        misconceptions: list[str] = []
        for evaluation, question in zip(evaluations, quiz.questions, strict=False):
            label = evaluation.misconception or question.misconception_targeted
            if label and not evaluation.is_correct and label not in misconceptions:
                misconceptions.append(label)

        report = QuizReport(
            quiz_id=quiz.id,
            title=quiz.title,
            total_questions=len(quiz.questions),
            attempted=len(attempted),
            correct=len(correct),
            score_percent=round(100.0 * earned / possible, 2),
            points_earned=round(earned, 3),
            points_possible=round(possible, 3),
            duration_seconds=duration_seconds,
            evaluations=evaluations,
            topic_breakdown=topic_breakdown,
            weak_topics=[topic for topic, value in weak if value < 70],
            strong_topics=[topic for topic, value in strong if value >= 80],
            misconceptions=misconceptions,
            recommendations=self._recommendations(topic_breakdown, misconceptions),
            study_minutes_logged=round(sum((seconds_per_question or {}).values()) / 60.0, 2),
        )
        self.trace(
            f"graded {quiz.id}",
            attempted=report.attempted,
            correct=report.correct,
            score=report.score_percent,
        )
        return report

    @staticmethod
    def _recommendations(
        topic_breakdown: Mapping[str, float], misconceptions: Sequence[str]
    ) -> list[str]:
        recommendations: list[str] = []
        weak = [topic for topic, value in sorted(topic_breakdown.items(), key=lambda i: i[1]) if value < 60]
        if weak:
            recommendations.append(
                f"Re-read the material on {', '.join(weak[:2])} and retake a targeted drill on "
                "just those topics."
            )
        if misconceptions:
            recommendations.append(
                "Ask the Tutor agent to explain why "
                + f"'{misconceptions[0]}' is a common mistake - it will contrast the two ideas."
            )
        if not recommendations:
            recommendations.append(
                "Accuracy is solid here. Move to harder items or a timed exam-mode paper."
            )
        recommendations.append("Use the Flashcards page to lock these topics into long-term memory.")
        return recommendations

    def render_report(self, report: QuizReport) -> str:
        """Markdown summary of an attempt (shown in the chat transcript)."""
        lines = [
            f"### Result: {report.score_percent:.1f}%",
            "",
            f"- **Answered** {report.attempted}/{report.total_questions}",
            f"- **Correct** {report.correct}",
            f"- **Points** {report.points_earned:g}/{report.points_possible:g}",
        ]
        if report.duration_seconds:
            lines.append(f"- **Time** {report.duration_seconds / 60:.1f} min")
        if report.topic_breakdown:
            lines += ["", "**By topic**", ""]
            lines += [
                f"- {topic}: {value:.0f}%" for topic, value in sorted(report.topic_breakdown.items())
            ]
        if report.misconceptions:
            lines += ["", "**Misconceptions spotted**", ""]
            lines += [f"- {item}" for item in report.misconceptions]
        if report.recommendations:
            lines += ["", "**What to do next**", ""]
            lines += [f"{i}. {text}" for i, text in enumerate(report.recommendations, start=1)]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rendering / prompting helpers
    # ------------------------------------------------------------------
    def render_quiz(self, quiz: QuizSet, *, reveal: bool = False) -> str:
        """Markdown rendering used by the chat view and for export."""
        header = [f"### {quiz.title}", ""]
        meta = [f"Difficulty **{quiz.difficulty}**"]
        if quiz.total_points:
            meta.append(f"{quiz.total_points:g} points")
        if quiz.timed and quiz.duration_minutes:
            meta.append(f"{quiz.duration_minutes} min limit")
        header.append(" \u00b7 ".join(meta))
        header.append("")

        lines = list(header)
        for index, question in enumerate(quiz.questions, start=1):
            lines.append(f"**Q{index}.** {question.question}")
            if question.question_type in ("mcq", "true_false") and question.options:
                lines += [
                    f"   - {letter}) {option}" for letter, option in zip(_LETTERS, question.options)
                ]
            if question.hint:
                lines.append(f"   *Hint:* {question.hint}")
            if reveal:
                lines.append(f"   **Answer:** {question.correct_answer}")
                if question.explanation:
                    lines.append(f"   *Why:* {question.explanation}")
                if question.citation_refs:
                    lines.append(f"   *Source chunks:* {', '.join(question.citation_refs)}")
            lines.append("")
        for warning in quiz.warnings:
            lines.append(f"> {warning}")
        return "\n".join(lines).strip()

    def _evidence_blocks(self, retrievals: Mapping[str, RetrievalOutcome]) -> str:
        """Render one labelled evidence block per retrieval, with stable markers."""
        blocks: list[str] = []
        for label, outcome in retrievals.items():
            if not outcome.chunks:
                continue
            body = trim_to_tokens(outcome.context, 6000)
            blocks.append(f"### {label}\n{body}")
        return "\n\n".join(blocks) or "CONTEXT:\n(nothing relevant was retrieved)"

    def _build_prompt(
        self,
        topic: str,
        blocks: str,
        *,
        wanted: int,
        difficulty: Difficulty,
        types: Sequence[QuestionType],
        misconceptions: Sequence[str],
    ) -> str:
        type_lines = "\n".join(
            f"   - {name}: {_QUESTION_TYPES[name]}" for name in types if name in _QUESTION_TYPES
        )
        misconception_block = (
            "KNOWN MISCONCEPTIONS TO TEST FOR:\n"
            + "\n".join(f"- {item}" for item in misconceptions[:5])
            + "\n\n"
            if misconceptions
            else ""
        )
        return (
            f"TOPIC:\n{topic}\n\n"
            f"REQUEST: exactly {wanted} questions, overall difficulty '{difficulty}', "
            f"spanning several distinct parts of the material.\n\n"
            f"ALLOWED QUESTION TYPES:\n{type_lines or '   - mcq: 4 distinct options'}\n\n"
            f"{misconception_block}"
            f"EVIDENCE:\n{blocks}\n\n"
            "RULES FOR THE JSON BELOW:\n"
            "- For mcq give exactly 4 options and set correct_index (0-based).\n"
            "- For true_false set correct_answer to \"True\" or \"False\"; options are generated.\n"
            "- For numerical the correct answer is a single number, with units in the explanation.\n"
            "- citation_refs must list the markers (e.g. \"[1]\") of the passages used.\n"
            "- explanation must justify the answer from the evidence, not restate it.\n\n"
            'Return STRICT JSON:\n'
            '{"title": "...", "questions": [{"question": "...", "question_type": "mcq", '
            '"difficulty": "easy|medium|hard", "topic": "...", "options": ["..."], '
            '"correct_index": 0, "correct_answer": "...", "accepted_answers": ["..."], '
            '"explanation": "...", "hint": "...", "citation_refs": ["[1]"], '
            '"misconception_targeted": "..." | null}]}'
        )

    def _build_question(
        self,
        spec: Mapping[str, Any],
        mapping: Mapping[str, str],
        default_difficulty: Difficulty,
        allowed_types: Sequence[QuestionType],
        fallback_topic: str,
    ) -> QuizQuestion | None:
        """Validate one model-supplied item, or discard it."""
        question_text = str(spec.get("question") or "").strip()
        if len(question_text) < 12:
            return None

        qtype = str(spec.get("question_type") or "mcq").strip().lower()
        if qtype not in _QUESTION_TYPES:
            qtype = "mcq"
        if allowed_types and qtype not in allowed_types:
            qtype = next((t for t in allowed_types if t in _QUESTION_TYPES), "mcq")

        refs = _citation_refs(spec.get("citation_refs"), mapping)
        if not refs:
            # An item with no evidence behind it is exactly the failure mode this
            # agent exists to prevent.
            return None

        difficulty = str(spec.get("difficulty") or default_difficulty).strip().lower()
        if difficulty not in ("easy", "medium", "hard", "expert"):
            difficulty = default_difficulty

        topic = str(spec.get("topic") or "").strip() or fallback_topic
        options = [str(o).strip() for o in (spec.get("options") or []) if str(o).strip()]
        correct_answer = str(spec.get("correct_answer") or "").strip()
        accepted = [str(a).strip() for a in (spec.get("accepted_answers") or []) if str(a).strip()]

        if qtype == "true_false":
            options = list(_TRUE_FALSE)
            if not correct_answer:
                index = spec.get("correct_index")
                correct_answer = _TRUE_FALSE[index] if isinstance(index, int) and index < 2 else "True"
            correct_answer = "True" if str(correct_answer).strip().lower().startswith("t") else "False"
            options = [
                "True" if correct_answer == "True" else "False",
                "False" if correct_answer == "True" else "True",
            ]
        elif qtype == "mcq":
            options, correct_answer = _resolve_mcq(options, spec.get("correct_index"), correct_answer)
            if not correct_answer:
                return None

        return QuizQuestion(
            id=f"q_{uuid.uuid4().hex[:10]}",
            question=question_text,
            question_type=qtype,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            topic=topic,
            options=options,
            correct_answer=correct_answer,
            accepted_answers=accepted,
            explanation=str(spec.get("explanation") or "").strip(),
            hint=str(spec.get("hint") or "").strip(),
            points=_clamp(spec.get("points"), 0.5, 10.0) or 1.0,
            citation_refs=refs,
            misconception_targeted=(
                str(spec["misconception_targeted"]).strip()
                if spec.get("misconception_targeted")
                else None
            ),
        )

    def _extractive_questions(
        self,
        blocks: str,
        *,
        needed: int,
        topic: str,
        difficulty: Difficulty,
    ) -> list[QuizQuestion]:
        """Build citation-backed items straight from the retrieved sentences."""
        if needed <= 0:
            return []

        # Map each numbered block back to its chunk ids.
        refs_by_sentence: list[tuple[str, str]] = []  # (sentence, marker)
        for match in re.finditer(r"\[(\d{1,2})\][^\n]*\n(.*?)(?=\n\[\d{1,2}\]\s|\Z)", blocks, re.DOTALL):
            marker = f"[{match.group(1)}]"
            for sentence in sentences_from_text(match.group(2), min_length=60):
                refs_by_sentence.append((sentence, marker))

        pool = [s for s in extract_key_terms(topic, limit=4)] or ["the material"]
        questions: list[QuizQuestion] = []
        used: set[str] = set()
        for sentence, marker in refs_by_sentence:
            if len(questions) >= needed:
                break
            key = normalise_answer(sentence)[:60]
            if key in used:
                continue
            used.add(key)
            statement = sentence if sentence.endswith(".") else sentence + "."
            options = [statement] + [
                "Not stated in the material",
                "It depends on the application",
                f"A different definition of {pool[0]}",
            ]
            rng = random.Random(f"{topic}|{statement[:40]}")
            order = list(range(len(options)))
            rng.shuffle(order)
            shuffled = [options[i] for i in order]
            questions.append(
                QuizQuestion(
                    id=f"q_{uuid.uuid4().hex[:10]}",
                    question=f"Which statement about {topic} is supported by your material?",
                    question_type="mcq",
                    difficulty=difficulty,
                    topic=topic,
                    options=shuffled,
                    correct_answer=statement,
                    explanation=f"The source passage states: {sentence}",
                    hint="Re-read the retrieved passage for this question.",
                    citation_refs=[marker],
                )
            )
        return questions

    @staticmethod
    def _quiz_confidence(quiz: QuizSet) -> float:
        """Confidence that the quiz is actually grounded and usable."""
        if not quiz.questions:
            return 0.0
        with_refs = sum(1 for q in quiz.questions if q.citation_refs)
        with_explanations = sum(1 for q in quiz.questions if q.explanation)
        coverage = with_refs / len(quiz.questions)
        justification = with_explanations / len(quiz.questions)
        diversity = min(1.0, len({q.topic for q in quiz.questions}) / 3.0)
        score = 0.45 * coverage + 0.3 * justification + 0.25 * diversity
        return round(min(0.92, score + 0.12), 4)


# ---------------------------------------------------------------------------
def _clamp(value: Any, low: float, high: float, *, default: float = 0.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _citation_refs(raw: Any, mapping: Mapping[str, str]) -> list[str]:
    """Normalise model-supplied citation refs into known chunk ids."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    refs: list[str] = []
    for item in raw:
        marker = str(item).strip()
        if not marker:
            continue
        if not marker.startswith("["):
            marker = f"[{marker}]"
        chunk_id = mapping.get(marker)
        if chunk_id and chunk_id not in refs:
            refs.append(chunk_id)
    return refs


def _resolve_mcq(
    options: Sequence[str], correct_index: Any, correct_answer: str
) -> tuple[list[str], str]:
    """Return ``(options, correct_answer_text)`` with a consistent ordering."""
    cleaned = [str(o).strip() for o in options if str(o).strip()]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for option in cleaned:
        key = normalise_answer(option)
        if key and key not in seen:
            seen.add(key)
            unique.append(option)

    answer = str(correct_answer or "").strip()
    answer_index: int | None = None

    if isinstance(correct_index, int) and 0 <= correct_index < len(unique):
        answer_index = correct_index
    elif isinstance(correct_index, str) and correct_index.strip().isdigit():
        position = int(correct_index.strip())
        if 0 <= position < len(unique):
            answer_index = position

    if answer_index is None and answer:
        for index, option in enumerate(unique):
            if normalise_answer(option) == normalise_answer(answer):
                answer_index = index
                break
        else:
            if answer.isdigit() and 1 <= int(answer) <= len(unique):
                answer_index = int(answer) - 1

    if answer_index is None and unique:
        # No usable key: fall back to the option that matches the answer text,
        # otherwise treat the first option as correct rather than inventing one.
        answer_index = 0

    if answer_index is not None and not answer:
        answer = unique[answer_index] if answer_index < len(unique) else ""

    if answer and answer not in unique:
        unique.append(answer)

    # Shuffle so the key is not always in the same position.
    rng = random.Random(f"{topic_seed(answer)}{len(unique)}")
    order = list(range(len(unique)))
    rng.shuffle(order)
    shuffled = [unique[i] for i in order]
    final_answer = answer if answer in shuffled else (shuffled[0] if shuffled else answer)
    return shuffled, final_answer


def topic_seed(text: str) -> str:
    """Small helper so option shuffling is deterministic per question."""
    return re.sub(r"\W+", "", text or "")[:32]
