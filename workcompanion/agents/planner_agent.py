"""Study Planner Agent.

Turns a learner's constraints (exam date, hours per day, known weak areas) into a
day-by-day schedule that respects three learning-science rules:

1. **Weak areas first.** Mastery below threshold gets the earliest, deepest slots.
2. **Spaced repetition.** Revision is scheduled on expanding intervals
   (≈1, 3, 7, 14 days after first exposure) rather than crammed at the end.
3. **Retrieval interleaving.** Every study block ends with something active
   (quiz, flashcards, self-explanation), and days are not spent rest-repeating a
   single topic back to back.

The LLM personalises the plan; a deterministic scheduler guarantees a usable plan
even when no provider is available.
"""

from __future__ import annotations

import uuid
from datetime import date as date_type
from datetime import timedelta
from typing import Any

from workcompanion.agents.base import AgentBase
from workcompanion.config.logging_config import get_logger
from workcompanion.schemas.common import AgentResult, GroundingLabel
from workcompanion.schemas.planner import Priority, StudyDay, StudyPlan, StudyPlanInput

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Study Planner Agent inside WorkCompanion AI. You turn a \
learner's constraints into a realistic, day-by-day revision schedule.

ABSOLUTE RULES
1. The plan must cover exactly the requested number of days and never exceed the \
learner's daily hour limit.
2. Weak areas get the earliest and most substantial slots.
3. Apply SPACED REPETITION: schedule revision of each topic on expanding intervals \
(~1, 3, 7 and 14 days after it is first learned), not all at the end.
4. Every day that is not a rest day must contain at least one ACTIVE task: quiz, \
flashcards, practice problems, or self-explanation without notes.
5. Be realistic. Assume a tired student: ~25-45 minutes of focused work per hour.
6. Schedule a full mock exam or mixed self-test in the final 2-3 days.
7. Do not repeat a topic on consecutive days unless it is an explicit revision slot.

OUTPUT
- One entry per day, in order, with concrete, specific activities (not "study")."""

#: Expanding revision offsets in days (classic Ebbinghaus-style ladder).
REVISION_INTERVALS = (1, 3, 7, 14)

_ACTIVITY_LABELS = {
    "learn": "Learn",
    "revise": "Revise",
    "quiz": "Quiz",
    "flashcards": "Flashcards",
    "practice": "Practice problems",
    "mock_exam": "Mock exam",
    "rest": "Rest & consolidate",
}


class PlannerAgent(AgentBase):
    """Adaptive study-plan generation."""

    name = "planner"
    description = "Builds spaced-repetition study plans from learner constraints and progress data."

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ------------------------------------------------------------------
    def create_plan(self, plan_input: StudyPlanInput) -> StudyPlan:
        """Generate a plan, falling back to the deterministic scheduler."""
        days = int(plan_input.days_available or 7)
        plan_id = uuid.uuid4().hex[:12]

        if days <= 0:
            return self._heuristic_plan(plan_input, plan_id)

        prompt = self._build_prompt(plan_input, days)
        request = self.build_request("study_plan", user=prompt, temperature=0.35, max_tokens=3000)

        payload: dict[str, Any] = {}
        try:
            payload = self.structured(request) or {}
        except Exception as exc:
            logger.warning("[planner] generation failed, using heuristic scheduler: %s", exc)

        parsed_days = [item for item in (payload.get("days") or []) if isinstance(item, dict)]
        warnings: list[str] = []
        strategy_notes = [
            str(note).strip() for note in (payload.get("strategy_notes") or []) if str(note).strip()
        ]

        study_days = self._normalise_days(parsed_days, plan_input, days)
        if study_days and not self._prioritises_weak_areas(study_days, plan_input):
            logger.info("[planner] model plan did not schedule weak areas first.")
            study_days = []
        if len(study_days) < max(1, days // 2):
            logger.info("[planner] model plan too sparse; using the deterministic scheduler.")
            plan = self._heuristic_plan(plan_input, plan_id)
            plan.strategy_notes = strategy_notes + plan.strategy_notes
            if study_days:
                plan.warnings.append(
                    "The generated schedule was too sparse to use, so a spaced-repetition "
                    "schedule was applied instead."
                )
            elif parsed_days:
                plan.warnings.append(
                    "The generated schedule did not put your weak areas first, so a "
                    "weak-area-first spaced-repetition schedule was applied instead."
                )
            return plan

        plan = self._assemble(plan_input, plan_id, study_days, strategy_notes, warnings)
        return plan

    # ------------------------------------------------------------------
    def adjust_plan(
        self,
        plan: StudyPlan,
        *,
        completed_day_numbers: list[int] | None = None,
        new_exam_date: date_type | None = None,
        hours_per_day: float | None = None,
    ) -> StudyPlan:
        """Re-plan around reality: days completed, a new deadline, less time."""
        completed = set(completed_day_numbers or [])
        remaining_days = [day for day in plan.days if day.day_number not in completed]

        days_available = len(remaining_days) or plan.total_days
        if new_exam_date is not None:
            days_available = max(1, min(days_available, (new_exam_date - date_type.today()).days or 1))

        updated_input = StudyPlanInput(
            subject=plan.subject,
            exam_date=new_exam_date or plan.exam_date,
            days_available=days_available,
            hours_per_day=hours_per_day or (
                round(plan.total_hours / plan.total_days, 2) if plan.total_days else 2.0
            ),
            topics=sorted({topic for day in remaining_days for topic in day.topics}),
            weak_areas=[],
            notes="Re-planned after a schedule change.",
        )
        revised = self.create_plan(updated_input)
        revised.id = plan.id
        if completed:
            revised.warnings.append(
                f"{len(completed)} completed day(s) were dropped from the revised plan."
            )
        return revised

    # ------------------------------------------------------------------
    def run(
        self,
        plan_input: StudyPlanInput,
        *,
        save_to_history_note: str | None = None,
    ) -> AgentResult:
        """Envelope-returning entry point for chat / crew workflows."""
        plan = self.create_plan(plan_input)
        markdown = self.render_plan(plan)
        if save_to_history_note:
            markdown = f"{markdown}\n\n_{save_to_history_note}_"
        return self.result(
            markdown,
            intent="plan",
            confidence=0.6 if plan.days else 0.0,
            grounding=[GroundingLabel.MODEL_REASONING],
            warnings=plan.warnings,
            metadata={
                "evidence": "learner_history",
                "plan_id": plan.id,
                "subject": plan.subject,
                "total_days": plan.total_days,
                "total_hours": plan.total_hours,
                "exam_date": plan.exam_date.isoformat() if plan.exam_date else None,
            },
        )

    # ------------------------------------------------------------------
    def render_plan(self, plan: StudyPlan) -> str:
        """Markdown rendering used by the chat view and the planner page."""
        lines = [f"### Study plan: {plan.subject}", ""]
        meta = [f"{plan.total_days} days", f"{plan.total_hours:g} h total"]
        if plan.exam_date:
            meta.append(f"exam {plan.exam_date.isoformat()}")
        lines.append(" \u00b7 ".join(meta))
        lines.append("")
        lines.append("| Day | Date | Focus | Activities | h | Why |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for day in plan.days:
            activities = ", ".join(
                _ACTIVITY_LABELS.get(activity, activity) for activity in day.activities
            )
            lines.append(
                f"| {day.day_number} | {day.day_date.isoformat() if day.day_date else '-'} | "
                f"{day.focus} | {activities} | {day.hours:g} | {day.rationale or '-'} |"
            )
        if plan.strategy_notes:
            lines += ["", "**Strategy**", ""]
            lines += [f"- {note}" for note in plan.strategy_notes]
        for warning in plan.warnings:
            lines.append(f"> {warning}")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Deterministic scheduler (always available)
    # ------------------------------------------------------------------
    def _heuristic_plan(self, plan_input: StudyPlanInput, plan_id: str) -> StudyPlan:
        """Build a valid plan without any model call."""
        days_available = int(plan_input.days_available or 7)
        daily_hours = float(plan_input.hours_per_day)
        start = date_type.today()

        weak = [t.strip() for t in plan_input.weak_areas if t.strip()]
        strong = [t.strip() for t in plan_input.strong_areas if t.strip()]
        topics = [t.strip() for t in plan_input.topics if t.strip()] or weak or [
            "Core concepts",
            "Key formulas and definitions",
            "Past-paper practice",
        ]

        # Weak areas first, then unknown topics, then already-strong topics (maintenance only).
        ordered_topics = [*weak, *[t for t in topics if t not in weak], *strong]
        seen: set[str] = set()
        queue = [t for t in ordered_topics if not (t in seen or seen.add(t))]

        study_days: list[StudyDay] = []
        revision_slots: dict[int, list[str]] = {}
        first_seen: dict[str, int] = {}
        phase_rotation = 0
        weak_cycle = 0

        for day_number in range(1, days_available + 1):
            day_date = start + timedelta(days=day_number - 1)
            in_final_window = days_available - day_number <= 1

            # A rest day every 7 days (never on the first or the final two days).
            if days_available >= 6 and day_number % 7 == 0 and not in_final_window:
                study_days.append(
                    StudyDay(
                        day_number=day_number,
                        day_date=day_date,
                        focus="Rest and consolidate",
                        activities=["rest", "flashcards"],
                        topics=[],
                        hours=round(min(1.0, daily_hours * 0.4), 2),
                        priority="low",
                        rationale="Deliberate recovery day; the memory consolidates during rest.",
                        resources=[],
                    )
                )
                continue

            if in_final_window and days_available >= 3:
                # Mock exam in the final window.
                study_days.append(
                    StudyDay(
                        day_number=day_number,
                        day_date=day_date,
                        focus="Mock exam and targeted repair",
                        activities=["mock_exam", "revise"],
                        topics=queue[:3] or ["Whole syllabus"],
                        hours=round(min(daily_hours, max(1.0, daily_hours * 0.8)), 2),
                        priority="high",
                        rationale=(
                            "A timed paper under exam conditions surfaces the gaps that passive "
                            "revision hides."
                        ),
                    )
                )
                continue

            # Spaced-repetition revisions due today.
            due = [topic for topic, seen_day in first_seen.items() if day_number - seen_day in REVISION_INTERVALS]
            for topic in due:
                revision_slots.setdefault(day_number, []).append(topic)

            primary = queue[day_number - 1] if day_number - 1 < len(queue) else queue[
                weak_cycle % len(queue)
            ]
            weak_cycle += 1

            activities: list[str] = ["learn", "practice"]
            if primary in weak or day_number <= 2:
                activities.append("quiz")
            elif day_number % 3 == 0:
                activities.append("flashcards")
            elif day_number % 3 == 1:
                activities.append("quiz")
            else:
                activities.append("flashcards")

            if revision_slots.get(day_number):
                activities.insert(0, "revise")

            first_seen.setdefault(primary, day_number)

            hours = round(min(daily_hours, max(0.75, daily_hours * 0.9)), 2)
            phase_rotation += 1
            rationale = (
                f"Weak area ('{primary}') scheduled early and revisited on a spaced interval."
                if primary in weak
                else "New material with an active retrieval task at the end of the block."
            )
            if revision_slots.get(day_number):
                rationale += " Spaced revision of: " + ", ".join(revision_slots[day_number][:3]) + "."

            study_days.append(
                StudyDay(
                    day_number=day_number,
                    day_date=day_date,
                    focus=primary,
                    activities=activities,
                    topics=[primary, *revision_slots.get(day_number, [])],
                    hours=hours,
                    priority="high" if primary in weak else "medium",  # type: ignore[arg-type]
                    rationale=rationale,
                )
            )

        notes = [
            f"Built from your constraints: {days_available} day(s) at {daily_hours:g} h/day.",
            "Revision follows an expanding ~1/3/7/14-day ladder for each topic.",
            "Every study day ends with retrieval practice (quiz or flashcards).",
        ]
        if weak:
            notes.insert(0, "Weak areas are scheduled first: " + ", ".join(weak[:4]) + ".")
        if strong:
            notes.append(
                "Strong areas (" + ", ".join(strong[:3]) + ") appear late as maintenance only."
            )

        return self._assemble(
            plan_input,
            plan_id,
            study_days,
            notes,
            [] if weak else ["No weak areas were supplied, so topics were scheduled evenly."],
        )

    # ------------------------------------------------------------------
    # Model-plan validation and normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _prioritises_weak_areas(study_days: list[StudyDay], plan_input: StudyPlanInput) -> bool:
        """Reject a model plan that buries weak areas behind topics already mastered.

        Rule 2 of the agent contract ("weak areas get the earliest slots") is a
        hard requirement, so a plan that violates it is discarded rather than
        patched - the deterministic scheduler already gets it right.
        """
        weak = {t.strip().lower() for t in plan_input.weak_areas if t.strip()}
        strong = {t.strip().lower() for t in plan_input.strong_areas if t.strip()}
        if not weak:
            return True

        first_weak = next(
            (
                index
                for index, day in enumerate(study_days)
                if weak & {t.lower() for t in day.topics} or weak & {day.focus.lower()}
            ),
            None,
        )
        if first_weak is None:
            # No weak area appears anywhere - the plan ignores the profile entirely.
            return False

        first_strong_only = next(
            (
                index
                for index, day in enumerate(study_days)
                if (strong & {t.lower() for t in day.topics} or strong & {day.focus.lower()})
                and not (weak & {t.lower() for t in day.topics})
            ),
            None,
        )
        return first_strong_only is None or first_weak <= first_strong_only

    def _normalise_days(
        self, raw_days: list[dict], plan_input: StudyPlanInput, days_available: int
    ) -> list[StudyDay]:
        """Validate model days against the hard constraints."""
        start = date_type.today()
        daily_cap = float(plan_input.hours_per_day)
        output: list[StudyDay] = []

        for index, spec in enumerate(raw_days[:days_available], start=1):
            focus = str(spec.get("focus") or "").strip()
            activities = [str(a).strip().lower() for a in (spec.get("activities") or []) if str(a).strip()]
            if not focus and not activities:
                continue
            if not focus:
                focus = ", ".join(str(t) for t in (spec.get("topics") or [])[:2]) or "Self-directed study"

            topics = [str(t).strip() for t in (spec.get("topics") or []) if str(t).strip()]
            if not topics:
                topics = [focus]

            # Never allow the model to blow the learner's daily budget.
            hours = _to_float(spec.get("hours"), default=round(min(daily_cap, 2.0), 2))
            hours = round(min(hours, daily_cap), 2)
            if hours <= 0:
                hours = round(min(daily_cap, 1.0), 2)

            # Every non-rest day gets a retrieval task, even if the model forgot.
            if not any(activity in ("quiz", "flashcards", "practice", "mock_exam") for activity in activities):
                activities.append("quiz")

            priority = str(spec.get("priority") or "medium").strip().lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"

            output.append(
                StudyDay(
                    day_number=index,
                    day_date=start + timedelta(days=index - 1),
                    focus=focus,
                    activities=activities,
                    topics=topics,
                    hours=hours,
                    priority=priority,  # type: ignore[arg-type]
                    rationale=str(spec.get("rationale") or "").strip(),
                    resources=[str(r).strip() for r in (spec.get("resources") or []) if str(r).strip()],
                )
            )
        return output

    def _assemble(
        self,
        plan_input: StudyPlanInput,
        plan_id: str,
        study_days: list[StudyDay],
        strategy_notes: list[str],
        warnings: list[str],
    ) -> StudyPlan:
        """Finalise totals, exam date and the revision schedule."""
        days_available = int(plan_input.days_available or len(study_days) or 7)
        exam_date = plan_input.exam_date
        if exam_date is None and days_available:
            exam_date = date_type.today() + timedelta(days=days_available)

        if len(study_days) < days_available:
            missing = days_available - len(study_days)
            warnings.append(
                f"{missing} day(s) could not be scheduled from the material available; the "
                "plan covers the days it could."
            )

        revision_schedule: dict[int, float] = {}
        seen: dict[str, int] = {}
        for day in study_days:
            revised = [activity for activity in day.activities if activity in ("revise", "quiz", "flashcards")]
            if not revised:
                continue
            share = round(day.hours * (0.3 if "revise" in day.activities else 0.2), 2)
            if share > 0:
                revision_schedule[day.day_number] = round(
                    revision_schedule.get(day.day_number, 0.0) + share, 2
                )
            for topic in day.topics:
                seen.setdefault(topic, day.day_number)

        total_hours = round(sum(day.hours for day in study_days), 2)
        if plan_input.notes:
            strategy_notes.append(f"Your note: {plan_input.notes}")

        return StudyPlan(
            id=plan_id,
            subject=plan_input.subject,
            exam_date=exam_date,
            total_days=len(study_days),
            total_hours=total_hours,
            days=study_days,
            strategy_notes=strategy_notes,
            revision_schedule=revision_schedule,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    def _build_prompt(self, plan_input: StudyPlanInput, days_available: int) -> str:
        topics = ", ".join(plan_input.topics) or "not specified"
        weak = ", ".join(plan_input.weak_areas) or "not specified"
        strong = ", ".join(plan_input.strong_areas) or "not specified"
        methods = ", ".join(plan_input.preferred_methods) or "any"
        exam = plan_input.exam_date.isoformat() if plan_input.exam_date else "not specified"

        return (
            f"SUBJECT:\n{plan_input.subject}\n\n"
            f"CONSTRAINTS:\n"
            f"- Days available: {days_available}\n"
            f"- Hours per day: {plan_input.hours_per_day:g} (hard maximum)\n"
            f"- Exam date: {exam}\n"
            f"- Current level: {plan_input.current_level}\n"
            f"- Preferred methods: {methods}\n"
            f"- TOPICS: {topics}\n"
            f"- WEAK AREAS: {weak}\n"
            f"- STRONG AREAS: {strong}\n"
            + (f"- Learner note: {plan_input.notes}\n" if plan_input.notes else "")
            + f"\nProduce EXACTLY {days_available} days, day_number 1..{days_available}, "
            f"in order.\n"
            "Apply spaced repetition: revise each topic again after ~1, 3, 7 and 14 days.\n"
            "No day may exceed the daily hour maximum. Put a mock exam in the last 3 days.\n\n"
            'Return STRICT JSON:\n'
            '{"strategy_notes": ["..."], "days": [{"day_number": 1, "focus": "...", '
            '"activities": ["learn","practice","quiz"], "topics": ["..."], "hours": 2.0, '
            '"priority": "high|medium|low", "rationale": "...", "resources": ["..."]}]}'
        )


# ---------------------------------------------------------------------------
def _to_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
