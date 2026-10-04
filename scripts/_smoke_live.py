"""End-to-end check against the live Groq API (skipped when no key is set).

Everything else in ``scripts/`` runs on the deterministic offline provider so it
is fast and reproducible.  This one exercises the real thing: real embeddings,
real retrieval, real generations, real citations.

Run:  .\\.venv\\Scripts\\python.exe scripts\\_smoke_live.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTES = """# Photosynthesis: light and limiting factors

## The light-dependent reactions

Light-dependent reactions happen in the thylakoid membranes of the chloroplast.
Light energy splits water into oxygen, protons and electrons:

    2 H2O -> O2 + 4 H+ + 4 e-

Oxygen is therefore a by-product of water splitting, not of carbon dioxide
reduction.

## The Calvin cycle

The Calvin cycle happens in the stroma. It does not need light directly, but it
does need the ATP and NADPH that the light reactions produce. The enzyme RuBisCO
fixes carbon dioxide onto ribulose bisphosphate.

## Limiting factors

The rate of photosynthesis is limited by whichever factor is in shortest supply.
Light intensity, carbon dioxide concentration and temperature are the usual
candidates. When light is raised beyond the saturation point, extra photons give
no extra rate because the photosystems are already working at capacity.
"""

_CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _CHECKS
    _CHECKS += 1
    mark = "PASS" if condition else "FAIL"
    suffix = f" -> {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    if not condition:
        raise AssertionError(label)


def main() -> int:  # noqa: PLR0915
    from workcompanion.config.settings import get_settings

    settings = get_settings()
    if not settings.has_groq_key:
        print("GROQ_API_KEY is not set - skipping the live check.")
        return 0
    if settings.llm_provider != "groq":
        print(f"LLM_PROVIDER is {settings.llm_provider!r} - skipping the live check.")
        return 0

    print(f"model: {settings.groq_model}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        (base / "documents").mkdir()
        source = base / "documents" / "photosynthesis.md"
        source.write_text(NOTES, encoding="utf-8")
        os.environ["DATA_DIR"] = str(base)
        os.environ["CACHE_DIR"] = str(base / "cache")
        os.environ["VECTOR_DB_PATH"] = str(base / "vs")
        os.environ["DOCUMENTS_DIR"] = str(base / "documents")
        os.environ["DATABASE_URL"] = f"sqlite:///{(base / 'wc.db').as_posix()}"
        os.environ["VECTOR_STORE_PROVIDER"] = "memory"
        os.environ["CACHE_ENABLED"] = "false"
        from workcompanion.config.settings import reload_settings

        reload_settings()

        from workcompanion.agents.bundle import get_bundle

        bundle = get_bundle(reload_settings())
        assert bundle.llm.available, "the Groq provider reports itself unavailable"
        print(f"provider: {bundle.llm.name} ({bundle.llm.model})\n")

        outcome = bundle.ingestion.ingest_file(source, subject="Biology")
        check("a document ingests", outcome.ok, outcome.summary())
        bundle.nexus.set_document_state(not bundle.pipeline.is_empty)

        print("\nRETRIEVAL")
        retrieval = bundle.pipeline.retrieve("What limits the rate of photosynthesis?")
        check("retrieval returns passages", bool(retrieval.chunks), str(len(retrieval.chunks)))
        check(
            "retrieval is confident",
            retrieval.confidence > 0.15,
            f"{retrieval.confidence:.2f}",
        )
        check(
            "citations carry real quotes",
            all(c.quote for c in retrieval.citations),
            f"{len(retrieval.citations)} citations",
        )
        check(
            "citations point at the source document",
            all(c.document_name == "photosynthesis.md" for c in retrieval.citations),
        )

        print("\nTUTOR (live generation)")
        from workcompanion.schemas.tutor import TutorRequest

        explained = bundle.tutor.teach(
            TutorRequest(
                question="Explain the Calvin cycle",
                subject="Biology",
                level="undergraduate",
                style="detailed",
            )
        )
        check("the tutor answers", bool(explained.answer), explained.answer[:70])
        check(
            "the answer is grounded, not generic",
            any(
                word in explained.answer.lower()
                for word in ("calvin", "rubisco", "stroma", "atp", "nadph")
            ),
            explained.answer[:90],
        )
        check("the tutor cites sources", bool(explained.sources), str(len(explained.sources)))
        check(
            "the tutor reports confidence", explained.confidence > 0, f"{explained.confidence:.2f}"
        )
        check("latency was measured", explained.latency_ms > 0, f"{explained.latency_ms:.0f}ms")
        check("a real model answered", "qwen" in (explained.model or ""), str(explained.model))

        print("\nKNOWLEDGE ANSWER (live, grounded)")
        answered = bundle.rag.answer("Why is oxygen produced during photosynthesis?")
        check("the grounded answer succeeds", answered.success, answered.answer[:70])
        check("it cites its evidence", bool(answered.sources), str(len(answered.sources)))
        check(
            "it mentions the actual mechanism",
            any(w in answered.answer.lower() for w in ("water", "h2o", "oxygen")),
            answered.answer[:90],
        )

        print("\nREFUSAL BELOW THE CONFIDENCE FLOOR")
        refused = bundle.rag.answer(
            "What is the airspeed velocity of an unladen swallow?",
            allow_general_knowledge=False,
            require_grounding=True,
        )
        check("an ungrounded question is refused", not refused.success, refused.answer[:70])
        check("the refusal explains itself", bool(refused.warnings), str(refused.warnings[:1]))

        print("\nQUIZ (live generation)")
        quiz = bundle.quiz.generate_quiz("Photosynthesis", num_questions=3)
        check("a quiz is produced", bool(quiz.questions), f"{len(quiz.questions)} questions")
        check(
            "every question is answerable",
            all(q.question and q.correct_answer for q in quiz.questions),
        )
        check(
            "multiple-choice items have options",
            all(
                len(q.options) >= 2
                for q in quiz.questions
                if str(getattr(q.question_type, "value", q.question_type))
                in ("multiple_choice", "true_false")
            ),
            str(
                [
                    (str(getattr(q.question_type, "value", q.question_type)), len(q.options))
                    for q in quiz.questions
                ]
            ),
        )

        print("\nGRADING (live)")
        responses = {q.id: q.correct_answer for q in quiz.questions}
        report = bundle.quiz.grade(quiz, responses)
        check(
            "grading returns a report",
            report.total_questions >= 1,
            f"{report.total_questions} marked",
        )
        check(
            "perfect answers score full marks",
            report.score_percent >= 99.0,
            f"{report.score_percent:.0f}%",
        )

        print("\nFLASHCARDS (live generation)")
        deck = bundle.flashcards.generate_deck("Photosynthesis", count=4)
        check("a deck is produced", bool(deck.cards), f"{len(deck.cards)} cards")
        check("every card has a back", all(c.back for c in deck.cards))

        print("\nPLANNER (live generation)")
        from workcompanion.schemas.planner import StudyPlanInput

        plan = bundle.planner.create_plan(
            StudyPlanInput(
                subject="Photosynthesis",
                days_available=5,
                hours_per_day=2.0,
                current_level="beginner",
                topics=["light-dependent reactions", "Calvin cycle", "limiting factors"],
            )
        )
        check("a plan is produced", bool(plan.days), f"{len(plan.days)} days")
        check("the plan explains itself", bool(plan.strategy_notes), str(plan.strategy_notes[:1]))
        check(
            "the plan fits the daily budget",
            all(day.hours <= 2.0 for day in plan.days),
            str([day.hours for day in plan.days]),
        )

    print(f"\nALL LIVE GROQ CHECKS COMPLETED ({_CHECKS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())