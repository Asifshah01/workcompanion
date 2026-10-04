"""Smoke test for the CrewAI orchestration layer.

Crews are optional, so this test asserts two things:
  1. the *definitions* are always valid (roles, steps, task/tool wiring), and
  2. :func:`run_workflow` degrades to ``None`` instead of raising when no
     API key is configured, so the direct agent path takes over.

Run:  .\\.venv\\Scripts\\python.exe scripts\\_smoke_crew.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Pin the configuration *before* importing the package: this suite is about the
# fallback path, and must not spend live API calls (or rate-limit budget)
# proving it.
os.environ["LLM_PROVIDER"] = "offline"
os.environ["ENABLE_CREWAI"] = "false"
os.environ.pop("GROQ_API_KEY", None)

from workcompanion.agents.bundle import build_bundle
from workcompanion.config.settings import get_settings
from workcompanion.crew import (
    ROLES,
    CrewToolbox,
    build_crew_llm,
    build_workflow,
    crew_enabled,
    crewai_available,
    run_workflow,
    workflow_names,
)
from workcompanion.crew.roles import WORKFLOW_ROLES
from workcompanion.rag.ingestion import IngestionService

NOTES = """# Entropy and the Second Law

## Entropy
Entropy S measures the number of microstates compatible with a macrostate.
For an ideal gas, S = nR ln(V/n) + n Cv ln(T) + const.

The second law of thermodynamics states that the entropy of an isolated system
never decreases. A spontaneous process is one for which the total entropy change
is positive.

## Gibbs free energy
At constant temperature and pressure, spontaneity is decided by the Gibbs free
energy G = H - TS. A process is spontaneous when the change in G is negative.
"""

_CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _CHECKS.append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(label)


def main() -> int:
    settings = get_settings()
    print("=" * 78)
    print("CREW AVAILABILITY")
    print("=" * 78)
    print(f"crewai installed : {crewai_available()}")
    print(f"crews enabled    : {crew_enabled(settings)}")
    llm = build_crew_llm(settings)
    print(f"crew LLM built   : {llm is not None}")
    check(
        "no API key -> crews report disabled",
        crew_enabled(settings) == bool(settings.groq_api_key and settings.enable_crewai),
    )
    check(
        "no API key -> no crew LLM",
        llm is None or bool(settings.groq_api_key),
    )

    print("\n" + "=" * 78)
    print("WORKFLOW DEFINITIONS")
    print("=" * 78)
    check("three workflows are defined", len(workflow_names()) == 3, str(workflow_names()))
    for name in workflow_names():
        workflow = build_workflow(
            name, "How does entropy govern spontaneity?", topic="thermodynamics", count=6
        )
        keys = [step.key for step in workflow.steps]
        print(f"  {name:15} {' -> '.join(keys)}")
        check(f"{name} has steps", bool(keys))
        check(f"{name} steps are unique", len(set(keys)) == len(keys))
        for step in workflow.steps:
            check(f"{name}.{step.key} references a known role", step.role in ROLES, step.role)
        # Every dependency must be an earlier step, or the context is empty.
        for index, step in enumerate(workflow.steps):
            for dependency in step.depends_on:
                check(
                    f"{name}.{step.key} dependency {dependency} runs earlier",
                    dependency in keys[:index],
                )

    print("\n" + "=" * 78)
    print("ROLES")
    print("=" * 78)
    for key, role in sorted(ROLES.items()):
        print(f"  {key:22} tools={list(role.tools)} delegate={role.allow_delegation}")
        check(f"role {key} has a goal", len(role.goal) > 20)
        check(f"role {key} has a backstory", len(role.backstory) > 20)
    for name, role_keys in WORKFLOW_ROLES.items():
        check(f"{name} uses known roles", set(role_keys) <= set(ROLES), str(role_keys))

    print("\n" + "=" * 78)
    print("TOOLBOX")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as tmp:
        local = settings.model_copy(
            update={
                "vector_store_provider": "memory",
                "data_dir": Path(tmp),
                "vector_db_path": Path(tmp) / "chroma",
                "cache_dir": Path(tmp) / "cache",
                "documents_dir": Path(tmp) / "docs",
            }
        )
        local.ensure_directories()
        bundle = build_bundle(local)

        # Ingest through the bundle's own store, exactly like the UI does, so the
        # agents keep seeing the same store instance after ingestion.
        ingestion = IngestionService(
            bundle.store, bundle.sparse, bundle.embeddings, settings=local
        )
        note_path = Path(tmp) / "entropy.md"
        note_path.write_text(NOTES, encoding="utf-8")
        result = ingestion.ingest_file(note_path)
        print(f"  {result.summary()}")
        check("demo document indexed", result.ok and result.chunk_count >= 2, str(result.status))
        bundle.nexus.set_document_state(not bundle.pipeline.is_empty)
        check(
            "pipeline sees the ingested chunks",
            bundle.pipeline.chunk_count == result.chunk_count,
            f"{bundle.pipeline.chunk_count} vs {result.chunk_count}",
        )

        toolbox = CrewToolbox(bundle, local)
        tools = {tool.name: tool for tool in toolbox.build()}
        print(f"  tools: {sorted(tools)}")
        check("seven tools are built", len(tools) == 7, str(sorted(tools)))
        for name, tool in sorted(tools.items()):
            check(f"tool {name} documents itself", len(tool.description) > 40)

        found = tools["search_notes"].run(query="What makes a process spontaneous?", top_k=3)
        print(f"\n  search_notes -> {found[:200]!r}")
        check("search_notes returns passages", "No passage" not in found)
        check("search_notes returns citations", "**Sources**" in found)

        taught = tools["teach_concept"].run(
            question="What is Gibbs free energy?", level="beginner"
        )
        print(f"  teach_concept -> {taught[:160]!r}")
        check("teach_concept returns an explanation", len(taught) > 80)

        quiz = tools["generate_quiz"].run(
            topic="entropy", num_questions=3, difficulty="medium"
        )
        print(f"  generate_quiz -> {quiz[:160]!r}")
        check("generate_quiz returns questions", quiz.count("\n**") >= 1 or "#" in quiz)

        cards = tools["generate_flashcards"].run(topic="entropy", count=4)
        print(f"  generate_flashcards -> {cards[:160]!r}")
        check("generate_flashcards returns cards", "->" in cards)

        plan = tools["make_study_plan"].run(
            subject="Thermodynamics", days_available=5, hours_per_day=1.5,
            weak_areas="Gibbs free energy",
        )
        print(f"  make_study_plan -> {plan[:160]!r}")
        check("make_study_plan returns a schedule", "| Day |" in plan and plan.count("| 1") >= 1)
        check("make_study_plan respects the budget", "7.5 h total" in plan)

        analysis = tools["analyze_paper"].run(
            document_id=result.document_id, level="beginner"
        )
        print(f"  analyze_paper -> {analysis[:160]!r}")
        check("analyze_paper returns an analysis", len(analysis) > 80)

        print("\n" + "=" * 78)
        print("EXECUTION FALLBACK")
        print("=" * 78)
        for name in workflow_names():
            workflow = build_workflow(name, "How does entropy govern spontaneity?")
            if crew_enabled(local):
                print(f"  {name}: crews are enabled, skipping the fallback assertion")
                continue
            outcome = run_workflow(workflow, bundle, local)
            check(f"{name} returns None when crews are unavailable", outcome is None)
        check(
            "build_crew_llm never raises without a key",
            build_crew_llm(local) is None or bool(local.groq_api_key),
        )
        try:
            build_workflow("nope", "x")
        except KeyError as exc:
            check("unknown workflow raises KeyError", "nope" in str(exc))
        else:
            check("unknown workflow raises KeyError", False)

    print(f"\nALL CREW SMOKE CHECKS COMPLETED ({len(_CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
