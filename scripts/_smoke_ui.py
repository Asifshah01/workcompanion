"""Smoke test for the Streamlit UI.

Uses ``streamlit.testing.v1.AppTest`` so every page is really rendered - not just
imported - and the widgets are driven the way a learner would drive them.

Run:  .\\.venv\\Scripts\\python.exe scripts\\_smoke_ui.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTES = """# Thermodynamics: entropy and the second law

## Entropy

Entropy S measures how many microstates are compatible with a macrostate.
For an ideal gas:

    S = nR ln(V/n) + n Cv ln(T) + constant

Entropy is a state function, so the change between two states is path
independent.

## The second law

The second law of thermodynamics states that the entropy of an isolated system
never decreases. A spontaneous process is one for which the total entropy change
is positive.

## Gibbs free energy

At constant temperature and pressure, spontaneity is decided by the Gibbs free
energy:

    G = H - T S

A process is spontaneous when the change in G is negative. A process at
equilibrium has dG equal to zero.

## Enthalpy and phase changes

Enthalpy H absorbs the energy of a phase change without a temperature change.
Latent heat is released in the reverse direction.
"""

_CHECKS: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _CHECKS.append(label)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}{(' -> ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(label)


def main() -> int:  # noqa: PLR0915 - a linear smoke script reads better flat
    """Render every page, drive one real turn through the router, then assert."""
    from streamlit.testing.v1 import AppTest

    # ``ignore_cleanup_errors`` matters on Windows: SQLite keeps the file handle
    # open for the life of the process, so a hard rmtree would fail.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)
        (base / "documents").mkdir(parents=True, exist_ok=True)
        (base / "documents" / "thermodynamics.md").write_text(NOTES, encoding="utf-8")
        (base / "data" / "documents").mkdir(parents=True, exist_ok=True)
        (base / "data" / "documents" / "thermodynamics.md").write_text(NOTES, encoding="utf-8")
        (base / "db").mkdir(parents=True, exist_ok=True)
        (base / "cache").mkdir(parents=True, exist_ok=True)

        env = {
            "DATA_DIR": str(base),
            "DOCUMENTS_DIR": str(base / "data" / "documents"),
            "CACHE_DIR": str(base / "cache"),
            "DATABASE_URL": f"sqlite:///{(base / 'wc.db').as_posix()}",
            "VECTOR_STORE_PROVIDER": "memory",
            "VECTOR_DB_PATH": str(base / "chroma"),
            "LLM_PROVIDER": "offline",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "ENABLE_WEB_RESEARCH": "false",
            "ENABLE_CREWAI": "false",
            "CACHE_ENABLED": "false",
            "LOG_LEVEL": "WARNING",
            "DEBUG": "true",
        }

        import os

        os.environ.update(env)

        print("=" * 78)
        print("APP BOOTSTRAP")
        print("=" * 78)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
        app.run()
        _assert_no_exceptions(app, "bootstrap")
        check("the app boots", not app.exception)
        check("the sidebar renders modes", len(app.sidebar.radio) >= 1, str(len(app.sidebar.radio)))
        modes = list(app.sidebar.radio[0].options)
        check("the sidebar offers nine modes before indexing", len(modes) == 9, str(modes))
        check(
            "document-dependent modes are hidden until there is material",
            "Research" not in modes,
            str(modes),
        )

        print("\n" + "=" * 78)
        print("EVERY PAGE RENDERS")
        print("=" * 78)
        from workcompanion.ui.layout import PAGE_DASHBOARD, PAGE_SETTINGS

        # ``Research`` needs indexed material, so it is covered after ingestion.
        for key in (PAGE_DASHBOARD, "tutor", "knowledge", "quiz", "flashcards", "exam",
                    "planner", "progress", PAGE_SETTINGS):
            app.session_state["wc_nav"] = _label_for(key)
            app.run()
            _assert_no_exceptions(app, f"page:{key}")
            check(f"{key} renders without error", not app.exception)
            app.session_state["wc_nav"] = _label_for(PAGE_DASHBOARD)

        print("\n" + "=" * 78)
        print("INDEX A DOCUMENT AND RE-RUN")
        print("=" * 78)
        app.session_state["wc_nav"] = "My Knowledge"
        app.run()
        _assert_no_exceptions(app, "knowledge-page")
        # Index through the bundle the UI is using, so the check exercises the
        # real ingestion path rather than a parallel store instance.
        _ingest_directly(base)
        app.run()
        _assert_no_exceptions(app, "after-ingest")
        modes_after = list(app.sidebar.radio[0].options)
        check("all ten modes appear once material is indexed", len(modes_after) == 10, str(modes_after))
        check("Research becomes available", "Research" in modes_after, str(modes_after))

        print("\n" + "=" * 78)
        print("RESEARCH PAGE (needs material)")
        print("=" * 78)
        app.session_state["wc_nav"] = "Research"
        app.run()
        _assert_no_exceptions(app, "page:research")
        check("research renders once material exists", not app.exception)

        print("\n" + "=" * 78)
        print("ROUTED TURN")
        print("=" * 78)
        app.session_state["wc_nav"] = _label_for(PAGE_DASHBOARD)
        app.run()
        _assert_no_exceptions(app, "dashboard")
        inputs = app.chat_input
        check("the dashboard has a chat input", len(inputs) >= 1, str(len(inputs)))
        if inputs:
            inputs[0].set_value("Explain the second law of thermodynamics").run()
            _assert_no_exceptions(app, "chat:explain")
            check("a routed answer is shown", not app.exception)

            # Only text bodies count as an answer - not the injected stylesheet.
            markdown = "\n".join(
                _safe(item.value)
                for item in app.markdown
                if "<style>" not in _safe(item.value)
            )
            check(
                "the answer explains the second law",
                "second law" in markdown.lower() or "entropy" in markdown.lower(),
                markdown[:100].replace("\n", " "),
            )
            check("the answer is shown to the learner", len(markdown) > 120)
            transcript = st_state_chat(app)
            check("the turn was recorded", len(transcript) >= 2, str(len(transcript)))
            check(
                "the turn kept its citations",
                bool(transcript and transcript[-1].get("sources")),
                str(bool(transcript and transcript[-1].get("sources"))),
            )
            check("a trace panel is offered", len(app.expander) >= 1)

        print("\n" + "=" * 78)
        print("QUIZ FLOW")
        print("=" * 78)
        app.session_state["wc_nav"] = "Quiz"
        app.run()
        _assert_no_exceptions(app, "quiz-builder")
        text_inputs = [item for item in app.text_input]
        if text_inputs:
            text_inputs[0].set_value("entropy").run()
        generate = [b for b in app.button if "Generate" in str(b.label)]
        check("the quiz builder has a generate button", bool(generate))
        if generate:
            generate[0].click().run()
            _assert_no_exceptions(app, "quiz-generated")
            check("questions are rendered", len(app.radio) + len(app.text_input) > 0)

        print("\n" + "=" * 78)
        print("PLANNER FLOW")
        print("=" * 78)
        app.session_state["wc_nav"] = "Study Planner"
        app.run()
        _assert_no_exceptions(app, "planner-form")
        subject_fields = [
            item for item in app.text_input if str(item.label) == "Subject"
        ]
        check("the planner asks for a subject", bool(subject_fields))
        if subject_fields:
            subject_fields[0].set_value("Thermodynamics").run()
        build = [b for b in app.button if "Build plan" in str(b.label)]
        check("the planner has a build button", bool(build))
        if build:
            build[0].click().run()
            _assert_no_exceptions(app, "planner-built")
            check(
                "a schedule is rendered as a table",
                bool(app.dataframe),
                str(len(app.dataframe)),
            )

        print("\n" + "=" * 78)
        print("SETTINGS SAVES TO .env.local")
        print("=" * 78)
        app.session_state["wc_nav"] = "Settings"
        app.run()
        _assert_no_exceptions(app, "settings-form")
        save = [b for b in app.button if "Save retrieval" in str(b.label)]
        check("settings exposes a retrieval form", bool(save))
        if save:
            save[0].click().run()
            _assert_no_exceptions(app, "settings-saved")

    print(f"\nALL UI SMOKE CHECKS COMPLETED ({len(_CHECKS)} checks)")
    return 0


def st_state_chat(app: object) -> list:
    """The in-session transcript, as the chat module stored it."""
    from workcompanion.ui.state import KEY_CHAT

    return list(app.session_state.get(KEY_CHAT) or [])


def _label_for(page_key: str) -> str:
    """Sidebar label for a page key (the radio is keyed by label, not key)."""
    from workcompanion.ui.layout import PAGES

    return PAGES[page_key].label


def _assert_no_exceptions(app: object, label: str) -> None:
    if app.exception:
        first = app.exception[0]
        message = _safe(getattr(first, "message", first))
        print(f"  [FAIL] {label} raised: {message}")
        for frame in getattr(first, "stack_trace", []) or []:
            print(f"         {_safe(frame)}")
        raise AssertionError(f"{label}: {message}")


def _safe(value: object) -> str:
    """Windows consoles default to cp1252; never let a traceback crash the test."""
    return str(value).encode("ascii", "replace").decode("ascii")


def _ingest_directly(base: Path) -> None:
    """Fallback when the bundled demo directory is absent."""
    from workcompanion.agents.bundle import get_bundle
    from workcompanion.config.settings import get_settings

    bundle = get_bundle(get_settings())
    outcome = bundle.ingestion.ingest_file(
        base / "data" / "documents" / "thermodynamics.md", subject="Thermodynamics"
    )
    bundle.nexus.set_document_state(not bundle.pipeline.is_empty)
    print(f"  indexed {outcome.summary()}")


if __name__ == "__main__":
    sys.exit(main())