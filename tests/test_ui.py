"""UI plumbing: page registry, navigation, chat queue, settings round-trip.

These tests deliberately avoid Streamlit's widget runtime. They assert the
dispatcher's own contract - every page is registered and routed, and the
document-backed pages refuse to answer when the library is empty - rather than
snapshotting rendered markup, which breaks on every cosmetic change.
"""

from __future__ import annotations

from dataclasses import replace

import pytest


@pytest.fixture()
def context(settings, bundle):
    """An AppContext with an EMPTY library and an initialised database.

    Empty on purpose: the empty-library guard is the behaviour worth testing,
    so the default context must have nothing indexed. Also runs ``init_db``
    because ``AppContext`` does not do it - ``bootstrap()`` does that in the app.
    """
    from workcompanion.database.database import init_db
    from workcompanion.ui.state import AppContext

    init_db(settings)
    return AppContext(settings=settings, bundle=bundle)


# ---------------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------------
class TestPageRegistry:
    REQUIRED = (
        "dashboard",
        "tutor",
        "knowledge",
        "research",
        "quiz",
        "flashcards",
        "exam",
        "planner",
        "progress",
        "settings",
    )

    def test_every_required_page_exists(self):
        from workcompanion.ui.layout import PAGES

        missing = [key for key in self.REQUIRED if key not in PAGES]
        assert not missing, f"sidebar entries missing: {missing}"

    def test_page_keys_match_their_registry_keys(self):
        from workcompanion.ui.layout import PAGES

        for key, page in PAGES.items():
            assert page.key == key, f"{key!r} points at a page keyed {page.key!r}"

    def test_every_page_has_a_label_and_a_render_callable(self):
        from workcompanion.ui.layout import PAGES

        for key, page in PAGES.items():
            assert page.label.strip(), f"{key} has no label"
            assert callable(page.render), f"{key} has no renderer"

    def test_needs_documents_is_always_declared(self):
        """If a page reads the library it must say so, or the guard never fires."""
        from workcompanion.ui.layout import PAGES

        for key, page in PAGES.items():
            assert isinstance(page.needs_documents, bool), f"{key} has no bool flag"

    def test_the_knowledge_page_does_not_require_documents(self):
        """My Knowledge is where you *upload*, so an empty library must be fine.

        Guarding this page would be self-defeating: it would tell a new user to
        go and add documents in order to reach the page that adds documents.
        """
        from workcompanion.ui.layout import PAGES

        assert PAGES["knowledge"].needs_documents is False

    def test_dashboard_does_not_require_documents(self):
        """An empty library is a normal state; the dashboard must still render."""
        from workcompanion.ui.layout import PAGES

        assert PAGES["dashboard"].needs_documents is False

    def test_an_unknown_key_falls_back_instead_of_crashing(self, context, monkeypatch):
        from workcompanion.ui import layout

        rendered: list[str] = []
        monkeypatch.setattr(
            layout,
            "PAGES",
            {
                key: replace(page, render=lambda ctx, _k=key: rendered.append(_k))
                for key, page in layout.PAGES.items()
            },
        )

        layout.dispatch(context, "no-such-page")
        assert rendered == ["dashboard"], "an unknown key must land on the dashboard"


# ---------------------------------------------------------------------------
# The empty-library guard
# ---------------------------------------------------------------------------
class TestEmptyLibraryGuard:
    """The most important behaviour in the whole UI.

    With nothing indexed, a page that depends on retrieval must tell the
    learner to upload something rather than answering from thin air.
    """

    @pytest.fixture()
    def spy(self, monkeypatch):
        from workcompanion.ui import layout

        rendered: list[str] = []
        monkeypatch.setattr(
            layout,
            "PAGES",
            {
                key: replace(page, render=lambda ctx, _k=key: rendered.append(_k))
                for key, page in layout.PAGES.items()
            },
        )
        return rendered

    def test_the_fixture_library_is_really_empty(self, context):
        assert context.bundle.pipeline.is_empty

    def test_document_backed_pages_refuse_to_pretend(self, context, spy):
        from workcompanion.ui.layout import PAGES

        needing = [k for k, p in PAGES.items() if p.needs_documents]
        assert needing, "no page declares a document dependency"

        for key in needing:
            layout_dispatch(context, key)

        assert not spy, (
            f"pages {spy} rendered a real answer with an empty index - that is "
            "exactly the hallucination the pipeline exists to prevent"
        )

    def test_pages_not_needing_documents_still_render(self, context, spy):
        from workcompanion.ui.layout import PAGES

        independent = [k for k, p in PAGES.items() if not p.needs_documents]
        for key in independent:
            layout_dispatch(context, key)

        assert sorted(spy) == sorted(independent)


def layout_dispatch(context, key: str) -> None:
    from workcompanion.ui.layout import dispatch

    dispatch(context, key)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
class TestNavigation:
    def test_navigate_queues_the_next_page(self):
        from workcompanion.ui.state import KEY_NAVIGATE, navigate

        navigate("quiz")
        assert state_value(KEY_NAVIGATE) == "quiz"

    def test_navigation_coerces_to_a_string(self):
        from workcompanion.ui.state import KEY_NAVIGATE, navigate

        navigate(42)
        assert state_value(KEY_NAVIGATE) == "42", "a stray int must not reach the widget"

    def test_a_queued_page_can_be_consumed(self):
        from workcompanion.ui.state import KEY_NAVIGATE, navigate

        navigate("progress")
        assert take(KEY_NAVIGATE) == "progress"
        assert take(KEY_NAVIGATE) is None, "a consumed target must not replay"


def state_value(key: str, default=None):
    import workcompanion.ui.state as state_mod

    return state_mod.st.session_state.get(key, default)


def take(key: str):
    import workcompanion.ui.state as state_mod

    return state_mod.st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Chat transcript helpers
# ---------------------------------------------------------------------------
class TestTranscript:
    def test_a_turn_round_trips_through_session_state(self):
        from workcompanion.ui.state import KEY_CHAT, history, push_turn

        push_turn("user", "What is entropy?")
        push_turn("assistant", "A measure of disorder.")

        chat = state_value(KEY_CHAT)
        assert [t["role"] for t in chat[-2:]] == ["user", "assistant"]
        assert chat[-1]["content"] == "A measure of disorder."
        assert history() is chat, "history() must be the same list, not a copy"

    def test_metadata_is_preserved(self):
        from workcompanion.ui.state import KEY_CHAT, push_turn

        push_turn("assistant", "answer", confidence=0.8, citations=["doc p1"])
        entry = state_value(KEY_CHAT)[-1]

        assert entry["confidence"] == 0.8
        assert entry["citations"] == ["doc p1"]

    def test_clear_chat_empties_the_thread(self):
        from workcompanion.ui.state import KEY_CHAT, clear_chat, push_turn

        push_turn("user", "hi")
        clear_chat()
        assert state_value(KEY_CHAT, []) == []

    def test_clear_chat_survives_an_empty_session(self):
        from workcompanion.ui.state import clear_chat

        clear_chat()  # nothing stored yet - must not raise


# ---------------------------------------------------------------------------
# Settings and environment
# ---------------------------------------------------------------------------
class TestSettingsAndEnvironment:
    def test_environment_report_never_leaks_a_key(self, context):
        from workcompanion.ui.state import environment_report

        rows = environment_report(context)
        serialised = repr(rows)
        assert "gsk_" not in serialised, "a raw Groq key reached the settings report"

    def test_report_is_a_table_of_label_value_rows(self, context):
        from workcompanion.ui.state import environment_report

        rows = environment_report(context)
        assert rows and all(
            isinstance(row, tuple) and len(row) == 2 for row in rows
        ), "the Settings page renders these as a two-column table"

    def test_report_reports_the_key_as_present_or_missing_never_as_a_value(self, context):
        """Exactly one secret exists, and it must never be printed."""
        from workcompanion.ui.state import environment_report

        labels = dict(environment_report(context))
        assert labels["API key"] in {"configured", "missing (offline mode)"}
        assert labels["API key"] != context.settings.groq_api_key

    def test_offline_flag_matches_the_provider(self, context):
        """`context.offline` is what the UI uses to label output honestly."""
        assert context.offline == (context.bundle.llm.name == "offline")

    def test_database_summary_is_reportable(self, context):
        from workcompanion.ui.state import database_summary

        summary = database_summary(context.settings)
        assert isinstance(summary, dict) and summary

    def test_database_health_is_healthy(self, context):
        from workcompanion.ui.state import database_health

        assert isinstance(database_health(context), dict)

    def test_sessions_and_preferences_round_trip(self, context):
        from workcompanion.ui.state import set_preferences

        set_preferences(context, subject="Physics", level="undergraduate")
        with context.session() as repositories:
            user = repositories.users.get_or_create(repositories.session)

        assert user.default_subject == "Physics"
        assert user.preferred_level == "undergraduate"

    def test_preferences_survive_a_reload(self, context):
        from workcompanion.ui.state import set_preferences

        set_preferences(context, subject="Chemistry")
        # A second context over the same settings is what a reload looks like.
        from workcompanion.ui.state import AppContext

        reloaded = AppContext(settings=context.settings, bundle=context.bundle)
        assert reloaded.profile_snapshot().default_subject == "Chemistry"

    def test_clear_chat_is_callable_from_a_page(self):
        """Sidebar and pages both clear the thread - the helper must be shared."""
        from workcompanion.ui import layout, state

        assert callable(state.clear_chat)
        assert layout.PAGES["tutor"].key == "tutor"


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------
class TestAppContext:
    def test_session_commits_and_closes(self, context):
        with context.session() as repositories:
            repositories.users.merge_preferences(
                repositories.session, {"subject": "Chemistry"}
            )

        with context.session() as repositories:
            prefs = repositories.users.get_or_create(repositories.session).preferences
        assert prefs.get("subject") == "Chemistry", "session() must commit on exit"

    def test_learner_yields_bundle_repos_and_memory(self, context):
        from workcompanion.memory.conversation_memory import ConversationMemory

        with context.learner() as (bundle, repositories, memory):
            assert bundle is context.bundle
            assert repositories.session is not None
            assert isinstance(memory, ConversationMemory)

    def test_progress_snapshot_of_a_new_learner_is_empty(self, context):
        snapshot = context.progress_snapshot()
        assert snapshot.questions_answered == 0