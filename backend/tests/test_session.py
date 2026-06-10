"""Shared call-session wiring used by both the browser and Twilio transports.

These pin the fix for the Twilio path, which previously built a manager with no
calendar (booking silently no-opped) and skipped the teardown caller save.
"""

from app.api.session import build_conversation, save_caller, use_tools_enabled
from app.config import Settings
from app.main import _cors_origins


class _Recorder:
    """Stands in for CalendarService.upsert_caller_sync, capturing the kwargs."""

    def __init__(self):
        self.calls: list[dict] = []

    def upsert_caller_sync(self, **kwargs):
        self.calls.append(kwargs)


class _App:
    def __init__(self, calendar):
        self.state = type("S", (), {"calendar": calendar})()


class TestSessionWiring:
    def test_build_conversation_wires_calendar_and_language(self):
        # The Twilio bug: no calendar → _try_book is a no-op → "no free slots".
        calendar = object()
        conv = build_conversation(Settings(language="de"), calendar, "call-1")
        assert conv._calendar is calendar
        assert conv.lang == "de"
        assert conv.state.call_id == "call-1"

    def test_use_tools_enabled_truth_table(self):
        assert use_tools_enabled(Settings(llm_provider="openai")) is True
        assert use_tools_enabled(Settings(llm_provider="ollama", use_tools_local=True)) is True
        assert use_tools_enabled(Settings(llm_provider="ollama", use_tools_local=False)) is False

    def test_save_caller_records_outcome(self):
        recorder = _Recorder()
        conv = build_conversation(Settings(language="de"), recorder, "call-2")
        conv.store_entity("name", "Max Mustermann", 0.95)
        conv.confirm_entity("name")
        conv.store_entity("phone", "+4915112345678", 0.95)
        conv.confirm_entity("phone")
        conv.state.booking_confirmed = True

        save_caller(_App(recorder), conv)

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["outcome"] == "booked"
        assert recorder.calls[0]["name"] == "Max Mustermann"

    def test_save_caller_skips_when_no_contact(self):
        recorder = _Recorder()
        conv = build_conversation(Settings(language="de"), recorder, "call-3")
        save_caller(_App(recorder), conv)
        assert recorder.calls == []  # nothing worth persisting


class TestCorsOrigins:
    def test_comma_separated_parsed_and_stripped(self):
        assert _cors_origins("http://localhost:8000, http://127.0.0.1:8000") == [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    def test_wildcard_passthrough(self):
        assert _cors_origins("*") == ["*"]

    def test_empty_falls_back_to_wildcard(self):
        assert _cors_origins("") == ["*"]
