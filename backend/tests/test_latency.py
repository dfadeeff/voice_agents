"""Tests for latency improvements: Whisper tuning, filler injection, fast paths."""

from unittest.mock import AsyncMock

from app.config import Settings
from app.conversation.locales import get_locale
from app.conversation.manager import ConversationManager
from app.models.schemas import CallPhase
from app.pipeline.processors import CallLogger, FillerInjector, TranscriptProcessor

# ---------------------------------------------------------------------------
# Whisper tuning config
# ---------------------------------------------------------------------------


class TestWhisperConfig:
    def test_default_beam_size_is_1(self):
        s = Settings(_env_file=None)
        assert s.whisper_beam_size == 1

    def test_default_vad_filter_is_false(self):
        s = Settings(_env_file=None)
        assert s.whisper_vad_filter is False

    def test_beam_size_from_env(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
        s = Settings(_env_file=None)
        assert s.whisper_beam_size == 3

    def test_vad_filter_from_env(self, monkeypatch):
        monkeypatch.setenv("WHISPER_VAD_FILTER", "true")
        s = Settings(_env_file=None)
        assert s.whisper_vad_filter is True

    def test_default_filler_delay(self):
        s = Settings(_env_file=None)
        assert s.filler_delay_ms == 1500

    def test_filler_delay_from_env(self, monkeypatch):
        monkeypatch.setenv("FILLER_DELAY_MS", "0")
        s = Settings(_env_file=None)
        assert s.filler_delay_ms == 0


# ---------------------------------------------------------------------------
# Filler injection
# ---------------------------------------------------------------------------


class TestFillers:
    def test_de_fillers_exist(self):
        locale = get_locale("de")
        assert hasattr(locale, "FILLERS")
        assert len(locale.FILLERS) >= 2

    def test_en_fillers_exist(self):
        locale = get_locale("en")
        assert hasattr(locale, "FILLERS")
        assert len(locale.FILLERS) >= 2

    def test_filler_rotation(self):
        conv = ConversationManager(call_id="filler-test", lang="de")
        injector = FillerInjector(conversation=conv, delay_s=1.5)
        fillers = []
        for _ in range(6):
            fillers.append(injector._get_filler())

        assert fillers[0] != fillers[1]
        assert fillers[0] == fillers[3]


# ---------------------------------------------------------------------------
# Fast path responses
# ---------------------------------------------------------------------------


class TestFillerInjector:
    def test_no_task_when_delay_zero(self):
        conv = ConversationManager(call_id="test", lang="de")
        injector = FillerInjector(conversation=conv, delay_s=0)
        injector.start_filler_timer()
        assert injector._filler_task is None


class TestFastPathResponses:
    def test_de_fast_path_responses_exist(self):
        locale = get_locale("de")
        assert hasattr(locale, "FAST_PATH_RESPONSES")
        assert "greeting" in locale.FAST_PATH_RESPONSES
        assert "callback_ask_name" in locale.FAST_PATH_RESPONSES

    def test_en_fast_path_responses_exist(self):
        locale = get_locale("en")
        assert hasattr(locale, "FAST_PATH_RESPONSES")
        assert "greeting" in locale.FAST_PATH_RESPONSES
        assert "callback_ask_name" in locale.FAST_PATH_RESPONSES


class TestFastPathDetection:
    def _make_processor(self, lang="de"):
        ws = AsyncMock()
        logger = CallLogger("fp-test")
        conv = ConversationManager(call_id="fp-test", lang=lang)
        proc = TranscriptProcessor(ws, logger, conv)
        return proc, conv

    def test_callback_entry_is_scripted(self):
        """After a callback request, the name ask is scripted (deterministic spine)."""
        proc, conv = self._make_processor("de")
        conv.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        line = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert line is not None
        assert conv.state.awaiting == "name"

    def test_callback_entry_is_scripted_en(self):
        proc, conv = self._make_processor("en")
        conv.add_user_message("I'd like to speak to Mr Smith please.")
        line = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert line is not None
        assert conv.state.awaiting == "name"

    def test_area_confirmation_is_scripted(self):
        """Area confirmation / matter-type question is scripted, not LLM-driven."""
        proc, conv = self._make_processor("de")
        conv.add_user_message("Ich wurde letzte Woche gekündigt.")
        line = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert line is not None
        assert conv.state.awaiting == "matter_type"

    def test_no_fast_path_for_ambiguous_routing(self):
        proc, conv = self._make_processor("de")
        conv.add_user_message("Ich habe da ein Problem.")  # no keyword → stays ROUTING
        result = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert result is None

    def test_phone_readback_is_scripted_in_callback(self):
        """The whole callback capture spine is scripted; phone is read back exactly."""
        proc, conv = self._make_processor("de")
        conv.add_user_message("Ich möchte bitte Herrn Schmid sprechen.")
        assert conv.state.phase == CallPhase.CAPTURE
        # Scripted name ask.
        assert proc._check_fast_path(conv.state.phase, conv.state.phase) is not None
        conv.add_user_message("Max Mustermann")
        # Scripted phone ask.
        assert proc._check_fast_path(conv.state.phase, conv.state.phase) is not None
        conv.add_user_message("0151 598 32614")
        line = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert line is not None
        assert "015159832614" in line

    def test_no_fast_path_when_not_callback(self):
        proc, conv = self._make_processor("de")
        result = proc._check_fast_path(CallPhase.CAPTURE, CallPhase.CAPTURE)
        assert result is None

    def test_greeting_fast_path_in_locale(self):
        locale = get_locale("de")
        greeting = locale.FAST_PATH_RESPONSES["greeting"]
        assert "Claudia" in greeting
        assert "?" in greeting
