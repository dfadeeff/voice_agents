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
        conv.add_user_message("Bitte rufen Sie mich zurück.")
        line = proc._check_fast_path(conv.state.phase, conv.state.phase)
        assert line is not None
        assert conv.state.awaiting == "name"

    def test_callback_entry_is_scripted_en(self):
        proc, conv = self._make_processor("en")
        conv.add_user_message("Could you call me back, please?")
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
        conv.add_user_message("Bitte rufen Sie mich zurück.")
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

    def test_duplicate_fast_path_suppressed_within_window(self):
        # A split utterance re-fires the same scripted line; the repeat must be
        # suppressed so the agent doesn't speak it twice.
        proc, _ = self._make_processor("de")
        line = "Ich habe notiert: klein@hotmail.de — ist das korrekt?"
        assert proc._is_duplicate_fast_path(line, now=100.0) is False  # first → speak
        assert proc._is_duplicate_fast_path(line, now=101.0) is True  # 1s later → suppress
        assert proc._is_duplicate_fast_path("Wie ist Ihr Name?", now=101.5) is False
        assert proc._is_duplicate_fast_path(line, now=120.0) is False  # later turn → allowed

    def test_greeting_fast_path_in_locale(self):
        locale = get_locale("de")
        greeting = locale.FAST_PATH_RESPONSES["greeting"]
        assert "Claudia" in greeting
        assert "?" in greeting


class TestPerTurnEndpointTuning:
    """The VAD end-of-speech window widens only while the next reply is a dictated
    number/email, so those aren't chopped by a mid-utterance pause."""

    def _proc(self):
        from pipecat.audio.vad.vad_analyzer import VADParams

        class _FakeVAD:
            def __init__(self):
                self.stop_secs = None

            def set_params(self, params):
                self.stop_secs = params.stop_secs

        vad = _FakeVAD()
        conv = ConversationManager(call_id="vad-test", lang="de")
        proc = TranscriptProcessor(
            AsyncMock(),
            CallLogger("vad-test"),
            conv,
            vad_analyzer=vad,
            vad_params=VADParams(confidence=0.5, stop_secs=0.8),
            dictation_stop_secs=2.0,
        )
        return proc, conv, vad

    def test_window_for_each_field(self):
        proc, _, _ = self._proc()
        # Free-form turns (dictated number/email + the open problem-description
        # turns where awaiting is None) get the wide window.
        for awaiting in ("email", "phone", "insurance", "insurance_confirm", None):
            assert proc._endpoint_secs_for(awaiting) == 2.0
        # Short scripted answers stay snappy.
        for awaiting in ("name", "name_confirm", "email_confirm", "slot", "matter_type"):
            assert proc._endpoint_secs_for(awaiting) == 0.8

    def test_tune_applies_and_restores_on_the_vad(self):
        proc, conv, vad = self._proc()
        conv.state.awaiting = "phone"
        proc._tune_endpoint()
        assert vad.stop_secs == 2.0  # widened for the phone dictation
        conv.state.awaiting = "slot"
        proc._tune_endpoint()
        assert vad.stop_secs == 0.8  # restored for a normal turn


class TestPhoneTranscriptDisplay:
    """The caller transcript shows a clean phone number, not Whisper's raw
    decimal-littered dictation ('01, 5.1, 5.6, 7.3, 8.6, 9.4')."""

    def _make_processor(self, lang="de"):
        ws = AsyncMock()
        logger = CallLogger("disp-test")
        conv = ConversationManager(call_id="disp-test", lang=lang)
        return TranscriptProcessor(ws, logger, conv), conv

    def test_phone_dictation_is_normalized_for_display(self):
        proc, conv = self._make_processor()
        conv.state.awaiting = "phone"
        assert proc._display_transcript("01, 5.1, 5.6, 7.3, 8.6, 9.4") == "+4915156738694"

    def test_non_phone_turn_is_left_verbatim(self):
        proc, conv = self._make_processor()
        conv.state.awaiting = "name"
        assert proc._display_transcript("Tommy Steinfeld") == "Tommy Steinfeld"

    def test_phone_turn_without_digits_left_verbatim(self):
        proc, conv = self._make_processor()
        conv.state.awaiting = "phone"
        # No usable digits — don't mangle it into an empty/garbage number.
        assert proc._display_transcript("ähm, einen Moment") == "ähm, einen Moment"


class TestWhisperTurnHotwords:
    """The local STT biases its decoder toward the field the agent just asked
    for — email domains on the email turn, spoken digits on the phone turn."""

    def _service(self):
        from app.pipeline.local_whisper import LocalWhisperSTTService

        # Bypass the heavy base __init__ (model resolution); we only exercise the
        # pure hotword-selection logic, which needs just the conversation ref.
        svc = LocalWhisperSTTService.__new__(LocalWhisperSTTService)
        svc._conversation = None
        conv = ConversationManager(call_id="hw-test", lang="de")
        svc.set_conversation(conv)
        return svc, conv

    def test_phone_turn_adds_digit_hotwords(self):
        svc, conv = self._service()
        conv.state.awaiting = "phone"
        hw = svc._hotwords_for_turn()
        assert "fünf" in hw

    def test_email_turn_is_not_biased(self):
        # Biasing "at"/"punkt" made Whisper emit them as literal tokens, which is
        # harder to parse than a plain mishearing — email stays unbiased.
        svc, conv = self._service()
        conv.state.awaiting = "email"
        hw = svc._hotwords_for_turn()
        assert "punkt" not in hw
        assert "gmail.com" not in hw

    def test_other_turn_uses_only_legal_hotwords(self):
        svc, conv = self._service()
        conv.state.awaiting = "name"
        hw = svc._hotwords_for_turn()
        assert "Mietrecht" in hw
