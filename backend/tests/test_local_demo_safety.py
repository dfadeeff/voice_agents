"""Tests covering the exact failures observed in the live local demo.

These verify: traffic area support, email required, false booking guard,
and tool name sanitization.
"""

import pytest
from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app.conversation.flow import CONTACT_FIELDS
from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea
from app.pipeline.orchestrator import register_tools_on_llm
from app.pipeline.processors import _guard_false_booking, _guard_impossible_handoff
from app.tools.registry import ToolRegistry


def _confirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.95, confirmed=True)


class TestTrafficArea:
    def test_traffic_area_exists(self):
        assert LegalArea.TRAFFIC.value == "traffic"

    def test_traffic_goes_to_booking_when_all_confirmed(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TRAFFIC
        ctx.state.insurance_resolved = True
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
            "matter_details": _confirmed("matter_details", "police on scene"),
            "name": _confirmed("name", "Dmitry Fadeev"),
            "email": _confirmed("email", "dima@example.com"),
            "phone": _confirmed("phone", "017612345678"),
        }
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.BOOKING


class TestEmailRequired:
    def test_contact_fields_are_name_email_phone(self):
        assert CONTACT_FIELDS == ("name", "email", "phone")

    def test_name_and_phone_without_email_stays_in_capture(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TRAFFIC
        ctx.state.insurance_resolved = True
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
            "matter_details": _confirmed("matter_details", "police on scene"),
            "name": _confirmed("name", "Dmitry Fadeev"),
            "phone": _confirmed("phone", "017612345678"),
        }
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.CAPTURE

    def test_all_three_fields_exits_capture(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TRAFFIC
        ctx.state.insurance_resolved = True
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
            "matter_details": _confirmed("matter_details", "police on scene"),
            "name": _confirmed("name", "Dmitry Fadeev"),
            "email": _confirmed("email", "dima@example.com"),
            "phone": _confirmed("phone", "017612345678"),
        }
        ctx.advance_phase()
        assert ctx.state.phase != CallPhase.CAPTURE


class TestFalseBookingGuard:
    def test_blocks_termin_gebucht(self):
        bad = "Ich habe gleich einen Termin für Sie gebucht."
        cleaned = _guard_false_booking(bad)
        assert "gebucht" not in cleaned.lower()
        assert "Terminwunsch" in cleaned

    def test_blocks_termin_bestätigt(self):
        bad = "Ihr Termin ist bestätigt."
        cleaned = _guard_false_booking(bad)
        assert "bestätigt" not in cleaned.lower()

    def test_blocks_termin_reserviert(self):
        bad = "Der Termin ist reserviert."
        cleaned = _guard_false_booking(bad)
        assert "reserviert" not in cleaned.lower()

    def test_allows_terminwunsch(self):
        ok = "Ich nehme Ihren Terminwunsch auf."
        assert _guard_false_booking(ok) == ok

    def test_allows_normal_speech(self):
        ok = "Verstanden, es geht um einen Unfall."
        assert _guard_false_booking(ok) == ok


class TestTruthfulHandoffGuard:
    def test_blocks_impossible_german_live_transfer(self):
        bad = "Moment bitte, ich verbinde Sie direkt zu Frau Landau."
        cleaned = _guard_impossible_handoff(bad, "de")
        assert "verbinde" not in cleaned.lower()
        assert "Rückrufwunsch" in cleaned

    def test_preserves_truthful_callback(self):
        text = "Ich kann Ihren Rückrufwunsch an das Kanzleiteam weitergeben."
        assert _guard_impossible_handoff(text, "de") == text


class TestToolNameSanitizer:
    @pytest.fixture()
    def _make_processor(self):
        from app.pipeline.processors import AgentTextProcessor

        class FakeWS:
            async def send_json(self, data):
                pass

        from app.pipeline.processors import CallLogger

        def factory(tool_names):
            return AgentTextProcessor(
                websocket=FakeWS(),
                call_logger=CallLogger("test"),
                lang="de",
                tool_names=tool_names,
            )

        return factory

    def test_strips_space_separated_tool_name(self, _make_processor):
        proc = _make_processor(["route_call"])
        result = proc._strip_tool_names("Moment bitte.-Route call.")
        assert "Route" not in result
        assert "route call" not in result.lower()

    def test_strips_underscore_tool_name(self, _make_processor):
        proc = _make_processor(["route_call"])
        result = proc._strip_tool_names("route_call")
        assert result == ""

    def test_strips_with_json_payload(self, _make_processor):
        proc = _make_processor(["route_call"])
        result = proc._strip_tool_names('route_call{"area":"traffic"}')
        assert result == ""

    def test_strips_request_handoff(self, _make_processor):
        proc = _make_processor(["request_handoff"])
        result = proc._strip_tool_names("Request handoff")
        assert result == ""

    def test_preserves_normal_text(self, _make_processor):
        proc = _make_processor(["route_call"])
        text = "Verstanden, es geht um einen Unfall."
        result = proc._strip_tool_names(text)
        assert result == text


class TestPreTTSSanitizer:
    def _make_sanitizer(self, tool_names=None):
        from app.pipeline.processors import PreTTSSanitizer

        return PreTTSSanitizer(lang="de", tool_names=tool_names)

    def test_strips_think_tags(self):
        san = self._make_sanitizer()
        assert san._strip_think("<think>reasoning</think>Hallo") == "Hallo"

    def test_germanizes_english_honorifics(self):
        san = self._make_sanitizer()
        assert "Herr Steinmeier" in san._sanitize("Vielen Dank, Mr. Steinmeier.")
        assert "Mr." not in san._sanitize("Vielen Dank, Mr. Steinmeier.")
        assert "Frau Sommer" in san._sanitize("Guten Tag, Mrs. Sommer!")
        assert "Frau Landau" in san._sanitize("Hallo Ms. Landau.")

    def test_spells_out_alphanumeric_reference(self):
        san = self._make_sanitizer()
        out = san._sanitize("Ich notiere Ihre Versicherungsnummer: F62415723")
        assert "F, 6, 2, 4, 1, 5, 7, 2, 3" in out

    def test_strips_tool_call_scars(self):
        san = self._make_sanitizer(tool_names=["confirm_caller_detail"])
        out = san._sanitize("Nummer: 0711()-Ich werde Herrn Schulz informieren.")
        assert "()" not in out
        assert "-Ich" not in out

    def test_strips_think_tags_across_chunks(self):
        san = self._make_sanitizer()
        assert san._strip_think("<think>start of thought") == ""
        assert san._in_think is True
        assert san._strip_think("still thinking</think>Guten Tag") == "Guten Tag"
        assert san._in_think is False

    def test_strips_cjk_in_process(self):
        san = self._make_sanitizer()
        result = san._strip_think("你好 Hallo")
        assert "Hallo" in result

    def test_tool_name_regex_built(self):
        san = self._make_sanitizer(["route_call"])
        assert san._tool_re is not None
        assert san._tool_re.search("route call")

    def test_no_tool_regex_when_empty(self):
        san = self._make_sanitizer()
        assert san._tool_re is None

    def test_drops_empty_internal_array(self):
        san = self._make_sanitizer()
        assert san._sanitize("[]") == ""

    def test_drops_internal_json_object(self):
        san = self._make_sanitizer()
        assert san._sanitize('{"status": "handoff_requested"}') == ""

    def test_rewrites_impossible_transfer_before_tts(self):
        san = self._make_sanitizer()
        cleaned = san._sanitize("Moment bitte, ich verbinde Sie direkt zu Frau Landau.")
        assert "verbinde" not in cleaned.lower()
        assert "Rückrufwunsch" in cleaned

    def test_drops_garbled_tool_name_with_payload(self):
        """Real bug: LLM leaked a mangled capture_caller_details as spoken text.

        The exact-name denylist missed 'roring_caller_details' because it isn't
        an exact tool name. The snake_case pattern catches it regardless.
        """
        san = self._make_sanitizer(["capture_caller_details"])
        assert san._sanitize('roring_caller_details {"matter_type": "accident"}') == ""

    def test_drops_bare_snake_case_identifier(self):
        san = self._make_sanitizer()
        assert san._sanitize("preferred_date") == ""

    def test_drops_unfilled_placeholder_sentence(self):
        """Real bug: 'Am liebsten am [preferred_date] um [preferred_time]?'."""
        san = self._make_sanitizer()
        assert san._sanitize("Am liebsten am [preferred_date] um [preferred_time]?") == ""

    def test_preserves_email_with_underscore(self):
        """snake_case stripping must NOT eat an email local-part like fade_jeff."""
        san = self._make_sanitizer(["capture_caller_details"])
        cleaned = san._sanitize("Ihre E-Mail ist fade_jeff@gmail.com, korrekt?")
        assert "fade_jeff" in cleaned or "fade" in cleaned
        assert "korrekt" in cleaned

    def test_preserves_normal_german_sentence(self):
        san = self._make_sanitizer(["capture_caller_details", "route_call"])
        text = "Habe ich Sie richtig verstanden, dass es um ein Verkehrsunfall geht?"
        assert san._sanitize(text) == text

    @pytest.mark.asyncio
    async def test_strips_tool_name_split_across_streamed_chunks(self):
        san = self._make_sanitizer(["classify_legal_area"])
        pushed = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        san.push_frame = capture
        await san.process_frame(TextFrame("Moment bitte.-Classify"), FrameDirection.DOWNSTREAM)
        assert pushed == []

        await san.process_frame(TextFrame(" legal area."), FrameDirection.DOWNSTREAM)
        # Trailing space is intentional: each sentence is pushed as its own frame
        # and the downstream TTS aggregator concatenates them verbatim, so the
        # separator must survive or adjacent sentences glue ("bitte.Wie…").
        assert [frame.text for frame in pushed] == ["Moment bitte. "]

    @pytest.mark.asyncio
    async def test_discards_partial_sentence_on_interruption(self):
        san = self._make_sanitizer(["classify_legal_area"])
        pushed = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        san.push_frame = capture
        await san.process_frame(TextFrame("Old booking details"), FrameDirection.DOWNSTREAM)
        await san.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await san.process_frame(TextFrame("Wie kann ich helfen?"), FrameDirection.DOWNSTREAM)

        spoken = [frame.text for frame in pushed if isinstance(frame, TextFrame)]
        assert spoken == ["Wie kann ich helfen? "]

    @pytest.mark.asyncio
    async def test_exact_named_person_failure_is_safe_before_tts(self):
        san = self._make_sanitizer(["request_handoff"])
        pushed = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        san.push_frame = capture
        await san.process_frame(
            TextFrame("Vielen Dank für Ihren Anruf.Moment bitte, "),
            FrameDirection.DOWNSTREAM,
        )
        await san.process_frame(
            TextFrame("ich verbinde Sie direkt zu Frau Landau. []"),
            FrameDirection.DOWNSTREAM,
        )
        await san.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        spoken = [frame.text for frame in pushed if type(frame) is TextFrame]
        assert spoken == [
            "Vielen Dank für Ihren Anruf. ",
            "Ich kann Ihren Rückrufwunsch aufnehmen und an das Kanzleiteam weitergeben. ",
        ]


class TestSpokenTranscript:
    @pytest.mark.asyncio
    async def test_frontend_receives_text_actually_submitted_to_tts(self):
        from app.pipeline.processors import AgentTextProcessor, CallLogger

        class FakeWS:
            def __init__(self):
                self.messages = []

            async def send_json(self, data):
                self.messages.append(data)

        ws = FakeWS()
        logger = CallLogger("test")
        proc = AgentTextProcessor(ws, logger, lang="de")

        async def discard(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        proc.push_frame = discard
        await proc.process_frame(
            TTSTextFrame("Das wurde gesprochen.", aggregated_by="sentence"),
            FrameDirection.DOWNSTREAM,
        )

        assert ws.messages == [{"type": "agent_text", "text": "Das wurde gesprochen."}]
        assert logger.entries[-1]["text"] == "Das wurde gesprochen."

    @pytest.mark.asyncio
    async def test_frontend_does_not_receive_generated_only_text(self):
        from app.pipeline.processors import AgentTextProcessor, CallLogger

        class FakeWS:
            def __init__(self):
                self.messages = []

            async def send_json(self, data):
                self.messages.append(data)

        ws = FakeWS()
        proc = AgentTextProcessor(ws, CallLogger("test"), lang="de")

        async def discard(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        proc.push_frame = discard
        await proc.process_frame(
            AggregatedTextFrame("Generated but interrupted.", aggregated_by="sentence"),
            FrameDirection.DOWNSTREAM,
        )

        assert ws.messages == []


def test_tool_calls_are_cancelled_when_caller_interrupts(conversation):
    async def noop(arguments, ctx):
        return {"status": "ok"}

    registry = ToolRegistry()
    registry.register("test_tool", noop, "Test", {"type": "object", "properties": {}})

    class FakeLLM:
        def __init__(self):
            self.registrations = []

        def register_function(self, name, handler, *, cancel_on_interruption=True):
            self.registrations.append((name, cancel_on_interruption))

    llm = FakeLLM()
    register_tools_on_llm(llm, registry, conversation)

    assert llm.registrations == [("test_tool", True)]
