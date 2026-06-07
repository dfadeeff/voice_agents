"""Tests covering the exact failures observed in the live local demo.

These verify: traffic area support, email required, false booking guard,
and tool name sanitization.
"""

import pytest
from app.conversation.flow import CONTACT_FIELDS
from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea
from app.pipeline.processors import _guard_false_booking


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
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
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
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
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
        ctx.state.entities = {
            "matter_type": _confirmed("matter_type", "accident"),
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
