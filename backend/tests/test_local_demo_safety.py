"""Tests covering the exact failures observed in the live local demo.

These verify: traffic area support, email required, false booking guard,
conflict check gating, and tool name sanitization.
"""

import pytest
from app.conversation.flow import REQUIRED_FIELDS
from app.conversation.manager import ConversationManager
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea
from app.pipeline.processors import _guard_false_booking


def _confirmed(field, value="test"):
    return ExtractedEntity(field_name=field, value=value, confidence=0.95, confirmed=True)


class TestTrafficArea:
    def test_traffic_area_exists(self):
        assert LegalArea.TRAFFIC.value == "traffic"

    def test_traffic_skips_conflict_check(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TRAFFIC
        ctx.state.intake_complete = True
        ctx.state.entities = {
            "name": _confirmed("name", "Dmitry Fadeev"),
            "email": _confirmed("email", "dima@example.com"),
            "phone": _confirmed("phone", "017612345678"),
        }
        ctx.advance_phase()
        assert ctx.state.phase != CallPhase.CONFLICT_CHECK

    def test_employment_still_requires_conflict_check(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.EMPLOYMENT
        ctx.state.intake_complete = True
        ctx.state.entities = {
            "name": _confirmed("name", "Dmitry Fadeev"),
            "email": _confirmed("email", "dima@example.com"),
            "phone": _confirmed("phone", "017612345678"),
        }
        ctx.advance_phase()
        assert ctx.state.phase == CallPhase.CONFLICT_CHECK

    def test_tenancy_skips_conflict_check(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TENANCY
        ctx.state.intake_complete = True
        ctx.state.entities = {
            "name": _confirmed("name"),
            "email": _confirmed("email"),
            "phone": _confirmed("phone"),
        }
        ctx.advance_phase()
        assert ctx.state.phase != CallPhase.CONFLICT_CHECK


class TestEmailRequired:
    def test_required_fields_are_name_email_phone(self):
        assert REQUIRED_FIELDS == ("name", "email", "phone")

    def test_name_and_phone_without_email_stays_in_capture(self):
        ctx = ConversationManager(call_id="test", lang="de")
        ctx.state.turn_count = 3
        ctx.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        ctx.state.legal_area = LegalArea.TRAFFIC
        ctx.state.intake_complete = True
        ctx.state.entities = {
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
        ctx.state.intake_complete = True
        ctx.state.entities = {
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
        proc = _make_processor(["classify_legal_area"])
        result = proc._strip_tool_names("Moment bitte.-Classify legal area.")
        assert "Classify" not in result
        assert "legal area" not in result

    def test_strips_underscore_tool_name(self, _make_processor):
        proc = _make_processor(["classify_legal_area"])
        result = proc._strip_tool_names("classify_legal_area")
        assert result == ""

    def test_strips_with_json_payload(self, _make_processor):
        proc = _make_processor(["classify_legal_area"])
        result = proc._strip_tool_names('classify_legal_area{"area":"traffic"}')
        assert result == ""

    def test_strips_escalate_to_human(self, _make_processor):
        proc = _make_processor(["escalate_to_human"])
        result = proc._strip_tool_names("Escalate to human")
        assert result == ""

    def test_preserves_normal_text(self, _make_processor):
        proc = _make_processor(["classify_legal_area"])
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
        san = self._make_sanitizer(["classify_legal_area"])
        assert san._tool_re is not None
        assert san._tool_re.search("classify legal area")

    def test_no_tool_regex_when_empty(self):
        san = self._make_sanitizer()
        assert san._tool_re is None
