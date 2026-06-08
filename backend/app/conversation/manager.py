from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from app.conversation.flow import PHASE_TOOLS, next_phase
from app.conversation.phone import normalize_phone_text
from app.conversation.policy import extract_target_person, is_explicit_handoff_request
from app.conversation.prompts import build_system_prompt, get_system_prompt_base
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea

logger = logging.getLogger(__name__)

# Deterministic contact-capture fallback (mirrors _try_auto_route): when the
# caller states a name or phone but the LLM forgets to call capture_caller_details,
# we extract and store it in code so the data still persists to the database.
_NAME_TRIGGER_RE = re.compile(
    r"(?:mein\s+name\s+ist|ich\s+hei(?:ß|ss)e|ich\s+bin|hier\s+(?:ist|spricht)|"
    r"my\s+name\s+is|i\s+am|i'm|this\s+is)\s+",
    re.IGNORECASE,
)
_CAPWORDS_RE = re.compile(r"([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2})")
# Cue that the agent's last turn asked for the name, so a bare reply ("Daniel
# Stein") is the name even without a "Mein Name ist" trigger.
_NAME_ASK_RE = re.compile(
    r"\b(?:ihren?\s+namen|ihr\s+name|wie\s+(?:hei(?:ß|ss)en\s+sie|ist\s+ihr\s+name)|"
    r"your\s+name|may\s+i\s+(?:have|take|ask)\b)",
    re.IGNORECASE,
)
# Capitalised sentence-starters that are not names (guards the bare-reply path).
_NAME_STOPWORDS = {
    "ich",
    "ja",
    "nein",
    "herr",
    "frau",
    "mein",
    "meine",
    "der",
    "die",
    "das",
    "es",
    "bitte",
    "danke",
    "guten",
    "hallo",
    "und",
    "aber",
    "nee",
    "ok",
    "okay",
}

# Keyword → matter_type per area, checked in priority order (most specific first).
# Backfills matter_type when the caller front-loads details or the LLM skips the
# tool, so the matter is recorded even if qualification is cut short by a handoff.
_MATTER_KEYWORDS: dict[LegalArea, tuple[tuple[str, tuple[str, ...]], ...]] = {
    LegalArea.TRAFFIC: (
        ("accident", ("unfall", "zusammenstoß", "zusammenstoss", "accident", "crash")),
        ("damage", ("kfz-schaden", "blechschaden", "kratzer", "beschädigung", "vehicle damage")),
        ("insurance", ("versicherung", "insurance", "insurer")),
    ),
    LegalArea.EMPLOYMENT: (
        ("dismissal", ("kündigung", "gekündigt", "entlass", "dismissal", "redundan")),
        ("warning", ("abmahnung", "abgemahnt", "warning")),
        ("wages", ("lohn", "gehalt", "lohnstreit", "wages", "salary")),
        ("contract", ("arbeitsvertrag", "contract")),
    ),
    LegalArea.TENANCY: (
        ("eviction", ("räumung", "raeumung", "eviction", "evict")),
        ("deposit", ("kaution", "deposit")),
        ("rent_increase", ("mieterhöhung", "mieterhoehung", "rent increase")),
        ("repairs", ("mängel", "maengel", "reparatur", "schimmel", "repairs", "mould", "mold")),
    ),
}

# Cue that the agent's last turn asked for an insurance/claim/damage number, so a
# number in the caller's reply is that reference (context-aware, not a blind grab).
_INSURANCE_ASK_RE = re.compile(
    r"versicherungs(?:nummer|nr)|schadens(?:nummer|nr|referenz)|policen(?:nummer|nr)|"
    r"insurance\s+number|claim\s+number|policy\s+number|damage\s+number",
    re.IGNORECASE,
)
_REF_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]*")

# Negation cue used by the insurance step ("nein", "noch keine").
_NEGATE_RE = re.compile(
    r"\b(?:nein|nicht|falsch|kein|keine|nö|nee|no|wrong|incorrect)\b",
    re.IGNORECASE,
)
# A denial of the area-confirmation specifically (not just any negation — "ich
# habe keine Versicherungsnummer" is not a denial of the accident).
_AREA_DENIAL_RE = re.compile(
    r"\b(?:nein|nö|nee|stimmt\s+nicht|nicht\s+richtig|falsch|"
    r"no|that'?s\s+(?:not\s+right|wrong)|incorrect)\b",
    re.IGNORECASE,
)
# Affirmation in reply to a read-back ("ja, stimmt", "korrekt", "passt").
_CONFIRM_YES_RE = re.compile(
    r"\b(?:ja|jawohl|genau|korrekt|stimmt|richtig|passt|yes|correct|right)\b",
    re.IGNORECASE,
)


def _extract_reference(text: str) -> str | None:
    """Pull an insurance/claim reference (≥5 alphanumerics, digit-bearing) from text.

    Negative replies ('keine', 'nicht', 'no') have no such token, so they yield
    None and nothing is stored.
    """
    tokens = _REF_TOKEN_RE.findall(text)
    kept: list[str] = []
    for i, tok in enumerate(tokens):
        if any(c.isdigit() for c in tok):
            kept.append(tok)
        elif kept:
            break  # a word after the number ends the reference
        elif (
            tok.isupper()
            and len(tok) <= 5
            and i + 1 < len(tokens)
            and any(c.isdigit() for c in tokens[i + 1])
        ):
            kept.append(tok)  # short prefix like 'VS' before the digits
    joined = "".join(kept)
    if len(re.sub(r"[^A-Za-z0-9]", "", joined)) >= 5:
        return joined
    return None


class ConversationManager:
    def __init__(self, call_id: str, lang: str = "de"):
        self.lang = lang
        self.state = ConversationState(call_id=call_id)
        self.state.messages = [{"role": "system", "content": get_system_prompt_base(lang)}]
        self._llm_context = None
        self._tools_builder: Callable[[list[str]], Any] | None = None

    def set_llm_context(self, context) -> None:
        self._llm_context = context

    def set_tools_builder(self, builder: Callable[[list[str]], Any]) -> None:
        self._tools_builder = builder

    def advance_phase(self) -> CallPhase:
        new_phase = next_phase(self.state)
        if new_phase != self.state.phase:
            old_phase = self.state.phase
            self.state.phase = new_phase
            self._update_llm_context()
            logger.info("Phase advanced: %s → %s", old_phase.value, new_phase.value)
        return self.state.phase

    def _update_llm_context(self) -> None:
        prompt = build_system_prompt(self.state, self.lang)
        if self.state.messages:
            self.state.messages[0]["content"] = prompt
        if self._llm_context and hasattr(self._llm_context, "messages"):
            ctx_messages = self._llm_context.messages
            if ctx_messages:
                ctx_messages[0]["content"] = prompt
            allowed = self.get_available_tools()
            if self._tools_builder and hasattr(self._llm_context, "set_tools"):
                self._llm_context.set_tools(self._tools_builder(allowed))

    def get_available_tools(self) -> list[str]:
        return PHASE_TOOLS.get(self.state.phase, [])

    def add_user_message(self, text: str) -> None:
        self.state.turn_count += 1
        self.state.messages.append({"role": "user", "content": text})
        was_callback = self.state.callback_requested
        if is_explicit_handoff_request(text, self.lang):
            self.state.callback_requested = True
            target = extract_target_person(text, self.lang)
            if target:
                self.state.target_person = target
        just_routed = False
        if self.state.caller_intent == CallerIntent.UNKNOWN:
            area_before = self.state.legal_area
            self._try_auto_route(text)
            just_routed = self.state.legal_area != area_before
        self.advance_phase()
        if not was_callback and self.state.callback_requested:
            self._update_llm_context()
        # Skip matter_type backfill on the routing turn so the area-confirmation
        # question still gets asked; capture it on later turns / handoffs instead.
        if not just_routed:
            self._try_capture_matter_type(text)
        self._try_confirm_readback(text)
        captured_insurance = self._try_capture_insurance(text)
        self._try_capture_contact(text, skip_phone=captured_insurance)

    def _try_confirm_readback(self, text: str) -> None:
        """Handle the caller's reply to a scripted phone read-back.

        'Ja, stimmt' confirms the number; a denial drops it so it is asked again
        (a re-stated number is then re-captured from the same turn).
        """
        if not self.state.callback_requested:
            return
        phone = self.state.entities.get("phone")
        if not phone or phone.confirmed:
            return
        lowered = text.lower()
        if _AREA_DENIAL_RE.search(lowered):
            del self.state.entities["phone"]
            self.advance_phase()
        elif _CONFIRM_YES_RE.search(lowered):
            self.confirm_entity("phone")

    def _recent_agent_text(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return str(msg["content"])
        return ""

    def _try_capture_insurance(self, text: str) -> bool:
        """Resolve the traffic insurance step once the agent has asked for it.

        Sets ``insurance_resolved`` so the qualification gate advances whether the
        caller gives a number or has none. Returns True if a number was stored, so
        phone capture stands down and won't mistake insurance digits for a phone.
        """
        if self.state.legal_area != LegalArea.TRAFFIC:
            return False
        if self.state.insurance_resolved or "insurance_number" in self.state.entities:
            return False
        if not _INSURANCE_ASK_RE.search(self._recent_agent_text()):
            return False
        value = _extract_reference(text)
        negative = bool(_NEGATE_RE.search(text.lower()))
        if not value and not negative:
            # Partial / unclear answer (e.g. "die lautet…"); wait for the number
            # rather than prematurely closing the step.
            return False
        if value:
            logger.info("Deterministic capture (LLM fallback): insurance_number=%r", value)
            self.update_and_confirm_entity("insurance_number", value)
        else:
            logger.info("Insurance step resolved: caller has no number")
        # Asked and answered → resolved either way, so the gate advances.
        self.state.insurance_resolved = True
        self.advance_phase()
        return bool(value)

    def _try_capture_matter_type(self, text: str) -> None:
        """Deterministic fallback for matter_type when the LLM skips the tool.

        Keeps the matter on record even when the caller front-loads everything
        (e.g. describes the issue and asks for a person in one breath).
        """
        if self.state.legal_area == LegalArea.UNKNOWN:
            return
        if "matter_type" in self.state.entities:
            return
        keyword_map = _MATTER_KEYWORDS.get(self.state.legal_area)
        if not keyword_map:
            return
        lowered = text.lower()
        # Don't capture when the caller denies the area-confirmation question.
        if _AREA_DENIAL_RE.search(lowered):
            return
        # Prefer the original complaint ("Ich hatte einen Unfall") over this turn's
        # words: it is the most reliable signal and avoids mis-reading a later
        # mention (e.g. "Versicherungsnummer") as the matter type.
        summary = (self.state.matter_summary or "").lower()
        for source in (summary, lowered):
            for value, keywords in keyword_map:
                if any(kw in source for kw in keywords):
                    self._store_matter_type(value)
                    return

    def _store_matter_type(self, value: str) -> None:
        logger.info("Deterministic capture (LLM fallback): matter_type=%r", value)
        self.update_and_confirm_entity("matter_type", value)
        self._update_llm_context()

    def _extract_name(self, text: str) -> str | None:
        """Get the caller's name from a trigger phrase, or from a bare reply when
        the agent just asked for it (e.g. 'Daniel Stein')."""
        trigger = _NAME_TRIGGER_RE.search(text)
        if trigger:
            match = _CAPWORDS_RE.match(text[trigger.end() :].strip())
            return match.group(1).strip() if match else None
        if _NAME_ASK_RE.search(self._recent_agent_text()):
            match = _CAPWORDS_RE.match(text.strip())
            if match and match.group(1).split()[0].lower() not in _NAME_STOPWORDS:
                return match.group(1).strip()
        return None

    def _try_capture_contact(self, text: str, skip_phone: bool = False) -> None:
        """Deterministic fallback for name/phone when the LLM skips the tool.

        Runs only while collecting contact details. Guarantees the data reaches
        the database even if capture_caller_details is never called. ``skip_phone``
        is set when the digits in this turn are an insurance number, not a phone.
        """
        if self.state.phase not in (CallPhase.CAPTURE, CallPhase.ESCALATION):
            return

        captured = False

        existing_name = self.state.entities.get("name")
        if not (existing_name and existing_name.confirmed):
            name = self._extract_name(text)
            if name:
                logger.info("Deterministic capture (LLM fallback): name=%r", name)
                self.update_and_confirm_entity("name", name)
                captured = True

        existing_phone = self.state.entities.get("phone")
        if not skip_phone and not (existing_phone and existing_phone.confirmed):
            normalized = normalize_phone_text(text)
            if len(re.sub(r"\D", "", normalized)) >= 7:
                logger.info("Deterministic capture (LLM fallback): phone=%r", normalized)
                self.store_entity("phone", normalized, 0.9)
                captured = True

        if captured:
            self._update_llm_context()

    def _try_auto_route(self, text: str) -> None:
        """Keyword fallback when the LLM skips route_call."""
        t = text.lower()
        scores: dict[LegalArea, int] = {
            LegalArea.EMPLOYMENT: 0,
            LegalArea.TENANCY: 0,
            LegalArea.TRAFFIC: 0,
        }
        employment_kw = [
            "kündigung",
            "gekündigt",
            "arbeitgeber",
            "abfindung",
            "abmahnung",
            "lohn",
            "gehalt",
            "arbeitsvertrag",
            "mobbing",
            "dismissal",
            "employer",
            "wages",
            "redundancy",
            "fired",
        ]
        tenancy_kw = [
            "vermieter",
            "miete",
            "kaution",
            "wohnung",
            "mängel",
            "nebenkosten",
            "mietvertrag",
            "räumung",
            "mieterhöhung",
            "landlord",
            "rent",
            "deposit",
            "tenant",
            "eviction",
            "flat",
        ]
        traffic_kw = [
            "unfall",
            "auto",
            "kfz",
            "schaden",
            "versicherung",
            "verkehr",
            "fahrzeug",
            "polizei",
            "auffahrunfall",
            "accident",
            "car",
            "vehicle",
            "damage",
            "insurance",
            "crash",
        ]
        for kw in employment_kw:
            if kw in t:
                scores[LegalArea.EMPLOYMENT] += 1
        for kw in tenancy_kw:
            if kw in t:
                scores[LegalArea.TENANCY] += 1
        for kw in traffic_kw:
            if kw in t:
                scores[LegalArea.TRAFFIC] += 1
        matches = [(area, s) for area, s in scores.items() if s > 0]
        if len(matches) == 1:
            area = matches[0][0]
            logger.info("Auto-route: %s (keyword match)", area.value)
            self.state.caller_intent = CallerIntent.BOOK_CONSULTATION
            self.state.legal_area = area
            self.state.matter_summary = text

    def set_transcription_confidence(self, confidence: float | None) -> None:
        self.state.last_transcription_confidence = confidence

    def add_assistant_message(self, text: str) -> None:
        self.state.messages.append({"role": "assistant", "content": text})

    def add_tool_call(self, assistant_text: str, tool_calls: list[dict]) -> None:
        self.state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": tool_calls,
            }
        )

    def add_tool_result(self, tool_call_id: str, name: str, result: dict) -> None:
        self.state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": json.dumps(result),
            }
        )

    def get_messages(self) -> list[dict]:
        return self.state.messages

    def set_route(self, intent: CallerIntent, area: LegalArea, summary: str | None = None) -> None:
        self.state.caller_intent = intent
        self.state.legal_area = area
        if summary:
            self.state.matter_summary = summary
        self.reset_misunderstanding_streak()
        self.advance_phase()

    def store_entity(self, field_name: str, value: str, confidence: float) -> ExtractedEntity:
        entity = ExtractedEntity(
            field_name=field_name,
            value=value,
            confidence=confidence,
            source_turn=self.state.turn_count,
        )
        self.state.entities[field_name] = entity
        self.advance_phase()
        return entity

    def confirm_entity(self, field_name: str) -> None:
        if field_name in self.state.entities:
            self.state.entities[field_name].confirmed = True
            self.advance_phase()

    def update_and_confirm_entity(self, field_name: str, value: str) -> None:
        entity = self.state.entities.get(field_name)
        if entity:
            entity.value = value
            entity.confirmed = True
            entity.source_turn = self.state.turn_count
        else:
            entity = ExtractedEntity(
                field_name=field_name,
                value=value,
                confidence=1.0,
                confirmed=True,
                source_turn=self.state.turn_count,
            )
            self.state.entities[field_name] = entity
        self.advance_phase()

    def record_misunderstanding(self) -> None:
        self.state.misunderstanding_streak += 1
        logger.info("Misunderstanding streak: %d", self.state.misunderstanding_streak)
        self.advance_phase()

    def request_handoff(self, reason: str, summary: str = "") -> None:
        if reason == "caller_requested_human":
            self.state.callback_requested = True
            self.state.escalation_summary = summary
        else:
            self.state.escalation_requested = True
            self.state.escalation_reason = reason
            self.state.escalation_summary = summary
        self.advance_phase()

    def reset_misunderstanding_streak(self) -> None:
        if self.state.misunderstanding_streak > 0:
            self.state.misunderstanding_streak = 0
