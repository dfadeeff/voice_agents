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
from app.conversation.script import compute_prompt
from app.conversation.state import ConversationState
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea

# STT confidence below this stores a contact field unconfirmed, forcing a
# read-back (the "double-check when unsure" path).
LOW_CONFIDENCE_THRESHOLD = 0.75

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
_EMAIL_VALID_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$")

# Slot selection (deterministic booking): ordinal words → index, or a clock time.
_ORDINAL = {
    "erste": 0,
    "erster": 0,
    "ersten": 0,
    "first": 0,
    "zweite": 1,
    "zweiter": 1,
    "zweiten": 1,
    "second": 1,
    "dritte": 2,
    "dritter": 2,
    "dritten": 2,
    "third": 2,
}
_SLOT_TIME_RE = re.compile(
    r"\b(\d{1,2})\s*(?:uhr|o'?clock)\s*(\d{1,2})?\b"
    r"|\b(\d{1,2})[:.]\s*(\d{2})\s*(?:uhr|o'?clock)?\b",
    re.IGNORECASE,
)
_SLOT_DECLINE_RE = re.compile(
    r"\b(?:nein|nö|nee|anders|andere[rn]?|später|spaeter|nichts\s+dabei|"
    r"no|none|other|later|something\s+else)\b|pass\w*\s+(?:mir\s+)?nicht|"
    r"doesn'?t\s+work|don'?t\s+work",
    re.IGNORECASE,
)


def _match_slot_choice(text: str, slots: list[dict]) -> dict | None:
    """Map a caller reply ('die erste', '10 Uhr') to one of the offered slots."""
    if not slots:
        return None
    low = text.lower()
    for word, idx in _ORDINAL.items():
        if re.search(rf"\b{word}\b", low) and idx < len(slots):
            return slots[idx]
    m = re.search(r"\boption\s*(\d)\b", low)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(slots):
            return slots[idx]
    tm = _SLOT_TIME_RE.search(low)
    if tm:
        if tm.group(1) is not None:
            hour = int(tm.group(1))
            minute = tm.group(2)
        else:
            hour = int(tm.group(3))
            minute = tm.group(4)
        if minute is not None:
            minute = str(int(minute)).zfill(2)
        for slot in slots:
            sh, _, sm = slot["time"].partition(":")
            if int(sh) == hour and (minute is None or sm == minute):
                return slot
    if len(slots) == 1 and _CONFIRM_YES_RE.search(low):
        return slots[0]
    return None


_DOMAIN_CORRECTIONS = {
    "smail": "gmail",
    "gmeil": "gmail",
    "g-mail": "gmail",
    "geemail": "gmail",
    "gemail": "gmail",
    "hotmeil": "hotmail",
    "hotemail": "hotmail",
}


# Spoken lead-in before the address proper ("meine E-Mail-Adresse lautet …"),
# stripped so it isn't glued onto the local part.
_EMAIL_LEADIN_RE = re.compile(
    r"^(?:\s*\b(?:ja|nein|also|genau|ähm|äh|meine?|die|das|ich|e[-\s]?mail|email|mail|"
    r"adresse|lautet|ist|wäre|my|the|email|address|it'?s|is)\b[\s.:,!?-]*)+",
    re.IGNORECASE,
)


def _parse_email(text: str) -> str | None:
    """Convert a spoken email to an address.

    Spoken emails arrive in chunks with stray spaces ("Lang M at gmail punkt com")
    and the local part is often dictated piece by piece. We strip a spoken lead-in,
    map connectors to symbols, then join the remaining whitespace so a
    space-separated local part ("lang m") becomes one token ("langm") instead of
    being truncated to whichever fragment happened to carry the '@'.
    """
    t = _EMAIL_LEADIN_RE.sub("", text.lower().strip())
    t = re.sub(r"\s*(?:\bat\b|\bät\b|@)\s*", "@", t)
    t = re.sub(r"\s*(?:\bpunkt\b|\bdot\b|\bpoint\b)\s*", ".", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    t = re.sub(r"\s+", "", t).strip(".,;:!?")
    t = t.replace("@www.", "@")
    if "@" not in t:
        return None
    local, _, domain = t.partition("@")
    parts = [p for p in domain.split(".") if p]
    if local and parts:
        parts[0] = _DOMAIN_CORRECTIONS.get(parts[0], parts[0])
    candidate = f"{local}@{'.'.join(parts)}"
    return candidate if _EMAIL_VALID_RE.match(candidate) else None


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
    def __init__(self, call_id: str, lang: str = "de", calendar=None):
        self.lang = lang
        self.state = ConversationState(call_id=call_id)
        self.state.messages = [{"role": "system", "content": get_system_prompt_base(lang)}]
        self._llm_context = None
        self._tools_builder: Callable[[list[str]], Any] | None = None
        # Optional CalendarService for deterministic in-code booking (sync access).
        self._calendar = calendar

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

    def next_prompt(self) -> str | None:
        """Deterministic next question for the scripted spine; sets state.awaiting.

        Returns the line to speak (the LLM is then skipped for this turn), or None
        to let the LLM drive (ROUTING / INFORMATION). Called by the pipeline after
        add_user_message on each turn. The reply parser dispatches on awaiting.
        """
        line, awaiting = compute_prompt(self.state, self.lang)
        self.state.awaiting = awaiting
        return line

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

        # Reply parsing dispatches on what the last scripted question asked for
        # (state.awaiting), not on regex over the agent's own prior sentence.
        awaiting = self.state.awaiting

        # The caller denies the area-confirmation ("nein, es geht um etwas
        # anderes") → re-route instead of recording a matter type.
        if awaiting == "matter_type" and _AREA_DENIAL_RE.search(text.lower()):
            self._reroute_after_denial(text)
        elif not just_routed:
            # Opportunistic: keyword-map the matter type from this turn / the
            # original complaint. Skipped on the routing turn so the area-
            # confirmation question still gets asked.
            self._try_capture_matter_type(text)

        self._try_capture_matter_details(text)
        self._try_confirm_readback(text)
        captured_insurance = self._try_capture_insurance(text)
        self._try_capture_contact(text, skip_phone=captured_insurance)
        self._try_skip_email(text)
        self._handle_email_attempt(awaiting)
        self._try_capture_preferred_time(text)
        self._try_book(text)
        self._persist()

    def _reroute_after_denial(self, text: str) -> None:
        """Caller rejected the area-confirmation; try to re-route, else hand back
        to the LLM routing step."""
        self.state.caller_intent = CallerIntent.UNKNOWN
        self._try_auto_route(text)
        if self.state.caller_intent == CallerIntent.UNKNOWN:
            self.state.legal_area = LegalArea.UNKNOWN
        self.advance_phase()

    def _try_capture_matter_details(self, text: str) -> None:
        """Store the free-text matter detail when it was the awaited reply
        (employment/tenancy follow-up; traffic uses the insurance step instead)."""
        if self.state.awaiting != "matter_details":
            return
        if "matter_details" in self.state.entities:
            return
        self.update_and_confirm_entity("matter_details", text.strip())
        self._update_llm_context()

    def _persist(self) -> None:
        """Write the caller row immediately after each turn so a dropped call
        never loses captured data (incremental, authoritative-at-decision-time)."""
        if self._calendar is None:
            return
        ents = self.state.entities

        def val(field: str) -> str:
            entity = ents.get(field)
            return entity.value if entity else ""

        if not (val("name") or val("phone")):
            return  # nothing material captured yet
        outcome = self.state.phase.value
        if self.state.booking_confirmed:
            outcome = "booked"
        elif self.state.callback_requested:
            outcome = "callback"
        elif self.state.escalation_requested:
            outcome = "escalation"
        self._calendar.upsert_caller_sync(
            call_id=self.state.call_id,
            name=val("name"),
            phone=val("phone"),
            email=val("email"),
            legal_area=self.state.legal_area.value,
            matter_type=val("matter_type"),
            matter_summary=self.state.matter_summary or "",
            case_reference=val("case_reference"),
            insurance_number=val("insurance_number"),
            outcome=outcome,
            preferred_time=self.state.preferred_time or "",
        )

    def _try_book(self, text: str) -> None:
        """Deterministic booking: offer slots, match the caller's choice, book it.

        Replaces the LLM's check_availability/book_consultation turns so booking is
        reliable. Declining offers the next batch (the unavailable-slot path).
        """
        if self._calendar is None or self.state.callback_requested:
            return
        if self.state.phase != CallPhase.BOOKING or self.state.booking_confirmed:
            return
        area = self.state.legal_area.value

        if not self.state.offered_slots:
            self.state.offered_slots = self._calendar.available_slots_sync(area, limit=3)
            self._update_llm_context()
            return

        choice = _match_slot_choice(text, self.state.offered_slots)
        if choice:
            ents = self.state.entities
            booking = self._calendar.book_slot_sync(
                slot_id=choice["id"],
                call_id=self.state.call_id,
                caller_name=ents["name"].value if "name" in ents else "",
                caller_email=ents["email"].value if "email" in ents else "",
                caller_phone=ents["phone"].value if "phone" in ents else "",
                matter_type=ents["matter_type"].value if "matter_type" in ents else area,
            )
            if booking:
                logger.info("Deterministic booking: slot %s booked", choice["id"])
                self.state.booking_confirmed = True
                self.state.booked_slot = booking
                self.advance_phase()
                return
            # Slot was taken between offer and booking — drop it and re-offer.
            self.state.offered_slot_ids.append(choice["id"])
        elif _SLOT_DECLINE_RE.search(text.lower()):
            self.state.offered_slot_ids.extend(s["id"] for s in self.state.offered_slots)
        else:
            return  # unrecognised reply → re-present the same offer

        self.state.offered_slots = self._calendar.available_slots_sync(
            area, exclude_ids=self.state.offered_slot_ids, limit=3
        )
        self._update_llm_context()

    def _try_skip_email(self, text: str) -> None:
        """Mark email as skipped when we asked for it and the caller has none."""
        if self.state.callback_requested or self.state.email_skipped:
            return
        if self.state.awaiting != "email":
            return
        if "email" in self.state.entities:
            return
        if not _NEGATE_RE.search(text.lower()):
            return
        # "Nein, die Domäne ist hotmail" is a correction, not a skip.
        if re.search(
            r"\b(?:dom[äa]ne|domain|gmail|hotmail|yahoo|outlook|web\.de|gmx|@)\b",
            text,
            re.IGNORECASE,
        ):
            return
        logger.info("Email skipped: caller has none")
        self.state.email_skipped = True
        self.advance_phase()

    _MAX_EMAIL_ATTEMPTS = 3

    def _handle_email_attempt(self, awaiting: str | None) -> None:
        """Track failed email attempts; re-ask, then skip after a few misses.

        Spoken email over phone-quality audio is the hardest field. Rather than
        loop forever, after a few attempts we mark it skipped — a phone number is
        enough to book or call back."""
        if awaiting != "email":
            return
        if "email" in self.state.entities or self.state.email_skipped:
            self.state.email_misheard = False
            return
        self.state.email_attempts += 1
        if self.state.email_attempts >= self._MAX_EMAIL_ATTEMPTS:
            logger.info("Email skipped after %d failed attempts", self.state.email_attempts)
            self.state.email_skipped = True
            self.state.email_misheard = False
            self.advance_phase()
        else:
            self.state.email_misheard = True

    def _try_capture_preferred_time(self, text: str) -> None:
        """Store the caller's preferred callback time once it has been asked."""
        if self.state.awaiting != "callback_time" or self.state.preferred_time:
            return
        self.state.preferred_time = text.strip()
        logger.info("Deterministic capture: preferred_time=%r", self.state.preferred_time)
        self.advance_phase()

    _CONFIRM_FIELDS = {"email_confirm": "email", "phone_confirm": "phone", "name_confirm": "name"}

    def _try_confirm_readback(self, text: str) -> None:
        """Handle the caller's reply to a scripted name/email/phone read-back.

        'Ja, stimmt' confirms the awaited field; a denial drops it so it is asked
        again (a re-stated value is then re-captured from the same turn).
        """
        field = self._CONFIRM_FIELDS.get(self.state.awaiting or "")
        if not field:
            return
        entity = self.state.entities.get(field)
        if not entity or entity.confirmed:
            return
        lowered = text.lower()
        if _AREA_DENIAL_RE.search(lowered):
            del self.state.entities[field]
            self.advance_phase()
        elif _CONFIRM_YES_RE.search(lowered):
            self.confirm_entity(field)

    def _try_capture_insurance(self, text: str) -> bool:
        """Resolve the traffic insurance step once we have asked for it.

        Sets ``insurance_resolved`` so the qualification gate advances whether the
        caller gives a number or has none. Returns True if a number was stored, so
        phone capture stands down and won't mistake insurance digits for a phone.
        """
        if self.state.awaiting != "insurance":
            return False
        if self.state.insurance_resolved or "insurance_number" in self.state.entities:
            return False
        value = _extract_reference(text)
        negative = bool(_NEGATE_RE.search(text.lower()))
        if not value and not negative:
            return False
        # "keine Schadensnummer, dafür aber Versicherungsnummer" — caller negates
        # one type but says they have another. Don't resolve; wait for the digits.
        if (
            not value
            and negative
            and re.search(r"\b(?:dafür|aber|habe|have)\b", text, re.IGNORECASE)
        ):
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
        we just asked for the name (e.g. 'Daniel Stein')."""
        trigger = _NAME_TRIGGER_RE.search(text)
        if trigger:
            match = _CAPWORDS_RE.match(text[trigger.end() :].strip())
            return match.group(1).strip() if match else None
        if self.state.awaiting == "name":
            match = _CAPWORDS_RE.match(text.strip())
            if match and match.group(1).split()[0].lower() not in _NAME_STOPWORDS:
                return match.group(1).strip()
        return None

    def _store_name(self, name: str) -> None:
        """Store the name, leaving it unconfirmed (forcing a read-back) when the
        STT confidence for this turn was low — the 'double-check when unsure' path."""
        conf = self.state.last_transcription_confidence
        if conf is not None and conf < LOW_CONFIDENCE_THRESHOLD:
            logger.info("Deterministic capture: name=%r (low conf %.2f → confirm)", name, conf)
            self.store_entity("name", name, conf)
        else:
            logger.info("Deterministic capture: name=%r", name)
            self.update_and_confirm_entity("name", name)

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
                self._store_name(name)
                captured = True

        existing_email = self.state.entities.get("email")
        if not (existing_email and existing_email.confirmed):
            email = _parse_email(text)
            if email:
                logger.info("Deterministic capture (LLM fallback): email=%r", email)
                self.store_entity("email", email, 0.9)
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
