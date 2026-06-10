from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from app.conversation.cues import (
    AREA_DENIAL_RE,
    CONFIRM_YES_RE,
    OFFSCRIPT_QUESTION_RE,
    REPEAT_REQUEST_RE,
)
from app.conversation.email_capture import EmailCapture
from app.conversation.flow import PHASE_TOOLS, next_phase
from app.conversation.insurance_capture import InsuranceCapture
from app.conversation.phone_capture import PhoneCapture
from app.conversation.policy import (
    extract_target_person,
    is_explicit_handoff_request,
    wants_callback,
)
from app.conversation.prompts import build_system_prompt, get_system_prompt_base
from app.conversation.script import compute_prompt
from app.conversation.state import ConversationState
from app.conversation.time_parse import parse_requested_time
from app.models.schemas import CallerIntent, CallPhase, ExtractedEntity, LegalArea
from app.validation import LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallerRecord:
    """Immutable snapshot of captured caller data + terminal outcome for the DB.

    The manager owns the conversation state; persistence callers (per-turn
    `_persist`, end-of-call `save_caller`) take this snapshot instead of reaching
    into `state.entities` and recomputing the outcome themselves. Field names
    match `CalendarService.upsert_caller_sync`, so it upserts via `**asdict(rec)`.
    """

    call_id: str
    name: str
    phone: str
    email: str
    legal_area: str
    matter_type: str
    matter_summary: str
    case_reference: str
    insurance_number: str
    outcome: str
    preferred_time: str


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
        # The agent's tenancy question offers "Kündigung der Wohnung", so a lease
        # termination must map to eviction here (not employment — area is tenancy).
        ("eviction", ("räumung", "raeumung", "kündigung", "gekündigt", "eviction", "evict")),
        ("deposit", ("kaution", "deposit")),
        ("rent_increase", ("mieterhöhung", "mieterhoehung", "rent increase")),
        ("repairs", ("mängel", "maengel", "reparatur", "schimmel", "repairs", "mould", "mold")),
    ),
}

# Maps a disambiguation reply ("Mietrecht" / "Wohnung") to a legal area. Includes
# the spoken area names so the reply resolves even when the opening was multi-area
# (e.g. "Mietvertrag gekündigt" → tenancy + employment).
_AREA_DISAMBIG_KEYWORDS: dict[LegalArea, tuple[str, ...]] = {
    LegalArea.EMPLOYMENT: ("arbeitsrecht", "arbeit", "arbeitgeber", "job", "employment", "work"),
    LegalArea.TENANCY: (
        "mietrecht",
        "miete",
        "mietvertrag",
        "wohnung",
        "vermieter",
        "kaution",
        "tenancy",
        "rent",
        "landlord",
        "flat",
    ),
    LegalArea.TRAFFIC: (
        "verkehrsrecht",
        "verkehr",
        "unfall",
        "auto",
        "kfz",
        "fahrzeug",
        "traffic",
        "accident",
        "car",
        "vehicle",
    ),
}

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
    if len(slots) == 1 and CONFIRM_YES_RE.search(low):
        return slots[0]
    return None


class ConversationManager:
    def __init__(self, call_id: str, lang: str = "de", calendar=None):
        self.lang = lang
        self.state = ConversationState(call_id=call_id)
        # Per-field capture strategies: each owns its parse → accumulate →
        # read-back → confirm policy; the manager dispatches and keeps state.
        self._insurance = InsuranceCapture(self)
        self._email = EmailCapture(self)
        self._phone = PhoneCapture(self)
        self.state.messages = [{"role": "system", "content": get_system_prompt_base(lang)}]
        self._llm_context = None
        self._tools_builder: Callable[[list[str]], Any] | None = None
        # Optional async LLM rescue for matter-type; None = keyword-only (local default).
        self._matter_classifier: Callable[[str, str], Any] | None = None
        # Optional CalendarService for deterministic in-code booking (sync access).
        self._calendar = calendar

    def set_llm_context(self, context) -> None:
        self._llm_context = context

    def set_tools_builder(self, builder: Callable[[list[str]], Any]) -> None:
        self._tools_builder = builder

    def set_email_extractor(self, extractor: Callable[[str], Any] | None) -> None:
        """Optional async LLM rescue for spoken email (regex stays the default)."""
        self._email.set_extractor(extractor)

    def set_matter_classifier(self, classifier: Callable[[str, str], Any] | None) -> None:
        """Optional async LLM rescue for matter-type (keyword match stays default)."""
        self._matter_classifier = classifier

    async def resolve_matter_if_pending(self, text: str) -> None:
        """LLM rescue when we asked for the matter type but keywords couldn't map it.

        Runs after add_user_message (keywords had first go). Only fires when the
        matter type is still unknown or was filled with the "other" fallback, and
        upgrades it to a specific label — so a free-phrased answer like "ich wurde
        aus meiner Wohnung geworfen" still classifies as eviction.
        """
        classifier = getattr(self, "_matter_classifier", None)
        if classifier is None or self.state.awaiting != "matter_type":
            return
        existing = self.state.entities.get("matter_type")
        if existing and existing.value != "other":
            return  # keywords already classified specifically
        try:
            label = await classifier(self.state.legal_area.value, text)
        except Exception as e:
            logger.warning("Matter classifier raised: %s", e)
            return
        if not label or (existing and label == "other"):
            return
        logger.info("LLM matter classification: %r → %s", text, label)
        self.state.matter_attempts = 0
        self._store_matter_type(label)
        self.persist_progress()

    async def resolve_email_if_pending(self, text: str) -> None:
        """LLM rescue for a spoken email the regex couldn't parse (see EmailCapture)."""
        if self.state.offscript_question:
            return  # a side question, not a mangled email — nothing to rescue
        await self._email.resolve_if_pending(text)

    def advance_phase(self) -> CallPhase:
        new_phase = next_phase(self.state)
        if new_phase != self.state.phase:
            old_phase = self.state.phase
            self.state.phase = new_phase
            self.refresh_llm_context()
            logger.info("Phase advanced: %s → %s", old_phase.value, new_phase.value)
        return self.state.phase

    def next_prompt(self) -> str | None:
        """Deterministic next question for the scripted spine; sets state.awaiting.

        Returns the line to speak (the LLM is then skipped for this turn), or None
        to let the LLM drive (ROUTING / INFORMATION). Called by the pipeline after
        add_user_message on each turn. The reply parser dispatches on awaiting.
        """
        if self.state.offscript_question:
            # Side question during a scripted slot: keep awaiting unchanged and
            # yield this one turn to the LLM, whose prompt instructs it to answer
            # briefly and re-issue the ask. The spine resumes on the next reply.
            return None
        line, awaiting = compute_prompt(self.state, self.lang)
        self.state.awaiting = awaiting
        return line

    def refresh_llm_context(self) -> None:
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
        # One-shot: a previous turn's side question was answered by the LLM; this
        # reply is back on script.
        self.state.offscript_question = False
        was_callback = self.state.callback_requested
        # An explicit "call me back" → callback request. A request to speak to a
        # (named) lawyer → book a consultation with them (not a callback): record
        # the target person and commit to booking; routing then asks the matter.
        if wants_callback(text, self.lang):
            self.state.callback_requested = True
            target = extract_target_person(text, self.lang)
            if target:
                self.state.target_person = target
        elif is_explicit_handoff_request(text, self.lang):
            target = extract_target_person(text, self.lang)
            if target:
                self.state.target_person = target
            if self.state.caller_intent == CallerIntent.UNKNOWN:
                self.state.caller_intent = CallerIntent.BOOK_CONSULTATION
        just_routed = False
        # Keyword-route while the legal area is still unknown — including the turn
        # after a person request, where intent is already 'book' but the area isn't
        # yet known (so gating on intent would skip routing the matter).
        if self.state.legal_area == LegalArea.UNKNOWN:
            self._try_auto_route(text)
            just_routed = self.state.legal_area != LegalArea.UNKNOWN
        self.advance_phase()
        if not was_callback and self.state.callback_requested:
            self.refresh_llm_context()

        # Reply parsing dispatches on what the last scripted question asked for
        # (state.awaiting), not on regex over the agent's own prior sentence.
        awaiting = self.state.awaiting

        # A side question instead of the asked-for datum ("Warum brauchen Sie
        # meine E-Mail?") → hand this one turn to the LLM (answer briefly,
        # re-ask) WITHOUT running the capture handlers, so it never burns an
        # attempt or gets answered with "ich habe die E-Mail nicht verstanden".
        if self._is_offscript_question(text, awaiting):
            logger.info("Off-script question while awaiting %r: %r", awaiting, text)
            self.state.offscript_question = True
            self.state.offscript_attempts += 1
            self.refresh_llm_context()
            self.persist_progress()
            return
        self.state.offscript_attempts = 0

        # The caller is answering "which area?" after an ambiguous opening.
        if awaiting == "area":
            self._handle_area_choice(text)
            if self.state.legal_area != LegalArea.UNKNOWN:
                just_routed = True

        # The caller denies the area-confirmation ("nein, es geht um etwas
        # anderes") → re-route instead of recording a matter type.
        if awaiting == "matter_type" and AREA_DENIAL_RE.search(text.lower()):
            self._reroute_after_denial(text)
        elif not just_routed:
            # Opportunistic: keyword-map the matter type from this turn / the
            # original complaint. Skipped on the routing turn so the area-
            # confirmation question still gets asked.
            self._try_capture_matter_type(text)
            # Loop guard: if we asked for the matter type and still couldn't
            # recognise the answer, record it as "other" after a couple of tries
            # rather than re-asking the same question forever.
            if awaiting == "matter_type" and "matter_type" not in self.state.entities:
                self.state.matter_attempts += 1
                if self.state.matter_attempts >= 2:
                    logger.info("matter_type unresolved → recording 'other'")
                    self._store_matter_type("other")

        self._try_capture_matter_details(text)
        self._try_confirm_readback(text)
        captured_insurance = self._insurance.handle(text)
        self._try_capture_contact(text, skip_phone=captured_insurance)
        self._phone.handle(text)
        self._email.try_capture_spelled(text)
        self._email.try_skip(text)
        self._email.track_attempt(awaiting)
        self._try_capture_preferred_time(text)
        self._try_book(text)
        self.persist_progress()

    # Scripted slots where a side question should divert to the LLM instead of
    # being miscounted as a failed answer. Excludes slot/area/matter turns, whose
    # natural replies often look like questions ("Haben Sie was am Donnerstag?").
    _OFFSCRIPT_AWAITING = frozenset(
        {
            "name",
            "name_confirm",
            "email",
            "email_confirm",
            "email_spell",
            "phone",
            "phone_confirm",
            "insurance",
            "insurance_confirm",
            "callback_time",
        }
    )

    # Consecutive side-question diversions before the turn is processed on-script
    # again, so a caller who only asks questions still hits the per-field attempt
    # caps instead of looping forever.
    _MAX_OFFSCRIPT_ATTEMPTS = 2

    def _is_offscript_question(self, text: str, awaiting: str | None) -> bool:
        """True when the reply to a scripted data ask is a side question/objection
        rather than an answer ("Warum brauchen Sie meine E-Mail?")."""
        if awaiting not in self._OFFSCRIPT_AWAITING:
            return False
        if self.state.offscript_attempts >= self._MAX_OFFSCRIPT_ATTEMPTS:
            return False
        # Anything carrying answer payload is an answer, question mark or not
        # (digits while dictating, '@'/'punkt' while giving an email).
        if re.search(r"[\d@]", text) or re.search(r"\bpunkt\b", text, re.IGNORECASE):
            return False
        # "Wie bitte?" asks for the question again — the scripted re-ask (with its
        # bounded attempts) is the right response, not an LLM digression.
        if REPEAT_REQUEST_RE.search(text):
            return False
        # A questioning confirmation ("ja, stimmt?") is still a confirmation.
        if awaiting.endswith("_confirm") and (
            CONFIRM_YES_RE.search(text.lower()) or AREA_DENIAL_RE.search(text.lower())
        ):
            return False
        return bool(OFFSCRIPT_QUESTION_RE.search(text))

    def _handle_area_choice(self, text: str) -> None:
        """Resolve the legal area from the caller's disambiguation reply.

        Restricted to the candidate areas when known, so "Mietrecht" / "Wohnung"
        reliably maps to tenancy even though the opening also mentioned a
        Kündigung. Keeps the call moving deterministically out of ROUTING.
        """
        if self.state.legal_area != LegalArea.UNKNOWN:
            self.state.area_options = []  # already routed (e.g. by keyword) this turn
            return
        options = [LegalArea(v) for v in self.state.area_options] or [
            LegalArea.EMPLOYMENT,
            LegalArea.TENANCY,
            LegalArea.TRAFFIC,
        ]
        low = text.lower()
        picked = {a for a in options if any(kw in low for kw in _AREA_DISAMBIG_KEYWORDS[a])}
        if len(picked) == 1:
            area = picked.pop()
            self.state.legal_area = area
            self.state.caller_intent = CallerIntent.BOOK_CONSULTATION
            self.state.area_options = []
            logger.info("Disambiguated area: %s", area.value)
            self.advance_phase()

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
        self.refresh_llm_context()

    def _outcome(self) -> str:
        """Terminal call outcome for persistence (single source of truth, used by
        both the per-turn write and the end-of-call save)."""
        if self.state.booking_confirmed:
            return "booked"
        if self.state.callback_requested:
            return "callback"
        if self.state.escalation_requested:
            return "escalation"
        return self.state.phase.value

    def snapshot_for_persistence(self) -> CallerRecord | None:
        """Snapshot caller data + outcome for the DB, or None when nothing material
        has been captured yet. Lets persistence callers avoid walking `state`."""
        ents = self.state.entities

        def val(field: str) -> str:
            entity = ents.get(field)
            return entity.value if entity else ""

        if not (val("name") or val("phone")):
            return None
        return CallerRecord(
            call_id=self.state.call_id,
            name=val("name"),
            phone=val("phone"),
            email=val("email"),
            legal_area=self.state.legal_area.value,
            matter_type=val("matter_type"),
            matter_summary=self.state.matter_summary or "",
            case_reference=val("case_reference"),
            insurance_number=val("insurance_number"),
            outcome=self._outcome(),
            preferred_time=self.state.preferred_time or "",
        )

    def persist_progress(self) -> None:
        """Write the caller row immediately after each turn so a dropped call
        never loses captured data (incremental, authoritative-at-decision-time)."""
        if self._calendar is None:
            return
        record = self.snapshot_for_persistence()
        if record is not None:
            self._calendar.upsert_caller_sync(**asdict(record))

    def _try_book(self, text: str) -> None:
        """Deterministic booking: offer slots, match the caller's choice, book it.

        Replaces the LLM's check_availability/book_consultation turns so booking is
        reliable. Declining offers the next batch (the unavailable-slot path).
        """
        if self._calendar is None or self.state.callback_requested:
            return
        if self.state.phase != CallPhase.BOOKING or self.state.booking_confirmed:
            return
        # One-shot apology flag: cleared each turn, re-set below if the caller's
        # requested time is unavailable (so the apology shows for exactly one offer).
        self.state.unavailable_time = None
        area = self.state.legal_area.value

        if not self.state.offered_slots:
            self.state.offered_slots = self._available_slots(area)
            self.refresh_llm_context()
            return

        choice = _match_slot_choice(text, self.state.offered_slots)
        if choice:
            if self._book_slot(choice):
                return
            # Slot was taken between offer and booking — drop it and re-offer.
            self.state.offered_slot_ids.append(choice["id"])
        elif _SLOT_DECLINE_RE.search(text.lower()):
            # Caller rejected these times → exclude the whole (date, time) pairs so
            # the next batch is genuinely different (not the same time, other lawyer).
            self.state.declined_slot_times.extend(
                f"{s['date']} {s['time']}" for s in self.state.offered_slots
            )
        elif (requested := parse_requested_time(text)) is not None:
            # Caller named a specific time that wasn't offered ("Können wir 13 Uhr
            # machen?"). Honour it if free; otherwise apologise and re-offer.
            slot = self._calendar.find_slot_sync(requested, area, self._requested_lawyer_surname())
            if slot and self._book_slot(slot):
                return
            self.state.unavailable_time = requested
        else:
            return  # unrecognised reply → re-present the same offer

        self.state.offered_slots = self._available_slots(area)
        self.refresh_llm_context()

    def _book_slot(self, slot: dict) -> bool:
        """Book a calendar slot and advance to confirmation. Returns False if the
        slot was taken between offer and booking (caller should be re-offered)."""
        ents = self.state.entities
        booking = self._calendar.book_slot_sync(
            slot_id=slot["id"],
            call_id=self.state.call_id,
            caller_name=ents["name"].value if "name" in ents else "",
            caller_email=ents["email"].value if "email" in ents else "",
            caller_phone=ents["phone"].value if "phone" in ents else "",
            matter_type=ents["matter_type"].value
            if "matter_type" in ents
            else self.state.legal_area.value,
        )
        if not booking:
            return False
        logger.info("Deterministic booking: slot %s booked", slot["id"])
        self.confirm_booking(booking)
        return True

    def _requested_lawyer_surname(self) -> str:
        """Surname of the lawyer the caller asked for, e.g. 'Frau Hoffmann' → 'Hoffmann'."""
        person = self.state.target_person
        return person.split()[-1] if person else ""

    def _available_slots(self, area: str) -> list[dict]:
        """Offer the requested lawyer's slots when named; fall back to any lawyer
        in the area so a request never dead-ends if that lawyer is full."""
        lawyer = self._requested_lawyer_surname()
        slots = self._calendar.available_slots_sync(
            area,
            exclude_ids=self.state.offered_slot_ids,
            exclude_times=self.state.declined_slot_times,
            lawyer=lawyer,
            limit=3,
        )
        if not slots and lawyer:
            slots = self._calendar.available_slots_sync(
                area,
                exclude_ids=self.state.offered_slot_ids,
                exclude_times=self.state.declined_slot_times,
                limit=3,
            )
        return slots

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
        if AREA_DENIAL_RE.search(lowered):
            if field == "email":
                self._email.on_rejected(entity.value)
            elif field == "phone":
                self._phone.reset()
            del self.state.entities[field]
            self.advance_phase()
        elif CONFIRM_YES_RE.search(lowered):
            self.confirm_entity(field)
            if field == "phone":
                self._phone.on_confirmed()

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
        if AREA_DENIAL_RE.search(lowered):
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
        self.refresh_llm_context()

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

        if self._email.capture_inline(text):
            captured = True

        if not skip_phone and self._phone.capture_volunteered(text):
            captured = True

        if captured:
            self.refresh_llm_context()

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
        elif len(matches) > 1:
            # The matter touches more than one area (e.g. "Mietvertrag gekündigt"):
            # we cannot guess. Record the candidates so the scripted spine asks a
            # disambiguation question instead of leaving the call stalled in ROUTING
            # (where the LLM is otherwise free to hallucinate a booking).
            self.state.area_options = [area.value for area, _ in matches]
            self.state.caller_intent = CallerIntent.BOOK_CONSULTATION
            self.state.matter_summary = text
            logger.info("Routing ambiguous %s → will disambiguate", self.state.area_options)

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

    def confirm_booking(self, slot: dict) -> None:
        """Record a confirmed booking and advance to confirmation. The one place
        booking state is set, so tools and the deterministic path stay in sync."""
        self.state.booking_confirmed = True
        self.state.booked_slot = slot
        self.advance_phase()

    def record_escalation(self, reason: str = "") -> None:
        """Flag the call for human escalation (e.g. out-of-scope area)."""
        self.state.escalation_requested = True
        if reason:
            self.state.escalation_reason = reason
        self.advance_phase()

    def record_offered_slots(self, slot_ids: list[int]) -> None:
        """Remember which slots were offered, so they aren't re-offered next round."""
        self.state.offered_slot_ids = list(slot_ids)

    def awaiting_field(self) -> str | None:
        """The datum the last scripted question asked for (or None when the LLM
        drives). Read-only accessor so processors don't reach into state."""
        return self.state.awaiting

    def is_booking_confirmed(self) -> bool:
        return self.state.booking_confirmed

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
