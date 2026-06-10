from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from app.conversation.flow import PHASE_TOOLS, next_phase
from app.conversation.phone import digit_words_to_digits, normalize_phone_text
from app.conversation.policy import (
    extract_target_person,
    is_explicit_handoff_request,
    wants_callback,
)
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


# German spoken hour words (12h/24h). Used to recognise a *requested* time that
# wasn't among the offered slots, so the caller can ask for a different one.
_HOUR_WORDS = {
    "ein": 1,
    "eins": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "fuenf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "zwoelf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "fuenfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
}
_MINUTE = r"(dreißig|dreissig|30)"
# "<hour> Uhr [dreißig]" OR "<hour> dreißig" (callers drop "Uhr": "vierzehn
# dreißig" → 14:30). A bare hour with neither "Uhr" nor a minute isn't treated
# as a time, so stray numbers don't false-match.
_REQUEST_TIME_RE = re.compile(
    r"\b(?:um\s+)?(\d{1,2}|" + "|".join(_HOUR_WORDS) + r")\s*"
    r"(?:uhr(?:\s*" + _MINUTE + r")?|" + _MINUTE + r")",
    re.IGNORECASE,
)


def _parse_requested_time(text: str) -> str | None:
    """Extract a specific clock time the caller asked for ('dreizehn Uhr' → '13:00',
    'neun Uhr dreißig' → '09:30', 'vierzehn dreißig' → '14:30'). Returns 'HH:MM'
    or None."""
    m = _REQUEST_TIME_RE.search(text.lower())
    if not m:
        return None
    token = m.group(1)
    hour = int(token) if token.isdigit() else _HOUR_WORDS.get(token)
    if hour is None or not 0 <= hour <= 23:
        return None
    minute = 30 if (m.group(2) or m.group(3)) else 0
    return f"{hour:02d}:{minute:02d}"


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
    r"^(?:\s*\b(?:ja|nein|also|genau|ähm|äh|meine?|die|das|es|und|hier|war|ich|"
    r"e[-\s]?mail|email|mail|adresse|lautet|ist|wäre|my|the|email|address|it'?s|is)\b"
    r"[\s.:,!?-]*)+",
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
    # When a spoken "at" sat between dots ("baum.at.gmail.com" / "baum at gmail
    # dot com"), the connector→@ swap leaves a stray dot touching the @. Collapse
    # ".@" / "@." so it doesn't produce an invalid local part ("baum.@gmail.com").
    t = re.sub(r"\.*@\.*", "@", t)
    if "@" not in t:
        return None
    local, _, domain = t.partition("@")
    parts = [p for p in domain.split(".") if p]
    if local and parts:
        parts[0] = _DOMAIN_CORRECTIONS.get(parts[0], parts[0])
    candidate = f"{local}@{'.'.join(parts)}"
    return candidate if _EMAIL_VALID_RE.match(candidate) else None


# Spoken letter names STT emits when a caller spells aloud, German first then
# common English renderings. Single vowels (a/e/i/o/u) are letters as-is.
_LETTER_NAMES = {
    "be": "b", "ce": "c", "de": "d", "ef": "f", "ge": "g", "ha": "h",
    "jot": "j", "ka": "k", "el": "l", "em": "m", "en": "n", "pe": "p",
    "ku": "q", "er": "r", "es": "s", "te": "t", "vau": "v", "we": "w",
    "weh": "w", "ix": "x", "ypsilon": "y", "zett": "z", "ah": "a", "oh": "o",
    "ay": "a", "bee": "b", "cee": "c", "see": "c", "dee": "d", "ee": "e",
    "eff": "f", "gee": "g", "aitch": "h", "jay": "j", "kay": "k", "ell": "l",
    "oh ": "o", "pee": "p", "cue": "q", "queue": "q", "ar": "r", "are": "r",
    "ess": "s", "tee": "t", "yu": "u", "vee": "v", "ex": "x", "wy": "y",
    "why": "y", "zee": "z", "zed": "z",
}  # fmt: skip
# "wie/für/as in/like" introduce an example word ("R wie Richard"); the letter is
# what precedes them. "doppel/double X" means the letter X twice.
_SPELL_EXAMPLE_RE = re.compile(r"\b([a-zäöü])\s+(?:wie|für|as\s+in|like|for)\s+\w+", re.IGNORECASE)
_SPELL_DOUBLE_RE = re.compile(r"\b(?:doppel|double)[\s-]*([a-zäöü])\b", re.IGNORECASE)


def _parse_spelled_local(text: str) -> str | None:
    """Reconstruct an email local part dictated letter by letter.

    Handles the conventions callers actually use when the address is being
    spelled: bare letters ("r i t t e r"), letter names STT renders as words
    ("er i te te e er"), phonetic examples ("R wie Richard"), and doubling
    ("doppel T"). Returns the assembled local part, or None when fewer than two
    letters were recognised — that ambiguous case is left to the LLM fallback.
    """
    t = _EMAIL_LEADIN_RE.sub("", text.lower().strip())
    t = _SPELL_EXAMPLE_RE.sub(r"\1", t)
    t = _SPELL_DOUBLE_RE.sub(r"\1 \1", t)
    letters: list[str] = []
    for tok in re.split(r"[\s,.;:!?]+", t):
        if not tok:
            continue
        if len(tok) == 1 and tok.isalpha():
            letters.append(tok)
        elif tok in _LETTER_NAMES:
            letters.append(_LETTER_NAMES[tok])
    return "".join(letters) if len(letters) >= 2 else None


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two short strings (iterative, O(len*len))."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anchor_email_to_name(email: str, name: str) -> str:
    """Rewrite a local part that is a near-miss of the caller's name.

    Spoken email over a phone line loses letters ('sigma' for 'Sigmar'). When the
    captured name is known and the local part is a close match (edit distance 1-2)
    to a name-derived form, snap it to that form. Only near-matches are touched, so
    a genuinely different address ('leon.legal') is left alone — and the scripted
    read-back still lets the caller reject it.
    """
    if not name or "@" not in email:
        return email
    local, _, domain = email.partition("@")
    if len(local) < 3:
        return email
    parts = [p for p in re.split(r"\s+", name.lower().strip()) if p]
    if not parts:
        return email
    candidates = {parts[0], parts[-1], "".join(parts), ".".join(parts)}
    if len(parts) >= 2:
        candidates.add(f"{parts[0]}.{parts[-1]}")
    best, best_d = None, 99
    for cand in candidates:
        d = _levenshtein(local, cand)
        if d < best_d:
            best, best_d = cand, d
    if best and 0 < best_d <= 2 and best_d < len(local):
        return f"{best}@{domain}"
    return email


def _ref_tokens(text: str) -> str:
    """Join the reference-like tokens in text (digit-bearing tokens, plus a short
    letter prefix such as 'VS' or a single spoken 'F'), without a length gate.

    Spoken numbers arrive in chunks across turns ('F vier' … 'fünf vier'); each
    chunk is short on its own, so the caller of this assembles them into a buffer.
    """
    tokens = _REF_TOKEN_RE.findall(text)
    kept: list[str] = []
    for i, tok in enumerate(tokens):
        if any(c.isdigit() for c in tok):
            kept.append(tok)
        elif kept:
            break  # a word after the number ends the reference
        elif (
            tok.isalpha()
            and (tok.isupper() or len(tok) == 1)
            and len(tok) <= 5
            and i + 1 < len(tokens)
            and any(c.isdigit() for c in tokens[i + 1])
        ):
            # Short prefix before the digits: 'VS' or a single letter like 'F'
            # (STT often lowercases a spoken letter — keep it, uppercased).
            kept.append(tok.upper())
    return "".join(kept)


def _extract_reference(text: str) -> str | None:
    """A complete insurance/claim reference (≥5 alphanumerics, digit-bearing), or
    None for negative replies / fragments too short to be a whole reference."""
    joined = _ref_tokens(text)
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
        # Optional async LLM rescue for spoken email; None = regex-only (local default).
        self._email_extractor: Callable[[str], Any] | None = None
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
        self._email_extractor = extractor

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
        self._persist()

    def _anchor_email(self, email: str) -> str:
        """Snap a near-miss local part to the caller's captured name (e.g. STT
        'sigma' → 'sigmar'). No-op when no name is known or the match isn't close."""
        name_ent = self.state.entities.get("name")
        name = name_ent.value if name_ent else ""
        anchored = _anchor_email_to_name(email, name)
        if anchored != email:
            logger.info("Email anchored to name %r: %r → %r", name, email, anchored)
        return anchored

    async def resolve_email_if_pending(self, text: str) -> None:
        """LLM rescue when we asked for the email but regex couldn't parse one.

        Runs after add_user_message (so regex had first go) and only when an
        extractor is configured. On success the email is stored unconfirmed —
        the scripted read-back then confirms it like any other email.
        """
        if self.state.awaiting == "email_spell":
            await self._resolve_spelled_email(text)
            return
        extractor = getattr(self, "_email_extractor", None)
        if extractor is None or self.state.awaiting != "email":
            return
        if "email" in self.state.entities:
            return  # regex already got it this turn
        # Use the accumulated buffer so a split dictation is rescued as a whole.
        source = self.state.email_buffer or text
        # Don't let the model invent a domain suffix the caller hasn't spoken yet —
        # it loves to guess ".com" (e.g. "Klein at Hotmail" → klein@hotmail.com)
        # when the TLD arrives in a separate, split utterance. Only rescue once a
        # suffix was actually said ("punkt/dot/point" or a literal ".de"/".com").
        if not re.search(r"\b(?:punkt|dot|point)\b", source, re.IGNORECASE) and not re.search(
            r"\.[a-zA-Z]{2,}", source
        ):
            logger.info("Email rescue skipped: no spoken domain suffix in %r", source)
            return
        name_ent = self.state.entities.get("name")
        name_hint = name_ent.value if name_ent else ""
        try:
            candidate = await extractor(source, name_hint)
        except Exception as e:
            logger.warning("Email extractor raised: %s", e)
            return
        if not candidate:
            return
        candidate = candidate.strip().lower()
        if not _EMAIL_VALID_RE.match(candidate):
            return
        candidate = self._anchor_email(candidate)
        logger.info("LLM email rescue: %r → %r", source, candidate)
        # Undo the miss the deterministic path may have just recorded.
        self.state.email_skipped = False
        self.state.email_misheard = False
        self.state.email_attempts = max(0, self.state.email_attempts - 1)
        self.state.email_buffer = ""
        self.store_entity("email", candidate, 0.9)  # unconfirmed → scripted read-back
        self._update_llm_context()
        self._persist()

    async def _resolve_spelled_email(self, text: str) -> None:
        """LLM fallback for spelling mode, plus the spell-loop bound.

        The deterministic spell parser already ran in add_user_message; if it
        landed an email this is a no-op. Otherwise an LLM (when configured) gets
        the spelled letters plus the known domain to assemble. If nothing usable
        emerges, count the miss and skip email after a few tries rather than
        asking the caller to spell forever."""
        if "email" in self.state.entities:
            return  # deterministic spell capture (or a restated address) already got it
        extractor = getattr(self, "_email_extractor", None)
        if extractor is not None:
            domain = self.state.email_domain or "gmail.com"
            source = f"{self.state.email_buffer or text} at {domain}"
            name_ent = self.state.entities.get("name")
            name_hint = name_ent.value if name_ent else ""
            try:
                candidate = (await extractor(source, name_hint) or "").strip().lower()
            except Exception as e:
                logger.warning("Spelled-email extractor raised: %s", e)
                candidate = ""
            if candidate and _EMAIL_VALID_RE.match(candidate):
                candidate = self._anchor_email(candidate)
                logger.info("LLM spelled-email rescue: %r → %r", source, candidate)
                self.state.email_buffer = ""
                self.state.email_spell_attempts = 0
                self.store_entity("email", candidate, 0.9)
                self._update_llm_context()
                self._persist()
                return
        # Still nothing — bound the loop.
        self.state.email_spell_attempts += 1
        if self.state.email_spell_attempts >= self._MAX_EMAIL_SPELL_ATTEMPTS:
            logger.info("Email skipped after %d spelling attempts", self.state.email_spell_attempts)
            self.state.email_skipped = True
            self.state.email_buffer = ""
            self.advance_phase()
            self._persist()

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
            self._update_llm_context()

        # Reply parsing dispatches on what the last scripted question asked for
        # (state.awaiting), not on regex over the agent's own prior sentence.
        awaiting = self.state.awaiting

        # The caller is answering "which area?" after an ambiguous opening.
        if awaiting == "area":
            self._handle_area_choice(text)
            if self.state.legal_area != LegalArea.UNKNOWN:
                just_routed = True

        # The caller denies the area-confirmation ("nein, es geht um etwas
        # anderes") → re-route instead of recording a matter type.
        if awaiting == "matter_type" and _AREA_DENIAL_RE.search(text.lower()):
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
        captured_insurance = self._try_capture_insurance(text)
        self._try_capture_contact(text, skip_phone=captured_insurance)
        self._try_capture_phone(text)
        self._try_capture_spelled_email(text)
        self._try_skip_email(text)
        self._handle_email_attempt(awaiting)
        self._try_capture_preferred_time(text)
        self._try_book(text)
        self._persist()

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
        # One-shot apology flag: cleared each turn, re-set below if the caller's
        # requested time is unavailable (so the apology shows for exactly one offer).
        self.state.unavailable_time = None
        area = self.state.legal_area.value

        if not self.state.offered_slots:
            self.state.offered_slots = self._available_slots(area)
            self._update_llm_context()
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
        elif (requested := _parse_requested_time(text)) is not None:
            # Caller named a specific time that wasn't offered ("Können wir 13 Uhr
            # machen?"). Honour it if free; otherwise apologise and re-offer.
            slot = self._calendar.find_slot_sync(requested, area, self._requested_lawyer_surname())
            if slot and self._book_slot(slot):
                return
            self.state.unavailable_time = requested
        else:
            return  # unrecognised reply → re-present the same offer

        self.state.offered_slots = self._available_slots(area)
        self._update_llm_context()

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
        self.state.booking_confirmed = True
        self.state.booked_slot = booking
        self.advance_phase()
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
    # Read-back rejections: after this many, offer to spell the local part; after
    # the higher bound, give up on email entirely (a phone number is enough).
    _REJECTS_BEFORE_SPELLING = 2
    _MAX_EMAIL_REJECTS = 4
    # Spelling attempts that yielded no usable local part before we skip email.
    _MAX_EMAIL_SPELL_ATTEMPTS = 3
    # Higher than email's: a reference is often dictated in several short bursts,
    # so allow a few turns to accumulate before giving up.
    _MAX_INSURANCE_ATTEMPTS = 4
    # A normalized German number (+49…) is ~12 digits for a mobile; we wait for at
    # least this many before reading it back, so a split dictation isn't confirmed
    # at its first short fragment. After a couple of asks we accept a shorter one.
    _MIN_PHONE_DIGITS = 10
    _MAX_PHONE_ATTEMPTS = 2

    def _try_capture_phone(self, text: str) -> None:
        """Accumulate a spoken phone number across split turns, reading it back only
        once it's plausibly complete.

        Callers dictate the number in bursts with pauses, so STT delivers it as
        several finals. We append each turn's digits to a buffer and store the
        whole (for the scripted read-back) only at a realistic length — and, while
        the read-back is up, fold in any further digits the caller keeps dictating
        (mirrors the insurance read-back). A short landline still lands after a
        couple of asks via the lowered fallback threshold."""
        awaiting = self.state.awaiting
        if awaiting not in ("phone", "phone_confirm"):
            return
        existing = self.state.entities.get("phone")
        if existing and existing.confirmed:
            return
        turn_has_digits = bool(re.search(r"\d", normalize_phone_text(text)))

        if awaiting == "phone_confirm":
            # Yes/No is handled by _try_confirm_readback (which ran first). If that
            # left the unconfirmed number in place and the caller is still adding
            # digits, extend it and read the fuller number back.
            if existing and not existing.confirmed and turn_has_digits:
                self.state.phone_buffer = f"{self.state.phone_buffer} {text}".strip()
                self.store_entity("phone", normalize_phone_text(self.state.phone_buffer), 0.9)
                self._update_llm_context()
            return

        # awaiting == "phone": accumulate and read back once plausibly complete.
        if not turn_has_digits:
            return
        self.state.phone_buffer = f"{self.state.phone_buffer} {text}".strip()
        normalized = normalize_phone_text(self.state.phone_buffer)
        digit_count = len(re.sub(r"\D", "", normalized))
        self.state.phone_attempts += 1
        enough = digit_count >= self._MIN_PHONE_DIGITS or (
            self.state.phone_attempts >= self._MAX_PHONE_ATTEMPTS and digit_count >= 7
        )
        if enough:
            logger.info("Phone captured (buffered): %r", normalized)
            self.store_entity("phone", normalized, 0.9)
            self._update_llm_context()

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
            self.state.email_buffer = ""
            self.advance_phase()
        else:
            self.state.email_misheard = True

    def _try_capture_spelled_email(self, text: str) -> None:
        """Capture an email whose local part is being spelled out (buchstabieren).

        Accumulates across split turns, then assembles a candidate two ways: a
        whole restated address still parses ("ritter at gmail punkt com"), and
        otherwise the spelled letters reattach to the domain heard before spelling
        ("r, i, doppel t, e, r" + gmail.com). On a miss the buffer is kept and the
        LLM fallback (resolve_email_if_pending) gets a turn."""
        if self.state.awaiting != "email_spell":
            return
        existing = self.state.entities.get("email")
        if existing and existing.confirmed:
            return
        self.state.email_buffer = f"{self.state.email_buffer} {text}".strip()
        source = self.state.email_buffer
        candidate = _parse_email(source)
        if not candidate:
            local = _parse_spelled_local(source)
            if local:
                domain = self.state.email_domain or "gmail.com"
                candidate = f"{local}@{domain}"
                candidate = candidate if _EMAIL_VALID_RE.match(candidate) else None
        if not candidate:
            return
        candidate = self._anchor_email(candidate)
        logger.info("Spelled email captured: %r → %r", source, candidate)
        self.state.email_buffer = ""
        self.state.email_spell_attempts = 0
        self.store_entity("email", candidate, 0.9)  # unconfirmed → scripted read-back
        self._update_llm_context()

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
            if field == "email":
                self._on_email_rejected(entity.value)
            elif field == "phone":
                # Wrong number → drop the accumulated buffer so the caller's
                # restated number is captured fresh, not appended to the bad one.
                self.state.phone_buffer = ""
                self.state.phone_attempts = 0
            del self.state.entities[field]
            self.advance_phase()
        elif _CONFIRM_YES_RE.search(lowered):
            self.confirm_entity(field)
            if field == "phone":
                self.state.phone_buffer = ""

    def _on_email_rejected(self, rejected: str) -> None:
        """The caller said the read-back email was wrong. After two rejections we
        switch to spelling the local part (what STT keeps mangling), keeping the
        already-recognised domain; after four we give up and skip email."""
        self.state.email_domain = rejected.partition("@")[2] or self.state.email_domain
        self.state.email_confirm_rejects += 1
        if self.state.email_confirm_rejects >= self._MAX_EMAIL_REJECTS:
            logger.info("Email skipped after %d rejections", self.state.email_confirm_rejects)
            self.state.email_skipped = True
        elif self.state.email_confirm_rejects >= self._REJECTS_BEFORE_SPELLING:
            logger.info("Switching to email spelling mode")
            self.state.email_spelling = True

    def _try_capture_insurance(self, text: str) -> bool:
        """Capture and confirm the traffic insurance reference.

        The number is dictated in fragments across split turns, then read back for
        confirmation — so the caller can keep adding digits ("…drei vier sieben")
        or correct it before we move on, instead of locking in a half-number.
        Sets ``insurance_resolved`` once confirmed (or skipped). Returns True when
        it consumed the turn (so phone capture stands down)."""
        if self.state.awaiting not in ("insurance", "insurance_confirm"):
            return False
        if self.state.insurance_resolved:
            return False
        chunk = _ref_tokens(digit_words_to_digits(text))
        low = text.lower()

        # Reading the captured number back for confirmation.
        if self.state.awaiting == "insurance_confirm":
            if chunk:
                # More digits → the caller is still dictating; append, read back again.
                self.state.insurance_buffer += chunk
                self.store_entity("insurance_number", self.state.insurance_buffer, 0.9)
                self.state.insurance_attempts = 0
                return True
            if _CONFIRM_YES_RE.search(low):
                self.confirm_entity("insurance_number")
                self.state.insurance_resolved = True
                self.advance_phase()
                return True
            if _NEGATE_RE.search(low):
                # Wrong → discard and ask again from scratch.
                self.state.entities.pop("insurance_number", None)
                self.state.insurance_buffer = ""
                self.state.insurance_attempts = 0
                return False
            # Unclear reply — don't loop forever; accept what we have after a couple.
            self.state.insurance_attempts += 1
            if self.state.insurance_attempts >= 2:
                self.confirm_entity("insurance_number")
                self.state.insurance_resolved = True
                self.advance_phase()
            return False

        # awaiting == "insurance": collect the number (across fragmented turns).
        if "insurance_number" in self.state.entities:
            return False
        negative = bool(_NEGATE_RE.search(low))
        says_has_other = bool(re.search(r"\b(?:dafür|aber|habe|have)\b", text, re.IGNORECASE))
        # A clear "no" with nothing dictated (now or earlier) → caller has none.
        if negative and not chunk and not self.state.insurance_buffer and not says_has_other:
            logger.info("Insurance step resolved: caller has no number")
            self.state.insurance_resolved = True
            self.advance_phase()
            return False
        if chunk:
            self.state.insurance_buffer += chunk
        buffered = self.state.insurance_buffer
        digits = re.sub(r"[^A-Za-z0-9]", "", buffered)
        # Enough to read back → store UNCONFIRMED; the scripted read-back then lets
        # the caller confirm or keep adding digits before we advance.
        if len(digits) >= 5:
            logger.info("Insurance captured (pending read-back): %r", buffered)
            self.state.insurance_attempts = 0
            self.store_entity("insurance_number", buffered, 0.9)
            return True
        # Still incomplete. Give a few turns to finish; then give up (or read back a
        # partial if we got a few chars).
        self.state.insurance_attempts += 1
        if self.state.insurance_attempts >= self._MAX_INSURANCE_ATTEMPTS:
            if len(digits) >= 3:
                self.store_entity("insurance_number", buffered, 0.9)  # read it back
            else:
                logger.info("Insurance unresolved → proceeding without it")
                self.state.insurance_resolved = True
                self.advance_phase()
        return False

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
        # In spelling mode the spelled-letters handler owns email capture, so this
        # opportunistic whole-address parse stands down (it would otherwise re-store
        # the same mangled address the caller just rejected and skip the spell offer).
        if not self.state.email_spelling and not (existing_email and existing_email.confirmed):
            # Email is often dictated in chunks across split turns ("Klein at
            # Hotmail" | "Punkt de"). When we asked for it, accumulate the turn
            # text and parse the whole buffer so the address reassembles.
            if self.state.awaiting == "email":
                self.state.email_buffer = f"{self.state.email_buffer} {text}".strip()
                source = self.state.email_buffer
            else:
                source = text
            email = _parse_email(source)
            if email:
                email = self._anchor_email(email)
                logger.info("Deterministic capture (LLM fallback): email=%r", email)
                self.store_entity("email", email, 0.9)
                self.state.email_buffer = ""
                captured = True

        existing_phone = self.state.entities.get("phone")
        # When the phone is the field being asked/confirmed, the dedicated
        # accumulator (_try_capture_phone) owns it — it buffers split dictation
        # and waits for a plausibly complete number. Here we only opportunistically
        # grab a number volunteered on some other turn (e.g. while giving the name).
        if (
            not skip_phone
            and self.state.awaiting not in ("phone", "phone_confirm")
            and not (existing_phone and existing_phone.confirmed)
        ):
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
