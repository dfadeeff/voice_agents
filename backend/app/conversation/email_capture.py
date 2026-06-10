"""Spoken-email capture: accumulate → parse → read-back → spell → skip.

A cohesive strategy owning the whole email policy — chunked dictation across
split turns, read-back rejections that escalate to spelling mode, the optional
LLM rescue, and the bounded attempts that skip email rather than loop forever
(a phone number is enough to book or call back). The capture *state* (buffers,
attempt counters, flags) stays on ConversationState, since flow/script read it;
this owns the *behaviour*.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.conversation.cues import NEGATE_RE
from app.conversation.email_parse import (
    anchor_email_to_name,
    parse_email,
    parse_spelled_local,
)
from app.validation import is_valid_email

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

# "Nein, die Domäne ist hotmail" is a correction, not a skip.
_DOMAIN_HINT_RE = re.compile(
    r"\b(?:dom[äa]ne|domain|gmail|hotmail|yahoo|outlook|web\.de|gmx|@)\b",
    re.IGNORECASE,
)
# A spoken domain suffix ("punkt/dot/point" or a literal ".de"/".com") — the LLM
# rescue must not run before one was said, or it invents a TLD.
_SPOKEN_DOT_RE = re.compile(r"\b(?:punkt|dot|point)\b", re.IGNORECASE)
_LITERAL_TLD_RE = re.compile(r"\.[a-zA-Z]{2,}")


class EmailCapture:
    """Owns the email capture/confirm/spell/skip turns for the scripted spine."""

    MAX_ATTEMPTS = 3
    # Read-back rejections: after this many, offer to spell the local part; after
    # the higher bound, give up on email entirely (a phone number is enough).
    REJECTS_BEFORE_SPELLING = 2
    MAX_REJECTS = 4
    # Spelling attempts that yielded no usable local part before we skip email.
    MAX_SPELL_ATTEMPTS = 3

    def __init__(self, manager: ConversationManager):
        self._m = manager
        # Optional async LLM rescue for spoken email; None = regex-only (local default).
        self._extractor: Callable[[str, str], Any] | None = None

    def set_extractor(self, extractor: Callable[[str, str], Any] | None) -> None:
        self._extractor = extractor

    def anchor(self, email: str) -> str:
        """Snap a near-miss local part to the caller's captured name (e.g. STT
        'sigma' → 'sigmar'). No-op when no name is known or the match isn't close."""
        name_ent = self._m.state.entities.get("name")
        name = name_ent.value if name_ent else ""
        anchored = anchor_email_to_name(email, name)
        if anchored != email:
            logger.info("Email anchored to name %r: %r → %r", name, email, anchored)
        return anchored

    async def resolve_if_pending(self, text: str) -> None:
        """LLM rescue when we asked for the email but regex couldn't parse one.

        Runs after add_user_message (so regex had first go) and only when an
        extractor is configured. On success the email is stored unconfirmed —
        the scripted read-back then confirms it like any other email.
        """
        state = self._m.state
        if state.awaiting == "email_spell":
            await self._resolve_spelled(text)
            return
        if self._extractor is None or state.awaiting != "email":
            return
        if "email" in state.entities:
            return  # regex already got it this turn
        # Use the accumulated buffer so a split dictation is rescued as a whole.
        source = state.email_buffer or text
        # Don't let the model invent a domain suffix the caller hasn't spoken yet —
        # it loves to guess ".com" (e.g. "Klein at Hotmail" → klein@hotmail.com)
        # when the TLD arrives in a separate, split utterance. Only rescue once a
        # suffix was actually said ("punkt/dot/point" or a literal ".de"/".com").
        if not _SPOKEN_DOT_RE.search(source) and not _LITERAL_TLD_RE.search(source):
            logger.info("Email rescue skipped: no spoken domain suffix in %r", source)
            return
        name_ent = state.entities.get("name")
        name_hint = name_ent.value if name_ent else ""
        try:
            candidate = await self._extractor(source, name_hint)
        except Exception as e:
            logger.warning("Email extractor raised: %s", e)
            return
        if not candidate:
            return
        candidate = candidate.strip().lower()
        if not is_valid_email(candidate):
            return
        candidate = self.anchor(candidate)
        logger.info("LLM email rescue: %r → %r", source, candidate)
        # Undo the miss the deterministic path may have just recorded.
        state.email_skipped = False
        state.email_misheard = False
        state.email_attempts = max(0, state.email_attempts - 1)
        state.email_buffer = ""
        self._m.store_entity("email", candidate, 0.9)  # unconfirmed → scripted read-back
        self._m.refresh_llm_context()
        self._m.persist_progress()

    async def _resolve_spelled(self, text: str) -> None:
        """LLM fallback for spelling mode, plus the spell-loop bound.

        The deterministic spell parser already ran in add_user_message; if it
        landed an email this is a no-op. Otherwise an LLM (when configured) gets
        the spelled letters plus the known domain to assemble. If nothing usable
        emerges, count the miss and skip email after a few tries rather than
        asking the caller to spell forever."""
        state = self._m.state
        if "email" in state.entities:
            return  # deterministic spell capture (or a restated address) already got it
        if self._extractor is not None:
            domain = state.email_domain or "gmail.com"
            source = f"{state.email_buffer or text} at {domain}"
            name_ent = state.entities.get("name")
            name_hint = name_ent.value if name_ent else ""
            try:
                candidate = (await self._extractor(source, name_hint) or "").strip().lower()
            except Exception as e:
                logger.warning("Spelled-email extractor raised: %s", e)
                candidate = ""
            if candidate and is_valid_email(candidate):
                candidate = self.anchor(candidate)
                logger.info("LLM spelled-email rescue: %r → %r", source, candidate)
                state.email_buffer = ""
                state.email_spell_attempts = 0
                self._m.store_entity("email", candidate, 0.9)
                self._m.refresh_llm_context()
                self._m.persist_progress()
                return
        # Still nothing — bound the loop.
        state.email_spell_attempts += 1
        if state.email_spell_attempts >= self.MAX_SPELL_ATTEMPTS:
            logger.info("Email skipped after %d spelling attempts", state.email_spell_attempts)
            state.email_skipped = True
            state.email_buffer = ""
            self._m.advance_phase()
            self._m.persist_progress()

    def try_skip(self, text: str) -> None:
        """Mark email as skipped when we asked for it and the caller has none."""
        state = self._m.state
        if state.callback_requested or state.email_skipped:
            return
        if state.awaiting != "email":
            return
        if "email" in state.entities:
            return
        if not NEGATE_RE.search(text.lower()):
            return
        if _DOMAIN_HINT_RE.search(text):
            return  # a domain correction, not a skip
        logger.info("Email skipped: caller has none")
        state.email_skipped = True
        self._m.advance_phase()

    def track_attempt(self, awaiting: str | None) -> None:
        """Track failed email attempts; re-ask, then skip after a few misses.

        Spoken email over phone-quality audio is the hardest field. Rather than
        loop forever, after a few attempts we mark it skipped — a phone number is
        enough to book or call back."""
        state = self._m.state
        if awaiting != "email":
            return
        if "email" in state.entities or state.email_skipped:
            state.email_misheard = False
            return
        state.email_attempts += 1
        if state.email_attempts >= self.MAX_ATTEMPTS:
            logger.info("Email skipped after %d failed attempts", state.email_attempts)
            state.email_skipped = True
            state.email_misheard = False
            state.email_buffer = ""
            self._m.advance_phase()
        else:
            state.email_misheard = True

    def try_capture_spelled(self, text: str) -> None:
        """Capture an email whose local part is being spelled out (buchstabieren).

        Accumulates across split turns, then assembles a candidate two ways: a
        whole restated address still parses ("ritter at gmail punkt com"), and
        otherwise the spelled letters reattach to the domain heard before spelling
        ("r, i, doppel t, e, r" + gmail.com). On a miss the buffer is kept and the
        LLM fallback (resolve_if_pending) gets a turn."""
        state = self._m.state
        if state.awaiting != "email_spell":
            return
        existing = state.entities.get("email")
        if existing and existing.confirmed:
            return
        state.email_buffer = f"{state.email_buffer} {text}".strip()
        source = state.email_buffer
        candidate = parse_email(source)
        if not candidate:
            local = parse_spelled_local(source)
            if local:
                domain = state.email_domain or "gmail.com"
                candidate = f"{local}@{domain}"
                candidate = candidate if is_valid_email(candidate) else None
        if not candidate:
            return
        candidate = self.anchor(candidate)
        logger.info("Spelled email captured: %r → %r", source, candidate)
        state.email_buffer = ""
        state.email_spell_attempts = 0
        self._m.store_entity("email", candidate, 0.9)  # unconfirmed → scripted read-back
        self._m.refresh_llm_context()

    def capture_inline(self, text: str) -> bool:
        """Opportunistic whole-address parse (the LLM-skipped-the-tool fallback).

        Email is often dictated in chunks across split turns ("Klein at Hotmail" |
        "Punkt de"). When we asked for it, accumulate the turn text and parse the
        whole buffer so the address reassembles. Stands down in spelling mode —
        the spelled-letters handler owns email capture there (this would otherwise
        re-store the same mangled address the caller just rejected and skip the
        spell offer). Returns True when an email was captured."""
        state = self._m.state
        if state.email_spelling:
            return False
        existing = state.entities.get("email")
        if existing and existing.confirmed:
            return False
        if state.awaiting == "email":
            state.email_buffer = f"{state.email_buffer} {text}".strip()
            source = state.email_buffer
        else:
            source = text
        email = parse_email(source)
        if not email:
            return False
        email = self.anchor(email)
        logger.info("Deterministic capture (LLM fallback): email=%r", email)
        self._m.store_entity("email", email, 0.9)
        state.email_buffer = ""
        return True

    def on_rejected(self, rejected: str) -> None:
        """The caller said the read-back email was wrong. After two rejections we
        switch to spelling the local part (what STT keeps mangling), keeping the
        already-recognised domain; after four we give up and skip email."""
        state = self._m.state
        state.email_domain = rejected.partition("@")[2] or state.email_domain
        state.email_confirm_rejects += 1
        if state.email_confirm_rejects >= self.MAX_REJECTS:
            logger.info("Email skipped after %d rejections", state.email_confirm_rejects)
            state.email_skipped = True
        elif state.email_confirm_rejects >= self.REJECTS_BEFORE_SPELLING:
            logger.info("Switching to email spelling mode")
            state.email_spelling = True
