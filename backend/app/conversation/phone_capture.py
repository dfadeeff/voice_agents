"""Spoken-phone capture: accumulate across split turns, read back when complete.

Callers dictate the number in bursts with pauses, so STT delivers it as several
finals. This strategy appends each turn's digits to a buffer and stores the whole
(for the scripted read-back) only at a realistic length — and, while the read-back
is up, folds in any further digits the caller keeps dictating (mirrors the
insurance read-back). The capture *state* (buffer/attempts) stays on
ConversationState, since flow/script read it; this owns the *behaviour*.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.conversation.phone import normalize_phone_text

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)


class PhoneCapture:
    """Owns the phone capture/confirm turns for the scripted spine."""

    # A normalized German number (+49…) is ~12 digits for a mobile; we wait for at
    # least this many before reading it back, so a split dictation isn't confirmed
    # at its first short fragment. After a couple of asks we accept a shorter one.
    MIN_DIGITS = 10
    MAX_ATTEMPTS = 2

    def __init__(self, manager: ConversationManager):
        self._m = manager

    def handle(self, text: str) -> None:
        """Accumulate a spoken phone number while it is the awaited field, reading
        it back only once it's plausibly complete. A short landline still lands
        after a couple of asks via the lowered fallback threshold."""
        state = self._m.state
        if state.awaiting not in ("phone", "phone_confirm"):
            return
        existing = state.entities.get("phone")
        if existing and existing.confirmed:
            return
        turn_has_digits = bool(re.search(r"\d", normalize_phone_text(text)))

        if state.awaiting == "phone_confirm":
            # Yes/No is handled by the read-back confirmation (which ran first). If
            # that left the unconfirmed number in place and the caller is still
            # adding digits, extend it and read the fuller number back.
            if existing and not existing.confirmed and turn_has_digits:
                state.phone_buffer = f"{state.phone_buffer} {text}".strip()
                self._m.store_entity("phone", normalize_phone_text(state.phone_buffer), 0.9)
                self._m.refresh_llm_context()
            return

        # awaiting == "phone": accumulate and read back once plausibly complete.
        if not turn_has_digits:
            return
        state.phone_buffer = f"{state.phone_buffer} {text}".strip()
        normalized = normalize_phone_text(state.phone_buffer)
        digit_count = len(re.sub(r"\D", "", normalized))
        state.phone_attempts += 1
        enough = digit_count >= self.MIN_DIGITS or (
            state.phone_attempts >= self.MAX_ATTEMPTS and digit_count >= 7
        )
        if enough:
            logger.info("Phone captured (buffered): %r", normalized)
            self._m.store_entity("phone", normalized, 0.9)
            self._m.refresh_llm_context()

    def capture_volunteered(self, text: str) -> bool:
        """Opportunistically grab a number volunteered on some other turn (e.g.
        while giving the name). When the phone is the field being asked/confirmed,
        ``handle`` owns it — it buffers split dictation and waits for a plausibly
        complete number. Returns True when a phone was captured."""
        state = self._m.state
        if state.awaiting in ("phone", "phone_confirm"):
            return False
        existing = state.entities.get("phone")
        if existing and existing.confirmed:
            return False
        normalized = normalize_phone_text(text)
        if len(re.sub(r"\D", "", normalized)) < 7:
            return False
        logger.info("Deterministic capture (LLM fallback): phone=%r", normalized)
        self._m.store_entity("phone", normalized, 0.9)
        return True

    def reset(self) -> None:
        """Wrong number on read-back → drop the accumulated buffer so the caller's
        restated number is captured fresh, not appended to the bad one."""
        state = self._m.state
        state.phone_buffer = ""
        state.phone_attempts = 0

    def on_confirmed(self) -> None:
        self._m.state.phone_buffer = ""
