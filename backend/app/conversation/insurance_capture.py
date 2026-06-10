"""Traffic insurance/claim-number capture.

A cohesive strategy owning the parse → accumulate → read-back → confirm policy for
the one alphanumeric reference field, so the manager dispatches to it rather than
carrying the logic inline. The capture *state* (resolved / attempts / buffer) stays
on ConversationState, since flow/script read it; this owns the *behaviour*.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.conversation.cues import CONFIRM_YES_RE, NEGATE_RE
from app.conversation.phone import digit_words_to_digits

if TYPE_CHECKING:
    from app.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

_REF_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]*")


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


class InsuranceCapture:
    """Owns the insurance-number capture turn. Returns True from ``handle`` when it
    consumed the turn (so phone capture stands down)."""

    # Higher than email's: a reference is dictated in several short bursts, so allow
    # a few turns to accumulate before giving up.
    MAX_ATTEMPTS = 4

    def __init__(self, manager: ConversationManager):
        self._m = manager

    def handle(self, text: str) -> bool:
        state = self._m.state
        if state.awaiting not in ("insurance", "insurance_confirm"):
            return False
        if state.insurance_resolved:
            return False
        chunk = _ref_tokens(digit_words_to_digits(text))
        low = text.lower()

        if state.awaiting == "insurance_confirm":
            return self._handle_confirm(text, low, chunk)
        return self._handle_collect(text, low, chunk)

    def _handle_confirm(self, text: str, low: str, chunk: str) -> bool:
        state = self._m.state
        if chunk:
            # More digits → the caller is still dictating; append, read back again.
            state.insurance_buffer += chunk
            self._m.store_entity("insurance_number", state.insurance_buffer, 0.9)
            state.insurance_attempts = 0
            return True
        if CONFIRM_YES_RE.search(low):
            self._m.confirm_entity("insurance_number")
            state.insurance_resolved = True
            self._m.advance_phase()
            return True
        if NEGATE_RE.search(low):
            # Wrong → discard and ask again from scratch.
            state.entities.pop("insurance_number", None)
            state.insurance_buffer = ""
            state.insurance_attempts = 0
            return False
        # Unclear reply — don't loop forever; accept what we have after a couple.
        state.insurance_attempts += 1
        if state.insurance_attempts >= 2:
            self._m.confirm_entity("insurance_number")
            state.insurance_resolved = True
            self._m.advance_phase()
        return False

    def _handle_collect(self, text: str, low: str, chunk: str) -> bool:
        state = self._m.state
        if "insurance_number" in state.entities:
            return False
        negative = bool(NEGATE_RE.search(low))
        says_has_other = bool(re.search(r"\b(?:dafür|aber|habe|have)\b", text, re.IGNORECASE))
        # A clear "no" with nothing dictated (now or earlier) → caller has none.
        if negative and not chunk and not state.insurance_buffer and not says_has_other:
            logger.info("Insurance step resolved: caller has no number")
            state.insurance_resolved = True
            self._m.advance_phase()
            return False
        if chunk:
            state.insurance_buffer += chunk
        buffered = state.insurance_buffer
        digits = re.sub(r"[^A-Za-z0-9]", "", buffered)
        # Enough to read back → store UNCONFIRMED; the scripted read-back then lets
        # the caller confirm or keep adding digits before we advance.
        if len(digits) >= 5:
            logger.info("Insurance captured (pending read-back): %r", buffered)
            state.insurance_attempts = 0
            self._m.store_entity("insurance_number", buffered, 0.9)
            return True
        # Still incomplete. Give a few turns to finish; then give up (or read back a
        # partial if we got a few chars).
        state.insurance_attempts += 1
        if state.insurance_attempts >= self.MAX_ATTEMPTS:
            if len(digits) >= 3:
                self._m.store_entity("insurance_number", buffered, 0.9)  # read it back
            else:
                logger.info("Insurance unresolved → proceeding without it")
                state.insurance_resolved = True
                self._m.advance_phase()
        return False
