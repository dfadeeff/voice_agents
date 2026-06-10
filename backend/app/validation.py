"""Shared validation primitives.

One source of truth for the email grammar and the low-confidence threshold, which
were previously duplicated (with subtly different patterns) across the manager,
the extraction tool, the LLM email rescue, and the TTS preprocessor.
"""

from __future__ import annotations

import re

# STT segment confidence below this stores a contact field *unconfirmed*, forcing
# a read-back (the "double-check when the model is unsure" path).
LOW_CONFIDENCE_THRESHOLD = 0.75

# Shared email grammar. ``EMAIL_RE`` validates a whole string; ``EMAIL_FIND_RE``
# locates one inside arbitrary text (e.g. to expand it to spoken form for TTS).
_EMAIL_CORE = r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}"
EMAIL_RE = re.compile(rf"^{_EMAIL_CORE}$", re.IGNORECASE)
EMAIL_FIND_RE = re.compile(_EMAIL_CORE, re.IGNORECASE)


def is_valid_email(value: str) -> bool:
    """True when the whole (trimmed) string is a syntactically valid email."""
    return bool(EMAIL_RE.match(value.strip()))


def find_email(text: str) -> str | None:
    """The first email-like substring in ``text``, or None."""
    m = EMAIL_FIND_RE.search(text)
    return m.group(0) if m else None
