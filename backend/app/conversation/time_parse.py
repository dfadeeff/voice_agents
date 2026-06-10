"""Parse a clock time a caller asks for during booking.

Pure function, no conversation state — extracted verbatim from the manager. Slot
matching itself stays in the manager (it shares the confirmation cue regex).
"""

from __future__ import annotations

import re

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
# Longest-first so "dreizehn" wins over "drei" now that "Uhr" is optional.
_HOUR_ALT = "|".join(sorted(_HOUR_WORDS, key=len, reverse=True))
# An hour (digit or word), with "Uhr" and the half-hour both optional, so every
# spoken form lands — "vierzehn Uhr", "vierzehn Uhr dreißig", "vierzehn dreißig",
# and a bare "um 15" / "vierzehn" (→ on the hour). Only ever called once the caller
# is choosing a slot (see _try_book), so a bare number is a chosen time.
_REQUEST_TIME_RE = re.compile(
    r"\b(?:um\s+)?(\d{1,2}|" + _HOUR_ALT + r")(?:\s*uhr)?(?:\s*" + _MINUTE + r")?",
    re.IGNORECASE,
)


def parse_requested_time(text: str) -> str | None:
    """Extract a specific clock time the caller asked for, with or without 'Uhr'
    ('dreizehn Uhr' → '13:00', 'vierzehn dreißig' → '14:30', 'um 15' → '15:00').
    Returns 'HH:MM' or None. Slots are on the hour and half hour, so only ':30'
    minutes are recognised."""
    m = _REQUEST_TIME_RE.search(text.lower())
    if not m:
        return None
    token = m.group(1)
    hour = int(token) if token.isdigit() else _HOUR_WORDS.get(token)
    if hour is None or not 0 <= hour <= 23:
        return None
    minute = 30 if m.group(2) else 0
    return f"{hour:02d}:{minute:02d}"
