"""Shared spoken-reply cue patterns (yes / no / area-denial).

Extracted so capture strategies and the manager share one definition instead of
importing regexes from each other (which would couple them in a cycle).
"""

from __future__ import annotations

import re

# Negation cue ("nein", "noch keine", "kein").
NEGATE_RE = re.compile(
    r"\b(?:nein|nicht|falsch|kein|keine|nö|nee|no|wrong|incorrect)\b",
    re.IGNORECASE,
)
# A denial of a confirmation/area question specifically (not just any negation —
# "ich habe keine Versicherungsnummer" is not a denial of the accident).
AREA_DENIAL_RE = re.compile(
    r"\b(?:nein|nö|nee|stimmt\s+nicht|nicht\s+richtig|falsch|"
    r"no|that'?s\s+(?:not\s+right|wrong)|incorrect)\b",
    re.IGNORECASE,
)
# Affirmation in reply to a read-back ("ja, stimmt", "korrekt", "passt").
CONFIRM_YES_RE = re.compile(
    r"\b(?:ja|jawohl|genau|korrekt|stimmt|richtig|passt|yes|correct|right)\b",
    re.IGNORECASE,
)
