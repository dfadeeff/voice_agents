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
# A side question/objection instead of the asked-for datum ("Warum brauchen Sie
# meine E-Mail?", "Muss ich die Nummer angeben?"). Detecting it keeps the
# off-script turn from being miscounted as a failed capture attempt.
OFFSCRIPT_QUESTION_RE = re.compile(
    r"\?\s*$"
    r"|\b(?:warum|wieso|weshalb|wozu|wof[üu]r|why|what\s+for)\b"
    r"|\b(?:muss\s+ich|m[üu]ssen\s+sie|brauchen\s+sie|"
    r"ist\s+das\s+(?:n[öo]tig|notwendig|pflicht)|datenschutz|"
    r"do\s+i\s+(?:really\s+)?have\s+to|is\s+(?:that|this)\s+(?:necessary|required)|"
    r"privacy)\b",
    re.IGNORECASE,
)
# "Wie bitte?" / "Können Sie das wiederholen?" asks for the question again — not
# a side question to answer. The scripted re-ask path handles it (and its attempt
# cap keeps the call moving), so it must NOT divert to the LLM.
REPEAT_REQUEST_RE = re.compile(
    r"\b(?:wie\s+bitte|was\s+bitte|wiederholen|noch\s+?mal|nochmal|"
    r"nicht\s+verstanden|pardon|come\s+again|say\s+(?:that\s+)?again|repeat)\b",
    re.IGNORECASE,
)
