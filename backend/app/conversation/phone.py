"""German phone number normalization for voice input."""

import re

GERMAN_DIGIT_WORDS = {
    "null": "0",
    "eins": "1",
    "ein": "1",
    "zwei": "2",
    "zwo": "2",
    "drei": "3",
    "vier": "4",
    "fünf": "5",
    "fuenf": "5",
    "funf": "5",
    "sechs": "6",
    "sieben": "7",
    "acht": "8",
    "neun": "9",
}

_DIGIT_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(GERMAN_DIGIT_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize_phone_text(text: str) -> str:
    """Convert German spoken phone text to a normalized digit string.

    Handles: digit words (vier → 4), plus prefix, commas between digits,
    and German +49 normalization.
    """
    t = text.lower()
    t = _DIGIT_WORD_RE.sub(lambda m: GERMAN_DIGIT_WORDS[m.group(1).lower()], t)
    t = t.replace("plus", "+")
    t = t.replace(",", " ")
    t = re.sub(r"[^\d+]", "", t)

    if not t:
        return ""
    if t.startswith("+49"):
        return t
    if t.startswith("49") and len(t) >= 9:
        return "+" + t
    if t.startswith("0"):
        return "+49" + t[1:]
    return t
