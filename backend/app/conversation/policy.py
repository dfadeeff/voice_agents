"""Deterministic policies for safety-critical caller requests."""

import re

_DE_PERSON_REQUESTS = (
    re.compile(
        r"\b(?:frau|herrn?)\s+[\wäöüß-]+\b.*\b(?:sprechen|verbinden|durchstellen)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:mit\s+)?(?:einem|einer|jemandem|dem|der)?\s*"
        r"(?:anwalt|anwältin|menschen|mitarbeiter|mitarbeiterin|person)\b"
        r".*\b(?:sprechen|reden|verbunden|durchgestellt)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:verbinden|durchstellen)\s+sie\s+mich\b", re.IGNORECASE),
)

_EN_PERSON_REQUESTS = (
    re.compile(
        r"\b(?:speak|talk)\s+(?:to|with)\s+(?:a|an|someone|mr|mrs|ms|miss)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:transfer|connect)\s+me\b", re.IGNORECASE),
)

_DE_PERSON_NAME_RE = re.compile(
    r"\b(frau|herrn?)\s+([\wäöüß-]+)\b",
    re.IGNORECASE,
)
_EN_PERSON_NAME_RE = re.compile(
    r"\b(mr|mrs|ms|miss)\s+(\w+)\b",
    re.IGNORECASE,
)
_EN_HONORIFICS = {"mr": "Mr", "mrs": "Mrs", "ms": "Ms", "miss": "Miss"}


def is_explicit_handoff_request(text: str, lang: str = "de") -> bool:
    """Return whether the caller explicitly asks to speak with a person."""
    patterns = _DE_PERSON_REQUESTS if lang == "de" else _EN_PERSON_REQUESTS
    return any(pattern.search(text) for pattern in patterns)


def extract_target_person(text: str, lang: str = "de") -> str | None:
    """Extract the requested person with honorific, e.g. 'Herrn Schulz' → 'Herr Schulz'."""
    pattern = _DE_PERSON_NAME_RE if lang == "de" else _EN_PERSON_NAME_RE
    match = pattern.search(text)
    if not match:
        return None
    honor_raw, name = match.group(1).lower(), match.group(2).title()
    if lang == "de":
        honor = "Frau" if honor_raw.startswith("frau") else "Herr"
    else:
        honor = _EN_HONORIFICS.get(honor_raw, honor_raw.title())
    return f"{honor} {name}"
