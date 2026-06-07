"""Deterministic policies for safety-critical caller requests."""

import re

_DE_PERSON_REQUESTS = (
    re.compile(
        r"\b(?:frau|herr)\s+[\wäöüß-]+\b.*\b(?:sprechen|verbinden|durchstellen)\b",
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


def is_explicit_handoff_request(text: str, lang: str = "de") -> bool:
    """Return whether the caller explicitly asks to speak with a person."""
    patterns = _DE_PERSON_REQUESTS if lang == "de" else _EN_PERSON_REQUESTS
    return any(pattern.search(text) for pattern in patterns)
