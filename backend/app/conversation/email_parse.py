"""Spoken-email parsing: turn noisy STT dictation into an address.

Pure functions, no conversation state — extracted from the manager so the
email grammar (connectors, chunked local parts, letter-by-letter spelling, and
name-anchoring) lives in one focused place.
"""

from __future__ import annotations

import re

from app.validation import is_valid_email

_DOMAIN_CORRECTIONS = {
    "smail": "gmail",
    "gmeil": "gmail",
    "g-mail": "gmail",
    "geemail": "gmail",
    "gemail": "gmail",
    "hotmeil": "hotmail",
    "hotemail": "hotmail",
}

# Spoken lead-in before the address proper ("meine E-Mail-Adresse lautet …"),
# stripped so it isn't glued onto the local part.
_EMAIL_LEADIN_RE = re.compile(
    r"^(?:\s*\b(?:ja|nein|also|genau|ähm|äh|meine?|die|das|es|und|hier|war|ich|"
    r"e[-\s]?mail|email|mail|adresse|lautet|ist|wäre|my|the|email|address|it'?s|is)\b"
    r"[\s.:,!?-]*)+",
    re.IGNORECASE,
)


def parse_email(text: str) -> str | None:
    """Convert a spoken email to an address.

    Spoken emails arrive in chunks with stray spaces ("Lang M at gmail punkt com")
    and the local part is often dictated piece by piece. We strip a spoken lead-in,
    map connectors to symbols, then join the remaining whitespace so a
    space-separated local part ("lang m") becomes one token ("langm") instead of
    being truncated to whichever fragment happened to carry the '@'.
    """
    t = _EMAIL_LEADIN_RE.sub("", text.lower().strip())
    t = re.sub(r"\s*(?:\bat\b|\bät\b|@)\s*", "@", t)
    t = re.sub(r"\s*(?:\bpunkt\b|\bdot\b|\bpoint\b)\s*", ".", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    t = re.sub(r"\s+", "", t).strip(".,;:!?")
    t = t.replace("@www.", "@")
    # When a spoken "at" sat between dots ("baum.at.gmail.com" / "baum at gmail
    # dot com"), the connector→@ swap leaves a stray dot touching the @. Collapse
    # ".@" / "@." so it doesn't produce an invalid local part ("baum.@gmail.com").
    t = re.sub(r"\.*@\.*", "@", t)
    if "@" not in t:
        return None
    local, _, domain = t.partition("@")
    parts = [p for p in domain.split(".") if p]
    if local and parts:
        parts[0] = _DOMAIN_CORRECTIONS.get(parts[0], parts[0])
    candidate = f"{local}@{'.'.join(parts)}"
    return candidate if is_valid_email(candidate) else None


# Spoken letter names STT emits when a caller spells aloud, German first then
# common English renderings. Single vowels (a/e/i/o/u) are letters as-is.
_LETTER_NAMES = {
    "be": "b", "ce": "c", "de": "d", "ef": "f", "ge": "g", "ha": "h",
    "jot": "j", "ka": "k", "el": "l", "em": "m", "en": "n", "pe": "p",
    "ku": "q", "er": "r", "es": "s", "te": "t", "vau": "v", "we": "w",
    "weh": "w", "ix": "x", "ypsilon": "y", "zett": "z", "ah": "a", "oh": "o",
    "ay": "a", "bee": "b", "cee": "c", "see": "c", "dee": "d", "ee": "e",
    "eff": "f", "gee": "g", "aitch": "h", "jay": "j", "kay": "k", "ell": "l",
    "oh ": "o", "pee": "p", "cue": "q", "queue": "q", "ar": "r", "are": "r",
    "ess": "s", "tee": "t", "yu": "u", "vee": "v", "ex": "x", "wy": "y",
    "why": "y", "zee": "z", "zed": "z",
}  # fmt: skip
# "wie/für/as in/like" introduce an example word ("R wie Richard"); the letter is
# what precedes them. "doppel/double X" means the letter X twice.
_SPELL_EXAMPLE_RE = re.compile(r"\b([a-zäöü])\s+(?:wie|für|as\s+in|like|for)\s+\w+", re.IGNORECASE)
_SPELL_DOUBLE_RE = re.compile(r"\b(?:doppel|double)[\s-]*([a-zäöü])\b", re.IGNORECASE)


def parse_spelled_local(text: str) -> str | None:
    """Reconstruct an email local part dictated letter by letter.

    Handles the conventions callers actually use when the address is being
    spelled: bare letters ("r i t t e r"), letter names STT renders as words
    ("er i te te e er"), phonetic examples ("R wie Richard"), and doubling
    ("doppel T"). Returns the assembled local part, or None when fewer than two
    letters were recognised — that ambiguous case is left to the LLM fallback.
    """
    t = _EMAIL_LEADIN_RE.sub("", text.lower().strip())
    t = _SPELL_EXAMPLE_RE.sub(r"\1", t)
    t = _SPELL_DOUBLE_RE.sub(r"\1 \1", t)
    letters: list[str] = []
    for tok in re.split(r"[\s,.;:!?]+", t):
        if not tok:
            continue
        if len(tok) == 1 and tok.isalpha():
            letters.append(tok)
        elif tok in _LETTER_NAMES:
            letters.append(_LETTER_NAMES[tok])
    return "".join(letters) if len(letters) >= 2 else None


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two short strings (iterative, O(len*len))."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anchor_email_to_name(email: str, name: str) -> str:
    """Rewrite a local part that is a near-miss of the caller's name.

    Spoken email over a phone line loses letters ('sigma' for 'Sigmar'). When the
    captured name is known and the local part is a close match (edit distance 1-2)
    to a name-derived form, snap it to that form. Only near-matches are touched, so
    a genuinely different address ('leon.legal') is left alone — and the scripted
    read-back still lets the caller reject it.
    """
    if not name or "@" not in email:
        return email
    local, _, domain = email.partition("@")
    if len(local) < 3:
        return email
    parts = [p for p in re.split(r"\s+", name.lower().strip()) if p]
    if not parts:
        return email
    candidates = {parts[0], parts[-1], "".join(parts), ".".join(parts)}
    if len(parts) >= 2:
        candidates.add(f"{parts[0]}.{parts[-1]}")
    best, best_d = None, 99
    for cand in candidates:
        d = _levenshtein(local, cand)
        if d < best_d:
            best, best_d = cand, d
    if best and 0 < best_d <= 2 and best_d < len(local):
        return f"{best}@{domain}"
    return email
