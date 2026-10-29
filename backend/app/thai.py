"""Decide whether a discovered creator is Thai enough to keep."""

from .models import KolProfile

THAI_LANGUAGE = frozenset({"th", "thai"})
THAI_REGION = "TH"
# One stray Thai character in an otherwise English profile is not evidence.
THAI_SCRIPT_MIN_CHARS = 2


def has_thai_script(text: str | None) -> bool:
    """True when ``text`` has at least two characters in U+0E00-U+0E7F."""
    if not text:
        return False
    count = 0
    for char in text:
        if "\u0e00" <= char <= "\u0e7f":
            count += 1
            if count >= THAI_SCRIPT_MIN_CHARS:
                return True
    return False


def is_thai_creator(kol: KolProfile) -> bool:
    """True when language, region, or any profile text shows a Thai creator."""
    if (kol.language or "").casefold() in THAI_LANGUAGE:
        return True
    if (kol.region or "").upper() == THAI_REGION:
        return True
    return any(
        has_thai_script(text)
        for text in (kol.nickname, kol.bio, kol.content_summary)
    )
