"""Decide whether a discovered creator is Thai enough to keep."""

from collections.abc import Iterable

from .models import KolProfile

THAI_LANGUAGE = frozenset({"th", "thai"})
THAI_REGION = "TH"
LAO_LANGUAGE = frozenset({"lo", "lao"})
LAO_REGION = "LA"
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


def _script_counts(texts: Iterable[str | None]) -> tuple[int, int]:
    """Count Thai (U+0E00-U+0E7F) and Lao (U+0E80-U+0EFF) characters."""
    thai = lao = 0
    for text in texts:
        for char in text or "":
            if "\u0e00" <= char <= "\u0e7f":
                thai += 1
            elif "\u0e80" <= char <= "\u0eff":
                lao += 1
    return thai, lao


def is_lao_creator(kol: KolProfile) -> bool:
    """True when Lao script dominates the profile, or Lao signals are unopposed.

    Lao text sits next to Thai in Unicode and is often tagged as Thai, so the
    script itself is the strongest signal. A Lao region or language counts
    unless the creator's text is mostly Thai.
    """
    thai, lao = _script_counts(
        (
            kol.nickname,
            kol.bio,
            kol.content_summary,
            *(video.caption for video in kol.videos),
        )
    )
    if lao >= THAI_SCRIPT_MIN_CHARS and lao > thai:
        return True
    lao_signal = (kol.region or "").upper() == LAO_REGION or (
        kol.language or ""
    ).casefold() in LAO_LANGUAGE
    return lao_signal and thai <= lao


def is_thai_creator(kol: KolProfile) -> bool:
    """True when language, region, or any profile text shows a Thai creator.

    Lao creators are never Thai, even when TikTok tags their text as Thai.
    """
    if is_lao_creator(kol):
        return False
    if (kol.language or "").casefold() in THAI_LANGUAGE:
        return True
    if (kol.region or "").upper() == THAI_REGION:
        return True
    return any(
        has_thai_script(text)
        for text in (kol.nickname, kol.bio, kol.content_summary)
    )
