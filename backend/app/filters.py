"""Free, rule-based hard filters for the matching pipeline.

Every check returns a stable rejection reason so the graph can count why
creators were dropped without turning expected rejections into errors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timedelta

from .brand_match import (
    MIN_ALIAS_CHARS,
    account_matches,
    normalize_for_match,
    normalized_aliases,
)
from .matching_config import MatchingConfig
from .models import BrandProfile, KolProfile
from .thai import has_thai_script, is_lao_creator, is_thai_creator

OWN_BRAND_ACCOUNT = "own_brand_account"
COMPETITOR_ACCOUNT = "competitor_account"
SELLER_ACCOUNT = "seller_account"
MEDIA_ACCOUNT = "media_account"
NOT_THAI = "not_thai"
FOLLOWERS_OUT_OF_RANGE = "followers_out_of_range"
INACTIVE = "inactive"

# Matched anywhere in a handle segment (split on ".", "_", "-") or nickname
# word: "gadgetshop" counts as a shop. Thai has no word spaces, so Thai tokens
# must be substring matches.
_SELLER_TOKENS = ("shop", "store", "official", "ร้าน")
_MEDIA_TOKENS = ("media", "magazine", "ข่าว", "สำนักข่าว")
# Short tokens hide inside ordinary words ("smallgirl", "itvlogs"), so they
# count only as a whole segment or word, or as a suffix after a real prefix
# ("gamernews", "channeltv"), never "small".
_SHORT_SELLER_TOKENS = ("mall",)
_SHORT_MEDIA_TOKENS = ("news", "tv")
_SHORT_TOKEN_MIN_PREFIX = 3
# Phrases that state the account's job. Broad words such as "ร้าน" and "ขาย"
# stay out: reviewers use them too ("ทุกร้านที่เราโพส"), and the judge decides
# the unclear accounts.
_SELLER_PHRASES = ("ตัวแทนจำหน่าย", "ศูนย์จำหน่าย", "พนักงานขาย", "ฝ่ายขาย")
# "เซลล์" is a salesperson, unless it is part of a skin-care or solar term.
_SALESPERSON_RE = re.compile(
    r"(?<!สเต็ม)(?<!โซลาร์)(?<!โซล่าร์)(?<!โซล่า)"
    r"เซลล์(?!ผิว|ต้นกำเนิด|แสงอาทิตย์|มะเร็ง|ประสาท|สมอง|ผม|ร่างกาย)"
)
_HANDLE_SPLIT_RE = re.compile(r"[._-]+")
_NICKNAME_SPLIT_RE = re.compile(r"[\s._\-|/·,()]+")
_THAI_LANGUAGES = frozenset({"th", "thai"})


def _word_has_token(
    word: str,
    tokens: tuple[str, ...],
    short_tokens: tuple[str, ...],
) -> bool:
    word = normalize_for_match(word)
    if not word:
        return False
    if any(normalize_for_match(token) in word for token in tokens):
        return True
    return any(
        word == token
        or (
            word.endswith(token)
            and len(word) - len(token) >= _SHORT_TOKEN_MIN_PREFIX
        )
        for token in short_tokens
    )


def _identity_words(kol: KolProfile) -> list[str]:
    return [
        *_HANDLE_SPLIT_RE.split(kol.handle.casefold()),
        *_NICKNAME_SPLIT_RE.split((kol.nickname or "").casefold()),
    ]


def _identity_names_business(kol: KolProfile, aliases: Iterable[str]) -> bool:
    """True when the handle or nickname carries a business's name.

    Short aliases hide inside ordinary words ("dime" in "dimension"), so they
    must be a whole handle segment or nickname word.
    """
    words = {normalize_for_match(word) for word in _identity_words(kol)}
    handle = normalize_for_match(kol.handle)
    nickname = normalize_for_match(kol.nickname or "")
    for raw in aliases:
        alias = normalize_for_match(str(raw))
        if not alias:
            continue
        if alias in words:
            return True
        if len(alias) >= MIN_ALIAS_CHARS and (alias in handle or alias in nickname):
            return True
    return False


def is_seller_account(
    kol: KolProfile,
    brief: BrandProfile,
    brand_aliases: Iterable[str],
    competitor_aliases: Iterable[str],
) -> bool:
    """True when the profile plainly belongs to a seller.

    Only clear signals count here. Accounts that need judgement, such as a
    farmer advertising their own service, are left to the LLM judge.
    """
    if any(
        _word_has_token(word, _SELLER_TOKENS, _SHORT_SELLER_TOKENS)
        for word in _identity_words(kol)
    ):
        return True
    profile_text = " ".join(part for part in (kol.nickname, kol.bio) if part)
    if any(phrase in profile_text for phrase in _SELLER_PHRASES):
        return True
    if _SALESPERSON_RE.search(profile_text):
        return True
    return _identity_names_business(
        kol,
        (
            *brand_aliases,
            *competitor_aliases,
            *normalized_aliases(brief.retailers),
        ),
    )


def _is_media(kol: KolProfile) -> bool:
    return any(
        _word_has_token(word, _MEDIA_TOKENS, _SHORT_MEDIA_TOKENS)
        for word in _identity_words(kol)
    )


def _is_thai(kol: KolProfile) -> bool:
    """Thai profile evidence, with captions as a fallback for unknown origins.

    Captions are often templated or quoted, so they do not override a creator
    whose language and region both explicitly say they are not Thai.
    """
    if is_lao_creator(kol):
        return False
    if is_thai_creator(kol):
        return True
    language_is_thai = (kol.language or "").casefold() in _THAI_LANGUAGES
    region_is_thai = (kol.region or "").upper() == "TH"
    if kol.language and kol.region and not language_is_thai and not region_is_thai:
        return False
    return any(has_thai_script(video.caption) for video in kol.videos)


def _newest_video_at(kol: KolProfile) -> datetime | None:
    dates = [video.posted_at for video in kol.videos if video.posted_at is not None]
    return max(dates) if dates else None


def hard_filter(
    kol: KolProfile,
    brief: BrandProfile,
    brand_aliases: Iterable[str],
    competitor_aliases: Iterable[str],
    cfg: MatchingConfig,
    now: datetime,
    *,
    check_activity: bool = True,
) -> str | None:
    """Return a rejection reason, or None when the creator passes every rule.

    Order matches the design: own account, competitor, seller, media, Thai,
    follower range, activity. Besides the aliases and config, the rules read
    the retailers named in ``brief``.

    ``check_activity=False`` skips the activity rule. The graph uses it before
    enrichment, because search hits are not a creator's newest videos, and
    applies the full filter again once profiles are enriched.
    """
    if account_matches(kol.handle, brand_aliases):
        return OWN_BRAND_ACCOUNT
    if account_matches(kol.handle, competitor_aliases):
        return COMPETITOR_ACCOUNT
    if is_seller_account(kol, brief, brand_aliases, competitor_aliases):
        return SELLER_ACCOUNT
    if _is_media(kol):
        return MEDIA_ACCOUNT
    if not _is_thai(kol):
        return NOT_THAI
    if not cfg.followers_min <= kol.followers <= cfg.followers_max:
        return FOLLOWERS_OUT_OF_RANGE

    if not check_activity:
        return None
    newest = _newest_video_at(kol)
    if newest is not None and newest < now - timedelta(days=cfg.active_days):
        return INACTIVE
    return None
