"""Free, rule-based hard filters for the matching pipeline.

Every check returns a stable rejection reason so the graph can count why
creators were dropped without turning expected rejections into errors.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timedelta

from .brand_match import account_matches, normalize_for_match
from .matching_config import MatchingConfig
from .models import BrandProfile, KolProfile
from .thai import has_thai_script, is_lao_creator, is_thai_creator

OWN_BRAND_ACCOUNT = "own_brand_account"
COMPETITOR_ACCOUNT = "competitor_account"
SHOP_MEDIA_ACCOUNT = "shop_media_account"
NOT_THAI = "not_thai"
FOLLOWERS_OUT_OF_RANGE = "followers_out_of_range"
INACTIVE = "inactive"

# Matched anywhere in a handle segment (split on ".", "_", "-") or nickname
# word: "gadgetshop" counts as a shop. Thai has no word spaces, so Thai tokens
# must be substring matches.
_SHOP_MEDIA_TOKENS = (
    "shop",
    "store",
    "official",
    "media",
    "magazine",
    "ร้าน",
    "ข่าว",
    "สำนักข่าว",
)
# Short tokens hide inside ordinary words ("smallgirl", "itvlogs"), so they
# count only as a whole segment or word, or as a suffix after a real prefix
# ("gamernews", "channeltv"), never "small".
_SHORT_SHOP_MEDIA_TOKENS = ("mall", "news", "tv")
_SHORT_TOKEN_MIN_PREFIX = 3
_HANDLE_SPLIT_RE = re.compile(r"[._-]+")
_NICKNAME_SPLIT_RE = re.compile(r"[\s._\-|/·,()]+")
_THAI_LANGUAGES = frozenset({"th", "thai"})


def _word_is_shop_or_media(word: str) -> bool:
    word = normalize_for_match(word)
    if not word:
        return False
    if any(normalize_for_match(token) in word for token in _SHOP_MEDIA_TOKENS):
        return True
    return any(
        word == token
        or (
            word.endswith(token)
            and len(word) - len(token) >= _SHORT_TOKEN_MIN_PREFIX
        )
        for token in _SHORT_SHOP_MEDIA_TOKENS
    )


def _is_shop_or_media(kol: KolProfile) -> bool:
    words = [
        *_HANDLE_SPLIT_RE.split(kol.handle.casefold()),
        *_NICKNAME_SPLIT_RE.split((kol.nickname or "").casefold()),
    ]
    return any(_word_is_shop_or_media(word) for word in words)


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

    Order matches the design: own account, competitor, shop/media, Thai,
    follower range, activity. ``brief`` is part of the boundary so callers can
    pass the same brief used to build the aliases; the rules themselves read
    only the aliases and config.

    ``check_activity=False`` skips the activity rule. The graph uses it before
    enrichment, because search hits are not a creator's newest videos, and
    applies the full filter again once profiles are enriched.
    """
    if account_matches(kol.handle, brand_aliases):
        return OWN_BRAND_ACCOUNT
    if account_matches(kol.handle, competitor_aliases):
        return COMPETITOR_ACCOUNT
    if _is_shop_or_media(kol):
        return SHOP_MEDIA_ACCOUNT
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
