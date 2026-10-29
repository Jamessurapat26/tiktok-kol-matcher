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
from .thai import has_thai_script, is_thai_creator

OWN_BRAND_ACCOUNT = "own_brand_account"
COMPETITOR_ACCOUNT = "competitor_account"
SHOP_MEDIA_ACCOUNT = "shop_media_account"
NOT_THAI = "not_thai"
FOLLOWERS_OUT_OF_RANGE = "followers_out_of_range"
INACTIVE = "inactive"

# Matched against handle segments (split on ".", "_", "-") and against the
# nickname. Short English tokens are deliberately not word-boundary checked:
# "gadgetshop" and "gamernews" should count as shop/news accounts.
_SHOP_MEDIA_TOKENS = (
    "shop",
    "store",
    "official",
    "mall",
    "news",
    "media",
    "tv",
    "magazine",
    "ร้าน",
    "ข่าว",
    "สำนักข่าว",
)
_HANDLE_SPLIT_RE = re.compile(r"[._-]+")
_THAI_LANGUAGES = frozenset({"th", "thai"})


def _is_shop_or_media(kol: KolProfile) -> bool:
    normalized_tokens = tuple(normalize_for_match(token) for token in _SHOP_MEDIA_TOKENS)
    for segment in _HANDLE_SPLIT_RE.split(kol.handle.casefold()):
        normalized_segment = normalize_for_match(segment)
        if any(token in normalized_segment for token in normalized_tokens if token):
            return True
    nickname = normalize_for_match(kol.nickname or "")
    return any(token in nickname for token in normalized_tokens if token)


def _is_thai(kol: KolProfile) -> bool:
    """Thai profile evidence, with captions as a fallback for unknown origins.

    Captions are often templated or quoted, so they do not override a creator
    whose language and region both explicitly say they are not Thai.
    """
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
) -> str | None:
    """Return a rejection reason, or None when the creator passes every rule.

    Order matches the design: own account, competitor, shop/media, Thai,
    follower range, activity. ``brief`` is part of the boundary so callers can
    pass the same brief used to build the aliases; the rules themselves read
    only the aliases and config.
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

    newest = _newest_video_at(kol)
    if newest is not None and newest < now - timedelta(days=cfg.active_days):
        return INACTIVE
    return None
