"""Live TikTok profile enrichment via ``clockworks/tiktok-profile-scraper``.

Store page: https://apify.com/clockworks/tiktok-profile-scraper

Recent videos for one handle are fetched with downloads, subtitles, and comment
scrapes turned off, and with pinned posts excluded so they are not billed.
Results are cached as JSON under ``data/cache/apify`` (or ``$KOL_CACHE_DIR/apify``)
for 7 days. A hit does not call Apify.

Engagement
----------
For each non-pinned video (pinned videos are used only when every returned
video is pinned):

    rate = (diggCount + commentCount + shareCount) / playCount

``diggCount`` is TikTok's like count. Saves (``collectCount``) are not included.
When ``playCount`` is missing or zero, that video falls back to the same
numerator divided by the profile's follower count (``authorMeta.fans``, or the
discovered ``KolProfile.followers`` when the actor omits fans). Videos with no
counts, and no follower count to divide by, are skipped.

The stored ``median_engagement_rate`` is the median of those per-video rates,
capped to ``[0, 1]`` because ``KolProfile.median_engagement_rate`` is a fraction.
``statistics.median`` averages the two middle values when the count is even.

``posts_per_week`` is ``(n - 1) / weeks`` between the oldest and newest
timestamp in that same video set (``createTime``, else ``createTimeISO``).
Fewer than two timestamps leaves the field unset.

``content_summary`` is a short line of up to three captions plus the hashtag
names on those videos, cut to 400 characters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ..models import KolProfile, normalize_handle
from .apify import ApifyClient, ApifyError, as_count, dig

PROFILE_ACTOR = "clockworks/tiktok-profile-scraper"
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
SUMMARY_CHARS = 400
_WEEK_SECONDS = 7 * 24 * 60 * 60

_QUIET_INPUT: dict[str, Any] = {
    "profileScrapeSections": ["videos"],
    "profileSorting": "latest",
    # Pinned posts are old and would skew cadence. The calculator also ignores
    # isPinned when any other video is present, for payloads that still include them.
    "excludePinnedPosts": True,
    "shouldDownloadVideos": False,
    "shouldDownloadCovers": False,
    "shouldDownloadSlideshowImages": False,
    "shouldDownloadAvatars": False,
    "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
    "commentsPerPost": 0,
}


def profile_input(handle: str, *, max_videos: int) -> dict[str, Any]:
    """Actor input for one profile. Media downloads stay off."""
    return {
        "profiles": [handle],
        "resultsPerPage": max_videos,
        **_QUIET_INPUT,
    }


def engagement_rate(item: dict[str, Any], *, followers: int) -> float | None:
    """Per-video rate. Views first; follower count only when views are missing.

    The returned rate is not capped. ``enrich_from_items`` caps the median.
    """
    likes = as_count(item.get("diggCount"))
    comments = as_count(item.get("commentCount"))
    shares = as_count(item.get("shareCount"))
    if likes is None and comments is None and shares is None:
        return None
    interactions = (likes or 0) + (comments or 0) + (shares or 0)
    views = as_count(item.get("playCount"))
    if views is not None and views > 0:
        return interactions / views
    fans = as_count(dig(item, "authorMeta", "fans"))
    denominator = fans if fans else followers
    if denominator > 0:
        return interactions / denominator
    return None


def posts_per_week(videos: list[dict[str, Any]]) -> float | None:
    """``(n - 1) / weeks`` across videos that have a timestamp."""
    stamps = [stamp for stamp in (_timestamp(video) for video in videos) if stamp is not None]
    if len(stamps) < 2:
        return None
    span = max(stamps) - min(stamps)
    if span <= 0:
        return None
    weeks = span / _WEEK_SECONDS
    return round((len(stamps) - 1) / weeks, 2)


def content_summary(videos: list[dict[str, Any]], *, limit: int = SUMMARY_CHARS) -> str | None:
    """Captions and hashtag names from the videos used for the engagement median."""
    captions: list[str] = []
    tags: list[str] = []
    seen_tags: set[str] = set()
    for video in videos:
        text = video.get("text")
        if isinstance(text, str):
            caption = re.sub(r"\s+", " ", text).strip()
            if caption:
                captions.append(caption[:140])
        for tag in video.get("hashtags") or []:
            name = tag.get("name") if isinstance(tag, dict) else None
            if not isinstance(name, str) or not name.strip():
                continue
            cleaned = name.strip().lstrip("#")
            key = cleaned.casefold()
            if not cleaned or key in seen_tags:
                continue
            seen_tags.add(key)
            tags.append(cleaned)

    parts: list[str] = []
    if captions:
        parts.append("Recent posts: " + " | ".join(captions[:3]))
    if tags:
        parts.append("Hashtags: " + ", ".join(f"#{tag}" for tag in tags[:8]))
    if not parts:
        return None
    summary = ". ".join(parts)
    if len(summary) <= limit:
        return summary
    return summary[:limit].rstrip()


def enrich_from_items(items: list[Any], *, followers: int) -> dict[str, Any]:
    """Turn profile-scraper dataset items into KolProfile field updates."""
    videos = _videos_for_stats(items)
    rates = [
        rate
        for video in videos
        if (rate := engagement_rate(video, followers=followers)) is not None
    ]
    update: dict[str, Any] = {}
    if rates:
        median = statistics.median(rates)
        update["median_engagement_rate"] = round(min(1.0, max(0.0, float(median))), 4)
    per_week = posts_per_week(videos)
    if per_week is not None:
        update["posts_per_week"] = per_week
    summary = content_summary(videos)
    if summary:
        update["content_summary"] = summary
    return update


def _videos_for_stats(items: list[Any]) -> list[dict[str, Any]]:
    videos = [
        item
        for item in items
        if isinstance(item, dict) and not item.get("errorCode") and _looks_like_video(item)
    ]
    unpinned = [video for video in videos if not video.get("isPinned")]
    return unpinned or videos


def _looks_like_video(item: dict[str, Any]) -> bool:
    return any(
        key in item
        for key in ("diggCount", "playCount", "commentCount", "shareCount", "text", "id")
    )


def _timestamp(item: dict[str, Any]) -> float | None:
    raw = item.get("createTime")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    iso = item.get("createTimeISO")
    if isinstance(iso, str) and iso.strip():
        try:
            return datetime.fromisoformat(iso.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def default_apify_cache_dir() -> Path:
    """Apify cache: ``$KOL_CACHE_DIR/apify`` or ``<repo>/data/cache/apify``."""
    override = os.environ.get("KOL_CACHE_DIR")
    if override:
        return Path(override) / "apify"
    # This file is backend/app/integrations/kol_enricher_live.py.
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "apify"


def _cache_path(cache_dir: Path, handle: str) -> Path:
    digest = hashlib.sha256(normalize_handle(handle).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_cached_update(
    path: Path,
    *,
    now: float,
    ttl_seconds: float,
) -> dict[str, Any] | None:
    """Return a cached update, an empty dict for a cached miss, or None to refetch."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = payload["fetched_at"]
        update = payload["update"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)):
        return None
    if not isinstance(update, dict):
        return None
    if now - float(fetched_at) > ttl_seconds:
        return None
    return _clean_update(update)


def _clean_update(update: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    rate = update.get("median_engagement_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 <= float(rate) <= 1:
        clean["median_engagement_rate"] = float(rate)
    posts = update.get("posts_per_week")
    if isinstance(posts, (int, float)) and not isinstance(posts, bool) and float(posts) >= 0:
        clean["posts_per_week"] = float(posts)
    summary = update.get("content_summary")
    if isinstance(summary, str) and summary.strip():
        clean["content_summary"] = summary.strip()
    return clean


class ApifyKolEnricher:
    """Fill engagement, cadence, and a caption summary from recent profile videos."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        max_videos: int = 10,
        timeout: float = 180.0,
        max_retries: int = 1,
        backoff: float = 1.0,
        max_charge_usd: float = 0.50,
        cache_dir: Path | None = None,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        sleeper: Any = None,
    ) -> None:
        if max_videos < 1:
            raise ValueError("max_videos must be at least 1")
        self.max_videos = max_videos
        self._cache_dir = cache_dir if cache_dir is not None else default_apify_cache_dir()
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        client_kwargs: dict[str, Any] = {
            "client": client,
            "timeout": timeout,
            "max_retries": max_retries,
            "backoff": backoff,
            "max_charge_usd": max_charge_usd,
        }
        if sleeper is not None:
            client_kwargs["sleeper"] = sleeper
        self._apify = ApifyClient(token, **client_kwargs)

    def enrich(self, kol: KolProfile) -> dict[str, Any]:
        """Return engagement updates for ``kol``. Cache hits skip the network."""
        cached = _load_cached_update(
            _cache_path(self._cache_dir, kol.handle),
            now=self._clock(),
            ttl_seconds=self._ttl_seconds,
        )
        if cached is not None:
            return cached

        handle = normalize_handle(kol.handle)
        if not handle:
            return {}

        items = self._apify.run_sync_get_items(
            PROFILE_ACTOR,
            profile_input(handle, max_videos=self.max_videos),
            max_items=self.max_videos,
        )
        error_items = [
            item for item in items if isinstance(item, dict) and item.get("errorCode")
        ]
        usable = _videos_for_stats(items)
        if not usable and error_items:
            first = error_items[0]
            code = first.get("errorCode")
            detail = first.get("error") or code
            raise ApifyError(f"{code}: {detail}")

        update = enrich_from_items(items, followers=kol.followers)
        self._write_cache(kol.handle, update)
        return update

    def _write_cache(self, handle: str, update: dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(self._cache_dir, handle)
        payload = {
            "handle": normalize_handle(handle),
            "fetched_at": self._clock(),
            "update": update,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
