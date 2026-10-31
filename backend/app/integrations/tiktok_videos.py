"""Map TikTok dataset items (and demo fixtures) to ``VideoStats`` dictionaries.

Both the search mapper and the live profile enricher keep one video per item.
Live items carry counts and either ``createTime`` (epoch seconds) or
``createTimeISO``. Demo fixtures instead store ``days_ago`` so the sample data
never goes stale. Everything here returns plain dictionaries that
``VideoStats`` can validate.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .apify import as_count

CAPTION_CHARS = 300


def posted_at_from_item(item: dict[str, Any]) -> str | None:
    """Return a tz-aware UTC ISO timestamp from ``createTime`` or ``createTimeISO``."""
    raw = item.get("createTime")
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.isdigit():
            raw = int(stripped)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return _iso(datetime.fromtimestamp(float(raw), UTC))
    iso = item.get("createTimeISO")
    if isinstance(iso, str) and iso.strip():
        try:
            parsed = datetime.fromisoformat(iso.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return _iso(parsed)
    return None


def video_url_from_item(item: dict[str, Any], handle: str | None) -> str | None:
    """Prefer the canonical video URL, else build it from the author and video id."""
    raw = item.get("webVideoUrl")
    if isinstance(raw, str) and raw.startswith("http"):
        return raw.split("?", 1)[0]
    video_id = item.get("id")
    if (
        handle
        and isinstance(video_id, (str, int))
        and not isinstance(video_id, bool)
        and str(video_id).strip()
    ):
        return f"https://www.tiktok.com/@{handle.lstrip('@')}/video/{video_id}"
    return None


def caption_from_item(item: dict[str, Any]) -> str:
    """Collapse whitespace in a video caption and keep it reasonably short."""
    raw = item.get("text")
    if not isinstance(raw, str):
        return ""
    return re.sub(r"\s+", " ", raw).strip()[:CAPTION_CHARS].rstrip()


def hashtags_from_item(item: dict[str, Any]) -> list[str]:
    """Hashtag names on one video, without the leading ``#``, once each."""
    raw = item.get("hashtags")
    tags: list[str] = []
    seen: set[str] = set()
    for tag in raw if isinstance(raw, list) else []:
        name = tag.get("name") if isinstance(tag, dict) else tag
        if not isinstance(name, str):
            continue
        cleaned = name.strip().lstrip("#")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            tags.append(cleaned)
    return tags


def video_from_item(item: dict[str, Any], handle: str | None) -> dict[str, Any]:
    """Build a ``VideoStats``-compatible row from one live dataset item."""
    return {
        "url": video_url_from_item(item, handle),
        "views": as_count(item.get("playCount")) or 0,
        "likes": as_count(item.get("diggCount")) or 0,
        "comments": as_count(item.get("commentCount")) or 0,
        "shares": as_count(item.get("shareCount")) or 0,
        "posted_at": posted_at_from_item(item),
        "caption": caption_from_item(item),
        "hashtags": hashtags_from_item(item),
        "is_ad": item.get("isAd") is True,
    }


def video_from_days_ago(video: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Build a ``VideoStats``-compatible row from a demo fixture video."""
    days = video.get("days_ago")
    posted: str | None
    if days is None:
        raw_posted = video.get("posted_at")
        posted = raw_posted if isinstance(raw_posted, str) and raw_posted else None
    else:
        try:
            offset = max(0, int(days))
        except (TypeError, ValueError):
            offset = 0
        posted = _iso(now - timedelta(days=offset))
    row: dict[str, Any] = {
        "url": video.get("url"),
        "views": _count(video.get("views")),
        "likes": _count(video.get("likes")),
        "comments": _count(video.get("comments")),
        "shares": _count(video.get("shares")),
        "posted_at": posted,
        "caption": caption_from_item({"text": video.get("caption", "")}),
        "hashtags": hashtags_from_item(video),
        "is_ad": video.get("is_ad") is True,
    }
    return row


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (
        value.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _count(value: object) -> int:
    return as_count(value) or 0
