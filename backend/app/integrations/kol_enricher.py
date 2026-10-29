import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..models import KolProfile, normalize_handle, outranks
from .tiktok_videos import video_from_days_ago

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kols.json"
_SUMMARY_CHARS = 400
_WEEK_SECONDS = 7 * 24 * 60 * 60


class KolEnricher(Protocol):
    def enrich(self, kol: KolProfile) -> dict[str, Any]:
        """Return KolProfile field updates for one KOL."""
        ...


class MockKolEnricher:
    """MOCK: replace with real implementation (Apify profile/posts scraper)."""

    def __init__(self, fixtures: list[dict] | None = None) -> None:
        if fixtures is None:
            fixtures = json.loads(_KOL_FIXTURE.read_text(encoding="utf-8"))
        self.fixtures = fixtures
        self._by_handle: dict[str, dict] = {}
        for item in fixtures:
            handle = normalize_handle(str(item.get("handle", "")))
            if not handle:
                continue
            current = self._by_handle.get(handle)
            # Strictly higher priority replaces the stored row. Equal priority
            # keeps the earlier fixture.
            if current is None or outranks(
                str(item.get("source", "")),
                str(current.get("source", "")),
            ):
                self._by_handle[handle] = item

    def enrich(self, kol: KolProfile) -> dict[str, Any]:
        raw = self._by_handle.get(normalize_handle(kol.handle), {})
        raw_videos = [
            video
            for video in raw.get("profile_videos", [])
            if isinstance(video, dict)
        ]
        if not raw_videos:
            return {}

        now = datetime.now(UTC)
        videos = [video_from_days_ago(video, now) for video in raw_videos]
        update: dict[str, Any] = {
            "videos": videos,
            "stats_source": "profile",
        }
        per_week = _posts_per_week(videos)
        if per_week is not None:
            update["posts_per_week"] = per_week
        summary = _content_summary(videos)
        if summary:
            update["content_summary"] = summary
        return update


def _posts_per_week(videos: list[dict[str, Any]]) -> float | None:
    """``(n - 1) / weeks`` across the fixture's dated profile videos."""
    stamps = [
        datetime.fromisoformat(str(video["posted_at"])).timestamp()
        for video in videos
        if video.get("posted_at")
    ]
    if len(stamps) < 2:
        return None
    span = max(stamps) - min(stamps)
    if span <= 0:
        return None
    return round((len(stamps) - 1) / (span / _WEEK_SECONDS), 2)


def _content_summary(videos: list[dict[str, Any]]) -> str | None:
    captions = [
        str(video["caption"]).strip()
        for video in videos
        if str(video.get("caption") or "").strip()
    ]
    if not captions:
        return None
    summary = "Recent posts: " + " | ".join(captions[:3])
    return summary[:_SUMMARY_CHARS].rstrip()
