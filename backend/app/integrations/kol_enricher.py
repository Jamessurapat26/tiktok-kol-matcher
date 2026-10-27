import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from ..models import KolProfile, normalize_handle, outranks

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kols.json"


def _handle_seed(handle: str) -> int:
    """Stable integer derived from a handle (never Python's salted hash)."""
    digest = hashlib.sha256(normalize_handle(handle).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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
        seed = _handle_seed(kol.handle)

        fixture_rate = raw.get("engagement_rate")
        if (
            isinstance(fixture_rate, (int, float))
            and 0 <= float(fixture_rate) <= 1
        ):
            engagement_rate = round(float(fixture_rate), 4)
        else:
            engagement_rate = round(0.01 + (seed % 90) / 1000, 4)

        update: dict[str, Any] = {
            "median_engagement_rate": engagement_rate,
            "posts_per_week": round(1.0 + (seed % 50) / 10, 1),
        }
        topics = [str(topic) for topic in raw.get("topics", []) if str(topic)]
        if topics:
            update["content_summary"] = "Topics: " + ", ".join(topics)

        return update
