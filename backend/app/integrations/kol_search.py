import json
from pathlib import Path
from typing import Protocol

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "kols.json"


class KolSearch(Protocol):
    def search(self, queries: list[str], round: int) -> list[dict]:
        """Search for KOLs and return raw provider records for this round."""
        ...


class MockKolSearch:
    """MOCK: replace with real implementation (Apify TikTok scraper)."""

    def __init__(self, fixtures: list[dict] | None = None) -> None:
        if fixtures is None:
            fixtures = json.loads(_KOL_FIXTURE.read_text(encoding="utf-8"))
        self.fixtures = fixtures

    def search(self, queries: list[str], round: int) -> list[dict]:
        # Deterministic mock: the search text is ignored; each fixture belongs to
        # exactly one query round so round 1 deliberately finds too few KOLs.
        return [
            dict(item) for item in self.fixtures if item.get("query_group") == round
        ]
