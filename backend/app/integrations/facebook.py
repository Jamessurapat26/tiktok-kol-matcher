import json
from pathlib import Path
from typing import Protocol

from ..models import FacebookPage

_FACEBOOK_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "facebook_page.json"


class FacebookScraper(Protocol):
    def fetch_page(self, url: str) -> FacebookPage:
        """Fetch a public Facebook page and return its profile fields."""
        ...


class MockFacebookScraper:
    """MOCK: replace with real implementation (Facebook page scraper)."""

    def __init__(self, fixture_path: Path = _FACEBOOK_FIXTURE) -> None:
        self._fixture_path = fixture_path

    def fetch_page(self, url: str) -> FacebookPage:
        # The mock ignores the URL and always returns the example-brand-like fixture.
        raw = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        return FacebookPage.model_validate(raw)
