from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_WEBSITE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "website.md"


class WebReadError(RuntimeError):
    """Raised when a web page cannot be fetched or read."""


MAX_CONTENT_CHARS = 20_000


def truncate(text: str) -> str:
    """Cut text to MAX_CONTENT_CHARS characters and strip surrounding whitespace."""
    return text[:MAX_CONTENT_CHARS].strip()


@dataclass(frozen=True)
class WebPage:
    content: str
    source: str


class WebReader(Protocol):
    def read(self, url: str) -> str:
        """Return the readable text content of a web page."""
        ...


class MockWebReader:
    """MOCK: replace with real implementation (HTTP fetch + text extraction)."""

    def __init__(self, fixture_path: Path = _WEBSITE_FIXTURE) -> None:
        self._fixture_path = fixture_path

    def read(self, url: str) -> str:
        # The mock ignores the URL and always returns the Ophtus-like fixture.
        return self._fixture_path.read_text(encoding="utf-8")
