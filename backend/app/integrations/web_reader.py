from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_WEBSITE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "website.md"


class WebReadError(RuntimeError):
    """Raised when a web page cannot be fetched or read."""


MAX_CONTENT_CHARS = 20_000

# Specific phrases only: a normal brand page may contain a small "Log in" link.
_LOGIN_WALL_MARKERS = (
    "log into facebook",
    "log in to facebook",
    "you must log in to continue",
    "เข้าสู่ระบบ facebook",
)
_UNAVAILABLE_MARKERS = (
    "this content isn't available",
    "this page isn't available",
    "content isn't available right now",
    "เนื้อหานี้ไม่พร้อมใช้งาน",
)


def blocked_page_reason(text: str) -> str | None:
    """Return why ``text`` is a login wall or unavailable page, or None.

    Only the specific phrases above are markers, so a readable page that happens
    to contain a small "Log in" link is not rejected.
    """
    normalized = text.casefold().replace("’", "'").replace("‘", "'")
    if any(marker in normalized for marker in _LOGIN_WALL_MARKERS):
        return "login wall"
    if any(marker in normalized for marker in _UNAVAILABLE_MARKERS):
        return "content not available"
    return None


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
        # The mock ignores the URL and always returns the example-brand-like fixture.
        return self._fixture_path.read_text(encoding="utf-8")
