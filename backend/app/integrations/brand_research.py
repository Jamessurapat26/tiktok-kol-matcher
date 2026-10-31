from dataclasses import dataclass
from typing import Protocol

from ..models import FacebookPage


class WebSearchError(RuntimeError):
    """The web search request failed or returned output we cannot use."""


class WebSearchConfigError(WebSearchError):
    """Live web search settings are missing."""


@dataclass(frozen=True)
class BrandResearch:
    """Notes from web search. ``notes`` is None when the web gave no evidence."""

    notes: str | None = None
    searches: int = 0
    llm_calls: int = 0


class BrandResearcher(Protocol):
    def research(
        self,
        website_url: str,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandResearch:
        """Search the web for facts about the brand the scraped pages leave out."""
        ...


class MockBrandResearcher:
    """MOCK: replace with real implementation (web search agent)."""

    def research(
        self,
        website_url: str,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandResearch:
        return BrandResearch()
