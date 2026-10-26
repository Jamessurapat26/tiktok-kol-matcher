from dataclasses import dataclass

from .embeddings import Embedder, MockEmbedder
from .facebook import FacebookScraper, MockFacebookScraper
from .kol_enricher import KolEnricher, MockKolEnricher
from .kol_search import KolSearch, MockKolSearch
from .llm import BrandAnalyzer, MockBrandAnalyzer
from .web_reader import MockWebReader, WebReader


@dataclass(frozen=True)
class Integrations:
    """External services used by the graph, grouped for easy injection."""

    web_reader: WebReader
    facebook: FacebookScraper
    llm: BrandAnalyzer
    kol_search: KolSearch
    enricher: KolEnricher
    embedder: Embedder


def mock_integrations(kol_fixtures: list[dict] | None = None) -> Integrations:
    """Build a fully mocked integration bundle."""
    return Integrations(
        web_reader=MockWebReader(),
        facebook=MockFacebookScraper(),
        llm=MockBrandAnalyzer(),
        kol_search=MockKolSearch(kol_fixtures),
        enricher=MockKolEnricher(kol_fixtures),
        embedder=MockEmbedder(),
    )


__all__ = [
    "BrandAnalyzer",
    "Embedder",
    "FacebookScraper",
    "Integrations",
    "KolEnricher",
    "KolSearch",
    "WebReader",
    "mock_integrations",
]
