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


def build_integrations(
    *,
    live_web: bool = False,
    kol_fixtures: list[dict] | None = None,
) -> Integrations:
    """Build the integration bundle.

    Web reading is live only when ``live_web`` is true. Every other service
    stays mocked, and the default (``live_web=False``) never touches the network.
    """
    return Integrations(
        web_reader=_live_web_reader() if live_web else MockWebReader(),
        facebook=MockFacebookScraper(),
        llm=MockBrandAnalyzer(),
        kol_search=MockKolSearch(kol_fixtures),
        enricher=MockKolEnricher(kol_fixtures),
        embedder=MockEmbedder(),
    )


def mock_integrations(kol_fixtures: list[dict] | None = None) -> Integrations:
    """Build a fully mocked integration bundle."""
    return build_integrations(kol_fixtures=kol_fixtures)


def real_integrations(kol_fixtures: list[dict] | None = None) -> Integrations:
    """Live web reader; every other service stays mocked."""
    return build_integrations(live_web=True, kol_fixtures=kol_fixtures)


def _live_web_reader() -> WebReader:
    from .web_reader_live import (
        CachedWebReader,
        FallbackWebReader,
        JinaWebReader,
        TrafilaturaWebReader,
        default_cache_dir,
    )

    return CachedWebReader(
        FallbackWebReader(JinaWebReader(), TrafilaturaWebReader()),
        default_cache_dir(),
    )


__all__ = [
    "BrandAnalyzer",
    "Embedder",
    "FacebookScraper",
    "Integrations",
    "KolEnricher",
    "KolSearch",
    "WebReader",
    "build_integrations",
    "mock_integrations",
    "real_integrations",
]
