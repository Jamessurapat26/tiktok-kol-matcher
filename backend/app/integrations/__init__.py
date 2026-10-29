import os
from dataclasses import dataclass

from .apify import (
    ApifyBudget,
    ApifyBudgetExceeded,
    ApifyConfigError,
    ApifyLimits,
    DiscoveryLimits,
)
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
    live_kols: bool = False,
    live_llm: bool = False,
    live_embed: bool = False,
    live_fb: bool = False,
    kol_fixtures: list[dict] | None = None,
    max_results_per_query: int | None = None,
    apify_token: str | None = None,
    llm_api_key: str | None = None,
    apify_limits: ApifyLimits | None = None,
) -> Integrations:
    """Build the integration bundle.

    Web reading is live only when ``live_web`` is true. TikTok search and
    enrichment are live only when ``live_kols`` is true (requires
    ``APIFY_TOKEN``). Brand analysis, query generation, and reranking are
    live only when ``live_llm`` is true (requires ``LLM_API_KEY``,
    ``NVIDIA_API_KEY``, or ``OPENAI_API_KEY``). Embeddings are live only when
    ``live_embed`` is true. That loads ``google/embeddinggemma-2`` through
    sentence-transformers on the first embed call (install the ``embed`` extra).
    The Facebook page is read live through Apify only when ``live_fb`` is true
    (requires ``APIFY_TOKEN``). Its fallback is always the live web reader,
    even without ``live_web``, because the mock website fixture is not a
    Facebook page. The default never touches the network and does not
    download a model.

    When any live Apify flag is on, one ``ApifyBudget`` is shared by Facebook,
    TikTok search, and enrichment. ``apify_limits`` defaults to
    ``ApifyLimits.from_env()``; the API builds this bundle per job, so the
    budget lasts one job, and the CLI gets one per run.

    Search uses ``TIKTOK_RESULTS_PER_QUERY`` unless ``max_results_per_query``
    overrides it. Enrichment always uses ``TIKTOK_PROFILE_VIDEOS``, and search
    keeps ``APIFY_ENRICH_RESERVE_USD`` free for it.
    """
    if max_results_per_query is not None and max_results_per_query < 1:
        raise ValueError("max_results_per_query must be at least 1")

    limits = apify_limits
    if limits is None and (live_kols or live_fb):
        limits = ApifyLimits.from_env()
    budget: ApifyBudget | None = limits.budget() if limits is not None else None

    if live_kols:
        token = _apify_token(apify_token, "live KOL search")
        kol_search, enricher = _live_kol_services(
            token,
            max_results_per_query,
            budget,
            limits,
        )
    else:
        kol_search = MockKolSearch(kol_fixtures)
        enricher = MockKolEnricher(kol_fixtures)

    if live_fb:
        facebook = _live_facebook(
            _apify_token(apify_token, "the live Facebook reader"),
            budget,
            limits,
        )
    else:
        facebook = MockFacebookScraper()

    return Integrations(
        web_reader=_live_web_reader() if live_web else MockWebReader(),
        facebook=facebook,
        llm=_live_llm(llm_api_key) if live_llm else MockBrandAnalyzer(),
        kol_search=kol_search,
        enricher=enricher,
        embedder=_live_embedder() if live_embed else MockEmbedder(),
    )


def mock_integrations(kol_fixtures: list[dict] | None = None) -> Integrations:
    """Build a fully mocked integration bundle."""
    return build_integrations(kol_fixtures=kol_fixtures)


def real_integrations(kol_fixtures: list[dict] | None = None) -> Integrations:
    """Live web reader; every other service stays mocked."""
    return build_integrations(live_web=True, kol_fixtures=kol_fixtures)


def _apify_token(apify_token: str | None, purpose: str) -> str:
    token = os.environ.get("APIFY_TOKEN") if apify_token is None else apify_token
    token = (token or "").strip()
    if not token:
        raise ApifyConfigError(f"APIFY_TOKEN is not set. Export it before using {purpose}.")
    return token


def _live_facebook(
    token: str,
    budget: ApifyBudget | None,
    limits: ApifyLimits | None,
) -> FacebookScraper:
    from .facebook_live import ApifyFacebookScraper

    if limits is None:
        limits = ApifyLimits.from_env()
    return ApifyFacebookScraper(
        token=token,
        fallback=_live_web_reader(),
        max_charge_usd=limits.facebook_max_charge_usd,
        budget=budget,
    )


def _live_kol_services(
    token: str,
    max_results: int | None,
    budget: ApifyBudget | None,
    limits: ApifyLimits | None,
) -> tuple[KolSearch, KolEnricher]:
    from .kol_enricher_live import ApifyKolEnricher
    from .kol_search_live import ApifyKolSearch

    if limits is None:
        limits = ApifyLimits.from_env()
    if max_results is None:
        max_results = limits.tiktok_results_per_query
    return (
        ApifyKolSearch(
            token=token,
            max_results=max_results,
            max_charge_usd=limits.tiktok_search_max_charge_usd,
            budget=budget,
            reserve_usd=limits.enrich_reserve_usd,
        ),
        ApifyKolEnricher(
            token=token,
            max_videos=limits.tiktok_profile_videos,
            max_charge_usd=limits.tiktok_profile_max_charge_usd,
            budget=budget,
        ),
    )


def _live_embedder() -> Embedder:
    from .embeddings_live import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder.from_env()


def _live_llm(api_key: str | None) -> BrandAnalyzer:
    from .llm_live import LiveBrandAnalyzer

    return LiveBrandAnalyzer(api_key=api_key)


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
    "ApifyBudget",
    "ApifyBudgetExceeded",
    "ApifyConfigError",
    "ApifyLimits",
    "BrandAnalyzer",
    "DiscoveryLimits",
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
