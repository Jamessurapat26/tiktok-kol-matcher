from collections.abc import Callable
from functools import wraps
from typing import Any

from .integrations import Integrations
from .models import BrandProfile, KolProfile, RankedKol
from .state import MatchState

Node = Callable[[MatchState], dict[str, Any]]


def _guarded(name: str, *, stage_on_error: bool = True) -> Callable[[Node], Node]:
    """Turn any node exception into an ``errors`` update instead of a crash."""

    def decorator(fn: Node) -> Node:
        @wraps(fn)
        def wrapper(state: MatchState) -> dict[str, Any]:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001 - node boundary must not crash
                update: dict[str, Any] = {"errors": [f"{name}: {exc}"]}
                if stage_on_error:
                    update["stage"] = "failed"
                return update

        return wrapper

    return decorator


def _normalize_handle(handle: str) -> str:
    return handle.strip().lower().lstrip("@")


def _kol_from_fixture(raw: dict) -> KolProfile:
    """Convert an Apify-style raw record into a KolProfile.

    Fixture-only fields (topics, query_group, engagement_rate) are deliberately
    dropped here; enrichment re-reads the fixture through its own integration.
    """
    return KolProfile(
        handle=str(raw["handle"]),
        url=str(raw["url"]),
        nickname=raw.get("nickname"),
        followers=int(raw.get("followers", 0)),
        bio=raw.get("bio"),
        language=raw.get("text_language"),
        region=raw.get("location_created"),
        source=raw["source"],
    )


def _brand_text(brand: BrandProfile) -> str:
    return " ".join(
        [
            brand.category,
            brand.tone,
            brand.content_style,
            *brand.products,
            *brand.keywords_th,
            *brand.keywords_en,
            *brand.hashtags,
            *brand.target_audience.interests,
        ]
    )


def _kol_text(kol: KolProfile) -> str:
    return " ".join(
        part for part in (kol.content_summary, kol.bio, kol.nickname) if part
    )


def make_nodes(
    integrations: Integrations,
    *,
    min_candidates: int,
    max_search_rounds: int,
    prefilter_n: int = 25,
) -> dict[str, Node]:
    """Build the graph nodes with integrations and tuning closed over."""

    @_guarded("scrape_website", stage_on_error=False)
    def scrape_website(state: MatchState) -> dict[str, Any]:
        content = integrations.web_reader.read(str(state.request.website_url))
        # No stage update: scrape_facebook runs in the same superstep.
        return {"website_content": content}

    @_guarded("scrape_facebook", stage_on_error=False)
    def scrape_facebook(state: MatchState) -> dict[str, Any]:
        page = integrations.facebook.fetch_page(str(state.request.facebook_url))
        # No stage update: scrape_website runs in the same superstep.
        return {"facebook_page": page}

    @_guarded("analyze_brand")
    def analyze_brand(state: MatchState) -> dict[str, Any]:
        missing = []
        if not state.website_content:
            missing.append("website_content")
        if state.facebook_page is None:
            missing.append("facebook_page")

        # The mock still returns a usable profile when a source is missing.
        profile = integrations.llm.analyze_brand(
            state.website_content,
            state.facebook_page,
        )
        update: dict[str, Any] = {
            "brand_profile": profile,
            "stage": "analyzing_brand",
        }
        if missing:
            update["errors"] = [
                f"analyze_brand: missing {name}" for name in missing
            ]
        return update

    @_guarded("generate_queries")
    def generate_queries(state: MatchState) -> dict[str, Any]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before generating queries")
        new_queries = integrations.llm.generate_queries(
            state.brand_profile,
            state.search_queries,
            state.search_rounds + 1,
        )
        return {
            "search_queries": new_queries,
            "stage": "generating_queries",
        }

    @_guarded("discover_kols")
    def discover_kols(state: MatchState) -> dict[str, Any]:
        next_round = state.search_rounds + 1
        raw_kols = integrations.kol_search.search(
            state.search_queries,
            round=next_round,
        )
        kols = [_kol_from_fixture(raw) for raw in raw_kols]
        return {
            "discovered": kols,
            "search_rounds": next_round,
            "stage": "discovering_kols",
        }

    def route_after_discover(state: MatchState) -> str:
        """Loop for more queries until enough KOLs or the round cap is hit."""
        if (
            len(state.discovered) < min_candidates
            and state.search_rounds < max_search_rounds
        ):
            return "generate_queries"
        return "filter_thai"

    @_guarded("filter_thai")
    def filter_thai(state: MatchState) -> dict[str, Any]:
        """Keep KOLs whose content is Thai or whose account region is Thailand."""
        candidates = [
            kol
            for kol in state.discovered
            if (kol.language or "").casefold() in {"th", "thai"}
            or (kol.region or "").upper() == "TH"
        ]
        return {"candidates": candidates, "stage": "filtering_thai"}

    @_guarded("enrich_kols")
    def enrich_kols(state: MatchState) -> dict[str, Any]:
        enriched = [
            kol.model_copy(update=integrations.enricher.enrich(kol))
            for kol in state.candidates
        ]
        return {"candidates": enriched, "stage": "enriching_kols"}

    @_guarded("embed_and_prefilter")
    def embed_and_prefilter(state: MatchState) -> dict[str, Any]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before prefiltering")
        scores = integrations.embedder.similarity(
            _brand_text(state.brand_profile),
            [_kol_text(kol) for kol in state.candidates],
        )
        ranked_pairs = sorted(
            zip(state.candidates, scores, strict=True),
            key=lambda pair: (-pair[1], _normalize_handle(pair[0].handle)),
        )
        shortlisted = [kol for kol, _ in ranked_pairs[:prefilter_n]]
        return {"shortlisted": shortlisted, "stage": "embedding_prefilter"}

    @_guarded("llm_rerank")
    def llm_rerank(state: MatchState) -> dict[str, Any]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before reranking")
        ranked = [
            RankedKol(
                kol=kol,
                scores=scores,
                total_score=state.request.weights.apply(scores),
                reason=reason,
            )
            for kol, scores, reason in integrations.llm.rerank(
                state.brand_profile,
                state.shortlisted,
            )
        ]
        ranked.sort(
            key=lambda item: (-item.total_score, _normalize_handle(item.kol.handle))
        )
        return {"ranked": ranked, "stage": "reranking"}

    @_guarded("rank_and_output")
    def rank_and_output(state: MatchState) -> dict[str, Any]:
        return {
            "results": state.ranked[: state.request.top_k],
            "stage": "done",
        }

    return {
        "scrape_website": scrape_website,
        "scrape_facebook": scrape_facebook,
        "analyze_brand": analyze_brand,
        "generate_queries": generate_queries,
        "discover_kols": discover_kols,
        "route_after_discover": route_after_discover,
        "filter_thai": filter_thai,
        "enrich_kols": enrich_kols,
        "embed_and_prefilter": embed_and_prefilter,
        "llm_rerank": llm_rerank,
        "rank_and_output": rank_and_output,
    }
