from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.graph import END
from pydantic import ValidationError

from .integrations import Integrations
from .integrations.web_reader import truncate
from .models import BrandProfile, KolProfile, RankedKol, normalize_handle
from .state import MatchState

Node = Callable[[MatchState], dict[str, Any]]


def _guarded(name: str, *, stage_on_error: bool = True) -> Callable[[Node], Node]:
    """Turn any node exception into an ``errors`` update instead of a crash.

    With ``stage_on_error`` (the default) the exception is fatal: ``stage``
    becomes ``"failed"`` and fail-fast routers stop the graph immediately.
    Scrapers pass ``stage_on_error=False`` because they share a parallel
    superstep and must not write ``stage``; they only record errors.
    """

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
        content = truncate(integrations.web_reader.read(str(state.request.website_url)))
        # No stage update: scrape_facebook runs in the same superstep.
        return {"website_content": content}

    @_guarded("scrape_facebook", stage_on_error=False)
    def scrape_facebook(state: MatchState) -> dict[str, Any]:
        page = integrations.facebook.fetch_page(str(state.request.facebook_url))
        # No stage update: scrape_website runs in the same superstep.
        return {"facebook_page": page}

    @_guarded("analyze_brand")
    def analyze_brand(state: MatchState) -> dict[str, Any]:
        # The website is the primary source. Missing it is fatal so the
        # fail-fast router ends the graph. A missing Facebook page is a warning.
        if not state.website_content:
            raise ValueError("missing website_content")

        profile = integrations.llm.analyze_brand(
            state.website_content,
            state.facebook_page,
        )
        update: dict[str, Any] = {
            "brand_profile": profile,
            "stage": "analyzing_brand",
        }
        if state.facebook_page is None:
            update["errors"] = ["analyze_brand: missing facebook_page"]
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
            "last_queries": new_queries,
            "stage": "generating_queries",
        }

    @_guarded("discover_kols")
    def discover_kols(state: MatchState) -> dict[str, Any]:
        next_round = state.search_rounds + 1
        errors: list[str] = []
        kols: list[KolProfile] = []
        try:
            raw_kols = integrations.kol_search.search(
                state.last_queries,
                round=next_round,
            )
        except Exception as exc:  # noqa: BLE001 - search failure is a state update
            errors.append(f"discover_kols: search failed: {exc}")
            raw_kols = None

        if raw_kols is not None:
            for index, raw in enumerate(raw_kols):
                try:
                    kols.append(_kol_from_fixture(raw))
                except Exception as exc:  # noqa: BLE001 - one bad row must not drop the rest
                    errors.append(f"discover_kols: skipped row {index}: {exc}")

        # Nothing usable yet (search raised, or every row was invalid) is fatal.
        # A later-round failure after round 1 already found KOLs is only a warning.
        if not state.discovered and not kols:
            stage = "failed"
        else:
            stage = "discovering_kols"

        update: dict[str, Any] = {
            "search_rounds": next_round,
            "stage": stage,
        }
        if kols:
            update["discovered"] = kols
        if errors:
            update["errors"] = errors
        return update

    def route_after_discover(state: MatchState) -> str:
        """Loop for more queries until enough KOLs or the round cap is hit.

        ``stage == "failed"`` (search raised or nothing valid was found, and
        no KOLs were discovered yet) stops the graph instead of retrying.
        """
        if state.stage == "failed":
            return END
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
        if not candidates:
            raise ValueError("no Thai KOLs found")
        return {"candidates": candidates, "stage": "filtering_thai"}

    @_guarded("enrich_kols")
    def enrich_kols(state: MatchState) -> dict[str, Any]:
        enriched: list[KolProfile] = []
        errors: list[str] = []
        for kol in state.candidates:
            try:
                # model_validate re-checks enrichment. model_dump() leaves url
                # as an HttpUrl; model_validate accepts that.
                enriched.append(
                    KolProfile.model_validate(
                        {**kol.model_dump(), **integrations.enricher.enrich(kol)}
                    )
                )
            except ValidationError as exc:
                enriched.append(kol)
                errors.append(
                    f"enrich_kols: rejected enrichment for {kol.handle}: {exc}"
                )
        update: dict[str, Any] = {
            "candidates": enriched,
            "stage": "enriching_kols",
        }
        if errors:
            update["errors"] = errors
        return update

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
            key=lambda pair: (-pair[1], normalize_handle(pair[0].handle)),
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
            key=lambda item: (
                -item.total_score,
                -(item.kol.median_engagement_rate or 0.0),
                -item.kol.followers,
                normalize_handle(item.kol.handle),
            )
        )
        return {"ranked": ranked, "stage": "reranking"}

    @_guarded("rank_and_output")
    def rank_and_output(state: MatchState) -> dict[str, Any]:
        # Fail-fast routing should not reach this node after a failure. Never
        # rewrite a failed run as done if it does.
        update: dict[str, Any] = {
            "results": state.ranked[: state.request.top_k],
        }
        if state.stage != "failed":
            update["stage"] = "done"
        return update

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
