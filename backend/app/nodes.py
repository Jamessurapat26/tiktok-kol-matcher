from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.graph import END
from pydantic import ValidationError

from .brand_match import (
    BRAND_UNMENTIONED_RELEVANCE_CAP,
    aliases_from_urls,
    brand_aliases,
    mentions_brand,
    normalize_for_match,
)
from .integrations import Integrations
from .integrations.apify import ApifyBudgetExceeded
from .integrations.embeddings import (
    MockEmbedder,
    brand_query_text,
    kol_document_text,
    kol_title,
)
from .integrations.llm import MockBrandAnalyzer
from .integrations.web_reader import truncate
from .models import BrandProfile, KolProfile, RankedKol, normalize_handle
from .state import MatchState
from .thai import is_thai_creator

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

    Per-video stats are kept so scoring can use search data. Fixture-only
    fields (query_group) are deliberately dropped here; enrichment re-reads the
    fixture through its own integration. Actor error items (``errorCode``) and
    mapper rejects (``error``) are raised so the discover loop skips and records
    that row.
    """
    error_code = raw.get("errorCode")
    error_detail = raw.get("error")
    if error_code or error_detail:
        code = str(error_code or "error")
        detail = str(error_detail or code)
        raise ValueError(f"{code}: {detail}")
    return KolProfile(
        handle=str(raw["handle"]),
        url=str(raw["url"]),
        nickname=raw.get("nickname"),
        followers=int(raw.get("followers", 0)),
        bio=raw.get("bio"),
        language=raw.get("text_language"),
        region=raw.get("location_created"),
        content_summary=raw.get("content_summary"),
        videos=raw.get("videos") or [],
        stats_source=raw.get("stats_source") or "search",
        source=raw["source"],
    )


def _thai_creator_count(kols: list[KolProfile]) -> int:
    """Count unique discovered creators that pass the Thai filter."""
    return len(
        {normalize_handle(kol.handle) for kol in kols if is_thai_creator(kol)}
    )


def _with_url_aliases(
    profile: BrandProfile,
    website_url: str,
    facebook_url: str,
) -> BrandProfile:
    """Append URL-derived brand aliases, keeping the model's names first."""
    names = list(profile.brand_names)
    seen = {normalize_for_match(name) for name in names}
    for alias in aliases_from_urls(website_url, facebook_url):
        key = normalize_for_match(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(alias)
    return profile.model_copy(update={"brand_names": names})


def make_nodes(
    integrations: Integrations,
    *,
    min_candidates: int,
    max_search_rounds: int,
    prefilter_n: int = 25,
    enrich_limit: int | None = None,
) -> dict[str, Node]:
    """Build the graph nodes with integrations and tuning closed over."""

    def _similarities(brand, kols: list[KolProfile]) -> list[float]:
        return integrations.embedder.similarity(
            brand_query_text(brand),
            [kol_document_text(kol) for kol in kols],
            titles=[kol_title(kol) for kol in kols],
        )

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
        profile = _with_url_aliases(
            profile,
            str(state.request.website_url),
            str(state.request.facebook_url),
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
        """Loop for more queries until enough Thai KOLs or the round cap is hit.

        The target counts unique discovered creators that already pass the Thai
        filter, so a round full of non-Thai accounts does not stop discovery.
        ``stage == "failed"`` (search raised or nothing valid was found, and
        no KOLs were discovered yet) stops the graph instead of retrying.
        """
        if state.stage == "failed":
            return END
        if (
            _thai_creator_count(state.discovered) < min_candidates
            and state.search_rounds < max_search_rounds
        ):
            return "generate_queries"
        return "filter_thai"

    @_guarded("filter_thai")
    def filter_thai(state: MatchState) -> dict[str, Any]:
        """Keep creators whose language, region, or profile text is Thai."""
        candidates = [kol for kol in state.discovered if is_thai_creator(kol)]
        if not candidates:
            raise ValueError("no Thai KOLs found")
        return {"candidates": candidates, "stage": "filtering_thai"}

    @_guarded("enrich_kols")
    def enrich_kols(state: MatchState) -> dict[str, Any]:
        enriched: list[KolProfile] = []
        errors: list[str] = []
        # Live enrichment is one paid actor run per profile, so only the
        # largest accounts (by followers, then handle) are enriched when capped.
        to_enrich = {
            normalize_handle(kol.handle)
            for kol in sorted(
                state.candidates,
                key=lambda k: (-k.followers, normalize_handle(k.handle)),
            )[:enrich_limit]
        }
        budget_reached = False
        skipped = 0
        for kol in state.candidates:
            if normalize_handle(kol.handle) not in to_enrich:
                enriched.append(kol)
                continue
            if budget_reached:
                enriched.append(kol)
                skipped += 1
                continue
            try:
                # model_validate re-checks enrichment. model_dump() leaves url
                # as an HttpUrl; model_validate accepts that.
                enriched.append(
                    KolProfile.model_validate(
                        {**kol.model_dump(), **integrations.enricher.enrich(kol)}
                    )
                )
            except ApifyBudgetExceeded:
                # Keep this profile and the rest un-enriched instead of failing.
                enriched.append(kol)
                budget_reached = True
                skipped += 1
            except ValidationError as exc:
                enriched.append(kol)
                errors.append(
                    f"enrich_kols: rejected enrichment for {kol.handle}: {exc}"
                )
            except Exception as exc:  # noqa: BLE001 - one profile must not drop the rest
                enriched.append(kol)
                errors.append(f"enrich_kols: failed for {kol.handle}: {exc}")
        if skipped:
            noun = "profile" if skipped == 1 else "profiles"
            errors.append(
                "enrich_kols: Apify job budget reached, skipped enrichment "
                f"for {skipped} {noun}"
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
        if prefilter_n < 1:
            raise ValueError("prefilter_n must be at least 1")
        scores = _similarities(state.brand_profile, state.candidates)
        ranked_pairs = sorted(
            zip(state.candidates, scores, strict=True),
            key=lambda pair: (-pair[1], normalize_handle(pair[0].handle)),
        )
        # A top_k above the configured width used to return only the width.
        limit = max(prefilter_n, state.request.top_k)
        kept = ranked_pairs[:limit]
        return {
            "shortlisted": [kol for kol, _score in kept],
            "prefilter_scores": [score for _kol, score in kept],
            "stage": "embedding_prefilter",
        }

    @_guarded("llm_rerank")
    def llm_rerank(state: MatchState) -> dict[str, Any]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before reranking")
        if state.prefilter_scores and len(state.prefilter_scores) != len(state.shortlisted):
            raise ValueError("prefilter_scores must have one score per shortlisted creator")
        # MockEmbedder scores are hashed bag-of-words noise. The mock reranker
        # needs them as its deterministic relevance; a live model must not be
        # told they are evidence.
        mock_scores_for_live_model = isinstance(
            integrations.embedder, MockEmbedder
        ) and not isinstance(integrations.llm, MockBrandAnalyzer)
        similarities: list[float] | None
        if mock_scores_for_live_model:
            similarities = None
        elif len(state.prefilter_scores) == len(state.shortlisted):
            similarities = list(state.prefilter_scores)
        else:
            similarities = _similarities(state.brand_profile, state.shortlisted)
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
                similarities=similarities,
            )
        ]
        aliases = brand_aliases(
            state.brand_profile,
            str(state.request.website_url),
            str(state.request.facebook_url),
        )
        if aliases:
            capped: list[RankedKol] = []
            for item in ranked:
                if (
                    not mentions_brand(item.kol, aliases)
                    and item.scores.relevance > BRAND_UNMENTIONED_RELEVANCE_CAP
                ):
                    scores = item.scores.model_copy(
                        update={"relevance": BRAND_UNMENTIONED_RELEVANCE_CAP}
                    )
                    item = item.model_copy(
                        update={
                            "scores": scores,
                            "total_score": state.request.weights.apply(scores),
                            "reason": (
                                f"{item.reason} Brand not named in sampled posts."
                            ),
                        }
                    )
                capped.append(item)
            ranked = capped
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
