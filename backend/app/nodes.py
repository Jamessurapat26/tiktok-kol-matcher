from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from langgraph.graph import END
from pydantic import ValidationError

from .brand_match import (
    MIN_ALIAS_CHARS,
    aliases_from_urls,
    brand_aliases,
    normalize_for_match,
    normalized_aliases,
)
from .filters import SELLER_ACCOUNT, is_seller_account
from .filters import hard_filter as apply_hard_filter
from .integrations import Integrations
from .integrations.apify import (
    ApifyBudgetExceeded,
    ApifyLimits,
    estimate_run_cost,
)
from .integrations.web_reader import truncate
from .matching_config import DEFAULT_CONFIG, MatchingConfig
from .models import BrandProfile, KolProfile, normalize_handle
from .ranking import UNREVIEWED_FLAG, diversify, rank_candidates, similarities
from .scoring import score_pool
from .state import MatchState
from .thai import is_thai_creator

Node = Callable[[MatchState], dict[str, Any]]

_SEARCH_ACTOR = "clockworks/tiktok-scraper"
_PROFILE_ACTOR = "clockworks/tiktok-profile-scraper"


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
        avatar_url=raw.get("avatar_url"),
        followers=int(raw.get("followers", 0)),
        bio=raw.get("bio"),
        language=raw.get("text_language"),
        region=raw.get("location_created"),
        content_summary=raw.get("content_summary"),
        videos=raw.get("videos") or [],
        stats_source=raw.get("stats_source") or "search",
        source=raw["source"],
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
    cfg: MatchingConfig = DEFAULT_CONFIG,
) -> dict[str, Node]:
    """Build the graph nodes with integrations and tuning closed over."""

    def _alias_sets(state: MatchState) -> tuple[list[str], list[str]]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required")
        brand = brand_aliases(
            state.brand_profile,
            str(state.request.website_url),
            str(state.request.facebook_url),
        )
        competitors = normalized_aliases(state.brand_profile.competitors)
        return brand, competitors

    def _search_target_count(state: MatchState) -> int:
        """Count unique discovered Thai creators, leaving out plain sellers."""
        brand_alias_list, competitor_alias_list = (
            _alias_sets(state) if state.brand_profile is not None else ([], [])
        )
        return len(
            {
                normalize_handle(kol.handle)
                for kol in state.discovered
                if is_thai_creator(kol)
                and (
                    state.brand_profile is None
                    or not is_seller_account(
                        kol,
                        state.brand_profile,
                        brand_alias_list,
                        competitor_alias_list,
                    )
                )
            }
        )

    def _scrape_reason(errors: list[str], node: str) -> str:
        prefix = f"{node}: "
        for error in reversed(errors):
            if error.startswith(prefix):
                return error[len(prefix):] or "missing"
        return "missing"

    @_guarded("scrape_website", stage_on_error=False)
    def scrape_website(state: MatchState) -> dict[str, Any]:
        content = truncate(integrations.web_reader.read(str(state.request.website_url)))
        # No stage update: scrape_facebook runs in the same superstep.
        return {"website_content": content}

    @_guarded("scrape_facebook", stage_on_error=False)
    def scrape_facebook(state: MatchState) -> dict[str, Any]:
        page = integrations.facebook.fetch_page(str(state.request.facebook_url))
        # No stage update: scrape_website runs in the same superstep. A page
        # from the web fallback keeps Apify's error as a warning.
        update: dict[str, Any] = {"facebook_page": page}
        if page.source_warning:
            update["errors"] = [f"scrape_facebook: warning: {page.source_warning}"]
        return update

    @_guarded("analyze_brand")
    def analyze_brand(state: MatchState) -> dict[str, Any]:
        # Either source is enough. Only a run with neither is fatal.
        if not state.website_content and state.facebook_page is None:
            raise ValueError(
                "no usable brand source "
                f"(website: {_scrape_reason(state.errors, 'scrape_website')}, "
                f"facebook: {_scrape_reason(state.errors, 'scrape_facebook')})"
            )

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
            "brand_sources": [
                "website: used" if state.website_content else "website: skipped",
                "facebook: used" if state.facebook_page is not None else "facebook: skipped",
            ],
            "stage": "analyzing_brand",
            "cost_log": ["llm: analyze_brand 1 call"],
        }
        warnings: list[str] = []
        if not state.website_content:
            warnings.append("analyze_brand: missing website_content")
        if state.facebook_page is None:
            warnings.append("analyze_brand: missing facebook_page")
        if warnings:
            update["errors"] = warnings
        return update

    @_guarded("generate_queries")
    def generate_queries(state: MatchState) -> dict[str, Any]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before generating queries")
        round_no = state.search_rounds + 1
        new_queries = integrations.llm.generate_queries(
            state.brand_profile,
            state.search_queries,
            round_no,
        )
        # Recall guard: a brand name in a query returns the brand's own account
        # and shops, not the creators the brief is looking for.
        blocked = [
            alias
            for alias in brand_aliases(
                state.brand_profile,
                str(state.request.website_url),
                str(state.request.facebook_url),
            )
            if len(alias) >= MIN_ALIAS_CHARS
        ]
        kept_queries: list[str] = []
        dropped = 0
        for query in new_queries:
            normalized_query = normalize_for_match(query)
            if any(alias in normalized_query for alias in blocked):
                dropped += 1
                continue
            kept_queries.append(query)
        remaining = cfg.max_search_queries - len(state.search_queries)
        kept_queries = kept_queries[:max(remaining, 0)]
        update: dict[str, Any] = {
            "search_queries": kept_queries,
            "last_queries": kept_queries,
            "stage": "generating_queries",
            "cost_log": [
                f"llm: generate_queries round {round_no} 1 call"
            ],
        }
        if dropped:
            update["errors"] = [
                f"generate_queries: dropped {dropped} brand-name queries"
            ]
        return update

    @_guarded("discover_kols")
    def discover_kols(state: MatchState) -> dict[str, Any]:
        next_round = state.search_rounds + 1
        queries = list(state.last_queries)
        cost_log: list[str] = []
        if queries:
            limits = ApifyLimits.from_env()
            per_run = estimate_run_cost(
                _SEARCH_ACTOR,
                limits.tiktok_results_per_query,
                limits.tiktok_search_max_charge_usd,
            )
            cost_log.append(
                f"search round {next_round}: est "
                f"${per_run * len(queries):.2f} ({len(queries)} queries)"
            )

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
        if cost_log:
            update["cost_log"] = cost_log
        return update

    def route_after_discover(state: MatchState) -> str:
        """Loop for more queries until enough Thai KOLs or the round cap is hit.

        The target counts unique discovered creators that already pass the Thai
        filter and are not plain sellers, so a round full of non-Thai accounts
        or dealers does not stop discovery.
        ``stage == "failed"`` (search raised or nothing valid was found, and
        no KOLs were discovered yet) stops the graph instead of retrying.
        """
        if state.stage == "failed":
            return END
        if len(state.search_queries) >= cfg.max_search_queries:
            return "hard_filter"
        if (
            _search_target_count(state) < min_candidates
            and state.search_rounds < max_search_rounds
        ):
            return "generate_queries"
        return "hard_filter"

    @_guarded("hard_filter")
    def hard_filter(state: MatchState) -> dict[str, Any]:
        """Keep creators that pass every hard rule and count each rejection.

        Activity is checked after enrichment instead: search hits are not a
        creator's newest videos, so one old viral hit must not reject them.
        """
        candidates, rejected = _filter_candidates(
            state, state.discovered, {}, check_activity=False
        )
        return {
            "candidates": candidates,
            "rejected": rejected,
            "stage": "filtering",
        }

    def _filter_candidates(
        state: MatchState,
        kols: list[KolProfile],
        rejected: dict[str, int],
        *,
        check_activity: bool,
    ) -> tuple[list[KolProfile], dict[str, int]]:
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before hard filtering")
        now = datetime.now(UTC)
        brand_alias_list, competitor_alias_list = _alias_sets(state)
        kept: list[KolProfile] = []
        counts = dict(rejected)
        for kol in kols:
            reason = apply_hard_filter(
                kol,
                state.brand_profile,
                brand_alias_list,
                competitor_alias_list,
                cfg,
                now,
                check_activity=check_activity,
            )
            if reason is None:
                kept.append(kol)
            else:
                counts[reason] = counts.get(reason, 0) + 1
        # Finding only sellers is a real answer, not a failed Search.
        if not kept and not counts.get(SELLER_ACCOUNT):
            raise ValueError("no candidates passed the hard filters")
        return kept, counts

    @_guarded("first_pass_rank")
    def first_pass_rank(state: MatchState) -> dict[str, Any]:
        """Rank every hard-filter survivor using data already in hand."""
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before first-pass ranking")
        brand_alias_list, competitor_alias_list = _alias_sets(state)
        sims = similarities(
            integrations.embedder,
            state.brand_profile,
            state.candidates,
        )
        scored = score_pool(
            state.candidates,
            state.brand_profile,
            sims,
            None,
            brand_alias_list,
            competitor_alias_list,
            cfg,
            datetime.now(UTC),
        )
        scored.sort(
            key=lambda item: (-item.final, normalize_handle(item.kol.handle))
        )
        return {
            "candidates": [item.kol for item in scored],
            "stage": "first_pass",
        }

    @_guarded("enrich_kols")
    def enrich_kols(state: MatchState) -> dict[str, Any]:
        enriched: list[KolProfile] = []
        errors: list[str] = []
        # Enrich the first-pass leaders, not the largest accounts.
        to_enrich = {
            normalize_handle(kol.handle)
            for kol in state.candidates[: cfg.enrich_top_n]
        }
        budget_reached = False
        skipped = 0
        attempted = 0
        for kol in state.candidates:
            if normalize_handle(kol.handle) not in to_enrich:
                enriched.append(kol)
                continue
            if budget_reached:
                enriched.append(kol)
                skipped += 1
                continue
            attempted += 1
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
                # The guard refused the run, so it is not billed.
                attempted -= 1
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
        # Re-apply every hard rule, activity included, now that enriched
        # profiles carry their real follower counts and newest videos.
        rejected = state.rejected
        if state.brand_profile is not None:
            enriched, rejected = _filter_candidates(
                state, enriched, rejected, check_activity=True
            )
        if skipped:
            noun = "profile" if skipped == 1 else "profiles"
            errors.append(
                "enrich_kols: Apify job budget reached, skipped enrichment "
                f"for {skipped} {noun}"
            )
        update: dict[str, Any] = {
            "candidates": enriched,
            "rejected": rejected,
            "stage": "enriching_kols",
        }
        if attempted:
            limits = ApifyLimits.from_env()
            per_profile = estimate_run_cost(
                _PROFILE_ACTOR,
                limits.tiktok_profile_videos,
                limits.tiktok_profile_max_charge_usd,
            )
            update["cost_log"] = [
                f"enrich: est ${per_profile * attempted:.2f} "
                f"({attempted} profiles)"
            ]
        if errors:
            update["errors"] = errors
        return update

    @_guarded("judge_and_rank")
    def judge_and_rank(state: MatchState) -> dict[str, Any]:
        """Judge the first-pass leaders once, then score every candidate."""
        if state.brand_profile is None:
            raise ValueError("brand_profile is required before judging")
        brand_alias_list, competitor_alias_list = _alias_sets(state)
        outcome = rank_candidates(
            state.brand_profile,
            state.candidates,
            embedder=integrations.embedder,
            judge_fn=integrations.llm.judge,
            brand_aliases=brand_alias_list,
            competitor_aliases=competitor_alias_list,
            cfg=cfg,
            now=datetime.now(UTC),
            top_k=state.request.top_k,
        )
        rejected = dict(state.rejected)
        for reason, count in outcome.rejected.items():
            rejected[reason] = rejected.get(reason, 0) + count
        update: dict[str, Any] = {
            "ranked": outcome.ranked,
            "rejected": rejected,
            "stage": "judging",
        }
        if outcome.judged_count:
            update["cost_log"] = [
                f"llm: judge 1 call ({outcome.judged_count} creators)"
            ]
        if outcome.errors:
            update["errors"] = outcome.errors
        return update

    @_guarded("rank_and_output")
    def rank_and_output(state: MatchState) -> dict[str, Any]:
        # Only creators the judge reviewed, or was asked about and failed to
        # answer for, can be recommended. The rest never had a seller check.
        results = diversify(
            [
                item
                for item in state.ranked
                if item.scores.judged or UNREVIEWED_FLAG in item.flags
            ],
            integrations.embedder,
            state.request.top_k,
            cfg,
        )
        update: dict[str, Any] = {"results": results}
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
        "hard_filter": hard_filter,
        "first_pass_rank": first_pass_rank,
        "enrich_kols": enrich_kols,
        "judge_and_rank": judge_and_rank,
        "rank_and_output": rank_and_output,
    }
