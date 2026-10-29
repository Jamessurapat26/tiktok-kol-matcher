"""Offline tests for the job-level Apify budget and its degradation paths."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

from app.integrations import build_integrations, mock_integrations
from app.integrations.apify import (
    ApifyBudget,
    ApifyBudgetExceeded,
    ApifyClient,
    ApifyConfigError,
    ApifyError,
    ApifyLimits,
    DiscoveryLimits,
    estimate_default_job_usd,
    estimate_run_cost,
)
from app.integrations.kol_enricher_live import ApifyKolEnricher
from app.integrations.kol_search_live import ApifyKolSearch
from app.models import KolProfile, MatchRequest
from app.nodes import make_nodes
from app.state import MatchState

_TOKEN = "super-secret-token"
_SEARCH_ACTOR = "clockworks/tiktok-scraper"
_LIMIT_ENV = (
    "APIFY_JOB_BUDGET_USD",
    "APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD",
    "APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD",
    "APIFY_FACEBOOK_MAX_CHARGE_USD",
)
_APIFY_ENV = (
    *_LIMIT_ENV,
    "TIKTOK_RESULTS_PER_QUERY",
    "TIKTOK_PROFILE_VIDEOS",
    "APIFY_ENRICH_RESERVE_USD",
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _kol(handle: str, followers: int) -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        followers=followers,
        source="apify",
    )


def _clear_limit_env(monkeypatch) -> None:
    for name in _APIFY_ENV:
        monkeypatch.delenv(name, raising=False)


def test_budget_tracks_spend_skips_and_summary():
    budget = ApifyBudget(1.0)

    assert budget.spent == 0.0
    assert budget.remaining == 1.0
    assert not budget.exhausted
    assert budget.skipped == 0

    budget.check(_SEARCH_ACTOR)
    budget.record(_SEARCH_ACTOR, 0.4)
    assert budget.spent == pytest.approx(0.4)
    assert budget.remaining == pytest.approx(0.6)
    assert not budget.exhausted

    budget.record(_SEARCH_ACTOR, 0.7)
    assert budget.spent == pytest.approx(1.1)
    assert budget.remaining == 0.0
    assert budget.exhausted

    with pytest.raises(ApifyBudgetExceeded, match=r"Apify job budget of \$1\.00"):
        budget.check("apify/facebook-pages-scraper")

    assert budget.skipped == 1
    summary = budget.summary()
    assert "$1.10" in summary
    assert "$1.00" in summary
    assert "1 calls skipped" in summary


def test_budget_rejects_a_non_positive_limit():
    with pytest.raises(ValueError, match="limit_usd"):
        ApifyBudget(0)


def test_budget_reserve_is_kept_for_enrichment():
    budget = ApifyBudget(1.0)
    budget.record(_SEARCH_ACTOR, 0.7)

    with pytest.raises(ApifyBudgetExceeded, match="kept for enrichment"):
        budget.check(_SEARCH_ACTOR, reserve_usd=0.3)

    # Without the reserve, later stages may still spend what is left.
    budget.check(_SEARCH_ACTOR)


def test_estimate_run_cost_uses_the_actor_rate_and_the_run_cap():
    assert estimate_run_cost(_SEARCH_ACTOR, 2, 0.50) == pytest.approx(0.01)
    assert estimate_run_cost("unknown/actor", 0, 0.50) == pytest.approx(0.015)
    assert estimate_run_cost(_SEARCH_ACTOR, 1_000, 0.50) == 0.50


def test_budget_records_are_thread_safe():
    budget = ApifyBudget(100.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: budget.record(_SEARCH_ACTOR, 0.001), range(1_000)))

    assert budget.spent == pytest.approx(1.0)


def test_client_records_an_estimate_then_refuses_without_http():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(201, json=[{"id": 1}, {"id": 2}])

    budget = ApifyBudget(0.01)
    client = ApifyClient(
        _TOKEN,
        client=_client(handler),
        budget=budget,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
    )

    items = client.run_sync_get_items(_SEARCH_ACTOR, {}, max_items=2)

    assert len(items) == 2
    assert budget.spent == pytest.approx(0.01)
    assert budget.exhausted
    with pytest.raises(ApifyBudgetExceeded):
        client.run_sync_get_items(_SEARCH_ACTOR, {}, max_items=2)

    assert calls["n"] == 1
    assert budget.skipped == 1


def test_client_timeout_records_the_full_run_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    budget = ApifyBudget(1.0)
    client = ApifyClient(
        _TOKEN,
        client=_client(handler),
        budget=budget,
        max_charge_usd=0.50,
        max_retries=2,
        backoff=0,
    )

    with pytest.raises(ApifyError, match="timed out"):
        client.run_sync_get_items(_SEARCH_ACTOR, {}, max_items=1)

    assert budget.spent == pytest.approx(0.50)


@pytest.mark.parametrize("status", [400, 402, 404, 422])
def test_client_refused_request_records_nothing(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "invalid input"}})

    budget = ApifyBudget(1.0)
    client = ApifyClient(
        _TOKEN,
        client=_client(handler),
        budget=budget,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
    )

    with pytest.raises(ApifyError):
        client.run_sync_get_items(_SEARCH_ACTOR, {}, max_items=1)

    assert budget.spent == 0.0


def test_client_server_error_records_the_full_run_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    budget = ApifyBudget(1.0)
    client = ApifyClient(
        _TOKEN,
        client=_client(handler),
        budget=budget,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
    )

    with pytest.raises(ApifyError):
        client.run_sync_get_items(_SEARCH_ACTOR, {}, max_items=1)

    assert budget.spent == pytest.approx(0.50)


def test_limits_from_env_defaults(monkeypatch):
    _clear_limit_env(monkeypatch)

    limits = ApifyLimits.from_env()

    assert limits.job_budget_usd == pytest.approx(1.00)
    assert limits.tiktok_search_max_charge_usd == pytest.approx(0.50)
    assert limits.tiktok_profile_max_charge_usd == pytest.approx(0.50)
    assert limits.facebook_max_charge_usd == pytest.approx(0.50)
    assert limits.tiktok_results_per_query == 20
    assert limits.tiktok_profile_videos == 10
    assert limits.enrich_reserve_usd == pytest.approx(0.30)
    assert limits.budget().remaining == pytest.approx(1.00)


def test_discovery_limits_from_env_defaults(monkeypatch):
    monkeypatch.delenv("DISCOVERY_MIN_THAI_CANDIDATES", raising=False)
    monkeypatch.delenv("DISCOVERY_MAX_ROUNDS", raising=False)

    limits = DiscoveryLimits.from_env()

    assert limits.min_thai_candidates == 30
    assert limits.max_rounds == 3


def test_limits_from_env_overrides(monkeypatch):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv("APIFY_JOB_BUDGET_USD", "2.5")
    monkeypatch.setenv("APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD", "0.6")
    monkeypatch.setenv("APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD", "0.7")
    monkeypatch.setenv("APIFY_FACEBOOK_MAX_CHARGE_USD", "0.8")
    monkeypatch.setenv("TIKTOK_RESULTS_PER_QUERY", "37")
    monkeypatch.setenv("TIKTOK_PROFILE_VIDEOS", "23")
    monkeypatch.setenv("APIFY_ENRICH_RESERVE_USD", "0.4")

    limits = ApifyLimits.from_env()

    assert limits.job_budget_usd == pytest.approx(2.5)
    assert limits.tiktok_search_max_charge_usd == pytest.approx(0.6)
    assert limits.tiktok_profile_max_charge_usd == pytest.approx(0.7)
    assert limits.facebook_max_charge_usd == pytest.approx(0.8)
    assert limits.tiktok_results_per_query == 37
    assert limits.tiktok_profile_videos == 23
    assert limits.enrich_reserve_usd == pytest.approx(0.4)


def test_discovery_limits_from_env_overrides(monkeypatch):
    monkeypatch.setenv("DISCOVERY_MIN_THAI_CANDIDATES", "12")
    monkeypatch.setenv("DISCOVERY_MAX_ROUNDS", "2")

    limits = DiscoveryLimits.from_env()

    assert limits.min_thai_candidates == 12
    assert limits.max_rounds == 2


@pytest.mark.parametrize("name", _LIMIT_ENV)
@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_limits_from_env_rejects_invalid_values(monkeypatch, name, value):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ApifyConfigError, match=name):
        ApifyLimits.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TIKTOK_RESULTS_PER_QUERY", "0"),
        ("TIKTOK_RESULTS_PER_QUERY", "51"),
        ("TIKTOK_RESULTS_PER_QUERY", "not-a-number"),
        ("TIKTOK_PROFILE_VIDEOS", "0"),
        ("TIKTOK_PROFILE_VIDEOS", "31"),
        ("TIKTOK_PROFILE_VIDEOS", "not-a-number"),
        ("APIFY_ENRICH_RESERVE_USD", "-1"),
        ("APIFY_ENRICH_RESERVE_USD", "not-a-number"),
    ],
)
def test_limits_from_env_rejects_invalid_sizes_or_reserve(
    monkeypatch, name, value
):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ApifyConfigError, match=name):
        ApifyLimits.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DISCOVERY_MIN_THAI_CANDIDATES", "0"),
        ("DISCOVERY_MIN_THAI_CANDIDATES", "not-a-number"),
        ("DISCOVERY_MAX_ROUNDS", "0"),
        ("DISCOVERY_MAX_ROUNDS", "6"),
        ("DISCOVERY_MAX_ROUNDS", "not-a-number"),
    ],
)
def test_discovery_limits_from_env_rejects_invalid_values(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ApifyConfigError, match=name):
        DiscoveryLimits.from_env()


def test_limits_from_env_rejects_a_reserve_that_is_not_below_the_budget(
    monkeypatch,
):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv("APIFY_JOB_BUDGET_USD", "0.25")
    monkeypatch.setenv("APIFY_ENRICH_RESERVE_USD", "0.25")

    with pytest.raises(ApifyConfigError, match="APIFY_ENRICH_RESERVE_USD"):
        ApifyLimits.from_env()


@pytest.mark.parametrize(
    "name",
    [
        "APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD",
        "APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD",
    ],
)
def test_limits_from_env_rejects_a_tiktok_cap_below_the_actor_minimum(
    monkeypatch, name
):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv(name, "0.49")

    with pytest.raises(ApifyConfigError, match="must be at least 0.50"):
        ApifyLimits.from_env()


def test_estimate_default_job_stays_within_budget_and_allows_eight_search_runs(
    monkeypatch,
):
    _clear_limit_env(monkeypatch)
    limits = ApifyLimits.from_env()

    estimate = estimate_default_job_usd(limits)
    search_run = estimate_run_cost(
        _SEARCH_ACTOR,
        limits.tiktok_results_per_query,
        limits.tiktok_search_max_charge_usd,
    )

    assert estimate <= limits.job_budget_usd + search_run
    empty = estimate_default_job_usd(limits, queries=0)
    assert estimate == pytest.approx(empty + 8 * search_run)

    many = estimate_default_job_usd(limits, queries=50)
    allowed_runs = round((many - empty) / search_run)
    assert allowed_runs >= 8


def test_search_stops_after_the_budget_and_keeps_found_creators():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            201,
            json=[{"authorMeta": {"name": "found_creator", "fans": 1200}}],
        )

    search = ApifyKolSearch(
        token=_TOKEN,
        client=_client(handler),
        max_results=5,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
        budget=ApifyBudget(0.005),
    )

    rows = search.search(["one", "two", "three"], round=1)

    assert calls["n"] == 1
    creators = [row for row in rows if not row.get("errorCode")]
    budget_rows = [
        row for row in rows if row.get("errorCode") == "APIFY_BUDGET_EXCEEDED"
    ]
    assert [row["handle"] for row in creators] == ["found_creator"]
    assert len(budget_rows) == 1
    assert "skipped 2 queries" in budget_rows[0]["error"]


def test_search_reserve_leaves_room_for_enrichment(tmp_path):
    calls = {"search": 0, "profile": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "profile-scraper" in request.url.path:
            calls["profile"] += 1
            return httpx.Response(
                201,
                json=[
                    {
                        "text": "latest post",
                        "diggCount": 20,
                        "commentCount": 0,
                        "shareCount": 0,
                        "playCount": 200,
                    }
                ],
            )
        calls["search"] += 1
        return httpx.Response(
            201,
            json=[{"authorMeta": {"name": "found_creator", "fans": 1200}}],
        )

    budget = ApifyBudget(0.305)
    search = ApifyKolSearch(
        token=_TOKEN,
        client=_client(handler),
        max_results=1,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
        budget=budget,
        reserve_usd=0.30,
    )

    rows = search.search(["one", "two", "three"], round=1)

    assert calls["search"] == 1
    assert len(
        [row for row in rows if row.get("errorCode") == "APIFY_BUDGET_EXCEEDED"]
    ) == 1

    enricher = ApifyKolEnricher(
        token=_TOKEN,
        client=_client(handler),
        max_videos=1,
        max_charge_usd=0.50,
        max_retries=0,
        backoff=0,
        budget=budget,
        cache_dir=tmp_path,
        clock=lambda: 1_000.0,
    )
    update = enricher.enrich(_kol("found_creator", 1200))

    assert calls["profile"] == 1
    assert update["median_engagement_rate"] == pytest.approx(0.1)
    assert budget.spent == pytest.approx(0.0075 + 0.007)


class _BudgetEnricher:
    def __init__(self) -> None:
        self.calls = 0

    def enrich(self, kol: KolProfile) -> dict:
        self.calls += 1
        raise ApifyBudgetExceeded(
            "Apify job budget of $1.00 reached (estimated spend $1.01)"
        )


def test_enrich_node_stops_at_the_budget_and_keeps_every_candidate():
    enricher = _BudgetEnricher()
    integrations = replace(mock_integrations(), enricher=enricher)
    node = make_nodes(
        integrations,
        min_candidates=1,
        max_search_rounds=1,
        enrich_limit=10,
    )["enrich_kols"]
    state = MatchState(
        request=MatchRequest(
            website_url="https://example-brand.co.th",
            facebook_url="https://www.facebook.com/examplebrand",
        ),
        candidates=[
            _kol("creator_a", 300),
            _kol("creator_b", 200),
            _kol("creator_c", 100),
        ],
    )

    update = node(state)

    assert enricher.calls == 1
    assert [kol.handle for kol in update["candidates"]] == [
        "creator_a",
        "creator_b",
        "creator_c",
    ]
    assert update["stage"] == "enriching_kols"
    assert len(update["errors"]) == 1
    assert "skipped enrichment for 3 profiles" in update["errors"][0]


def test_build_integrations_shares_one_budget_across_live_apify_services(
    monkeypatch, tmp_path
):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("APIFY_JOB_BUDGET_USD", "2.5")
    monkeypatch.setenv("APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD", "0.6")
    monkeypatch.setenv("APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD", "0.7")
    monkeypatch.setenv("APIFY_FACEBOOK_MAX_CHARGE_USD", "0.8")
    monkeypatch.setenv("TIKTOK_RESULTS_PER_QUERY", "7")
    monkeypatch.setenv("TIKTOK_PROFILE_VIDEOS", "6")
    monkeypatch.setenv("APIFY_ENRICH_RESERVE_USD", "0.4")

    bundle = build_integrations(
        live_kols=True,
        live_fb=True,
        apify_token=_TOKEN,
        max_results_per_query=3,
    )
    from_env = build_integrations(
        live_kols=True,
        apify_token=_TOKEN,
    )

    search_budget = bundle.kol_search._apify._budget
    enricher_budget = bundle.enricher._apify._budget
    facebook_budget = bundle.facebook._apify._budget

    assert search_budget is not None
    assert search_budget is enricher_budget
    assert search_budget is facebook_budget
    assert search_budget.remaining == pytest.approx(2.5)
    assert bundle.kol_search._apify._max_charge_usd == pytest.approx(0.6)
    assert bundle.enricher._apify._max_charge_usd == pytest.approx(0.7)
    assert bundle.facebook._apify._max_charge_usd == pytest.approx(0.8)
    # The CLI flag overrides search only; enrichment always uses its env value.
    assert bundle.kol_search.max_results == 3
    assert bundle.enricher.max_videos == 6
    assert bundle.kol_search._apify._reserve_usd == pytest.approx(0.4)
    assert bundle.enricher._apify._reserve_usd == 0.0
    assert bundle.facebook._apify._reserve_usd == 0.0
    assert from_env.kol_search.max_results == 7
    assert from_env.enricher.max_videos == 6

    for service in (bundle, from_env):
        service.kol_search._apify._client.close()
        service.enricher._apify._client.close()
    bundle.facebook._apify._client.close()
