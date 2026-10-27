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
    ApifyError,
    ApifyLimits,
    estimate_run_cost,
)
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
    for name in _LIMIT_ENV:
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


def test_budget_records_are_thread_safe():
    budget = ApifyBudget(100.0)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: budget.record(_SEARCH_ACTOR, 0.001), range(1_000)))

    assert budget.spent == pytest.approx(1.0)


def test_estimate_run_cost_uses_the_actor_rate_and_the_run_cap():
    assert estimate_run_cost(_SEARCH_ACTOR, 2, 0.50) == pytest.approx(0.02)
    assert estimate_run_cost("unknown/actor", 0, 0.50) == pytest.approx(0.02)
    assert estimate_run_cost(_SEARCH_ACTOR, 1_000, 0.50) == 0.50


def test_client_records_an_estimate_then_refuses_without_http():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(201, json=[{"id": 1}, {"id": 2}])

    budget = ApifyBudget(0.02)
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
    assert budget.spent == pytest.approx(0.02)
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
    assert limits.budget().remaining == pytest.approx(1.00)


def test_limits_from_env_overrides(monkeypatch):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv("APIFY_JOB_BUDGET_USD", "2.5")
    monkeypatch.setenv("APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD", "0.6")
    monkeypatch.setenv("APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD", "0.7")
    monkeypatch.setenv("APIFY_FACEBOOK_MAX_CHARGE_USD", "0.8")

    limits = ApifyLimits.from_env()

    assert limits.job_budget_usd == pytest.approx(2.5)
    assert limits.tiktok_search_max_charge_usd == pytest.approx(0.6)
    assert limits.tiktok_profile_max_charge_usd == pytest.approx(0.7)
    assert limits.facebook_max_charge_usd == pytest.approx(0.8)


@pytest.mark.parametrize("name", _LIMIT_ENV)
@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_limits_from_env_rejects_invalid_values(monkeypatch, name, value):
    _clear_limit_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
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

    with pytest.raises(ValueError, match="must be at least 0.50"):
        ApifyLimits.from_env()


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
        budget=ApifyBudget(0.01),
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

    bundle = build_integrations(
        live_kols=True,
        live_fb=True,
        apify_token=_TOKEN,
        max_results_per_query=3,
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

    bundle.kol_search._apify._client.close()
    bundle.enricher._apify._client.close()
    bundle.facebook._apify._client.close()
