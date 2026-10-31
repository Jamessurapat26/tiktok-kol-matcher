"""The default live job plan must fit the $1.55 budget with 20+ enrichments."""

import os

import pytest

from app.integrations.apify import (
    ApifyLimits,
    TIKTOK_ACTOR,
    estimate_default_job_usd,
    estimate_run_cost,
)
from app.matching_config import DEFAULT_CONFIG


def _clear_apify_env(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith(("APIFY_", "TIKTOK_")):
            monkeypatch.delenv(name, raising=False)


def _uncapped_run_cost(actor: str, items: int, *, proxied: bool = False) -> float:
    return estimate_run_cost(actor, items, float("inf"), proxied=proxied)


def test_default_job_plan_enriches_at_least_twenty_profiles(monkeypatch):
    _clear_apify_env(monkeypatch)
    limits = ApifyLimits.from_env()

    estimate = estimate_default_job_usd(limits)

    assert limits.job_budget_usd == pytest.approx(1.55)
    assert estimate.profile_runs >= 20
    # Every first-pass leader the config asks for fits in the budget.
    assert estimate.profile_runs >= DEFAULT_CONFIG.enrich_top_n
    assert 6 <= estimate.search_runs <= DEFAULT_CONFIG.max_search_queries
    assert estimate.total_usd == pytest.approx(
        estimate.facebook_usd + estimate.search_usd + estimate.enrich_usd
    )
    assert estimate.total_usd <= limits.job_budget_usd


def test_worst_case_searches_still_leave_room_for_the_planned_profiles(
    monkeypatch,
):
    _clear_apify_env(monkeypatch)
    limits = ApifyLimits.from_env()
    estimate = estimate_default_job_usd(limits)

    search_run = _uncapped_run_cost(
        TIKTOK_ACTOR,
        limits.tiktok_results_per_query,
        proxied=True,
    )
    profile_run = _uncapped_run_cost(
        TIKTOK_ACTOR,
        limits.tiktok_profile_videos,
    )
    worst_case = (
        estimate.facebook_usd
        + DEFAULT_CONFIG.max_search_queries * search_run
        + estimate.profile_runs * profile_run
    )

    assert worst_case <= limits.job_budget_usd
