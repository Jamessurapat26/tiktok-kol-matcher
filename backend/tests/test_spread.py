"""Section H behavior tests against the demo fixture pool.

Every test here reads one module-scoped demo run. The run uses the default
mock integrations (MockEmbedder included), so it never touches the network or
downloads a model.
"""

from __future__ import annotations

import statistics

import pytest

from app.brand_match import normalize_for_match
from app.graph import build_graph, initial_state
from app.models import MatchRequest

_BRAND_ALIASES = {"examplebrand", "examplebrandth"}


def _make_request(top_k: int = 10) -> MatchRequest:
    return MatchRequest(
        website_url="https://example-brand.co.th",
        facebook_url="https://facebook.com/examplebrand",
        top_k=top_k,
    )


@pytest.fixture(scope="module")
def demo_state() -> dict:
    """Run the demo mock graph once for every spread test in this module."""
    return build_graph().invoke(initial_state(_make_request(top_k=10)))


def _top(demo_state: dict, limit: int = 20) -> list:
    return demo_state["ranked"][:limit]


def test_scores_spread_across_the_demo_top_20(demo_state):
    top = _top(demo_state)
    assert len(top) == 20

    for name in ("F", "E", "T", "A_eff"):
        values = [getattr(item.scores, name) for item in top]
        stdev = statistics.pstdev(values)
        assert stdev > 0.05, (
            f"score {name} is near-constant across the demo top 20: "
            f"stdev={stdev:.6f} (expected > 0.05)"
        )


def test_brand_own_handle_never_ranked(demo_state):
    discovered = {normalize_for_match(kol.handle) for kol in demo_state["discovered"]}
    assert "examplebrandth" in discovered  # the fixture really contains the alias.

    ranked_handles = [item.kol.handle for item in demo_state["ranked"]]
    result_handles = [item.kol.handle for item in demo_state["results"]]
    offenders = [
        handle
        for handle in (*ranked_handles, *result_handles)
        if normalize_for_match(handle) in _BRAND_ALIASES
    ]

    assert not offenders, (
        "brand-owned handles must never be ranked or returned: "
        f"{sorted(set(offenders))}"
    )


def test_off_niche_top_engagement_ranks_below_on_niche_average(demo_state):
    ranked = demo_state["ranked"]
    by_handle = {item.kol.handle: item for item in ranked}
    wander = by_handle["wanderwithjin"]

    # On-niche = judged creators with at least half their recent videos on topic.
    on_niche = [
        item for item in ranked if item.scores.judged and item.scores.on_topic >= 0.5
    ]
    assert len(on_niche) >= 5
    median_e = statistics.median(item.scores.E for item in on_niche)
    average = min(on_niche, key=lambda item: abs(item.scores.E - median_e))

    result_handles = {item.kol.handle for item in demo_state["results"]}

    assert wander.scores.on_topic == 0  # off-niche
    assert wander.scores.E == max(item.scores.E for item in ranked)  # top engagement
    assert wander.kol.handle not in result_handles
    assert wander.final < average.final, (
        f"off-niche {wander.kol.handle} (final={wander.final:.4f}) must rank "
        f"below on-niche median-E {average.kol.handle} "
        f"(E={average.scores.E:.4f}, final={average.final:.4f})"
    )
