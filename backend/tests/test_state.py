import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.models import KolProfile, MatchRequest, MatchWeights
from app.state import MatchState, merge_kols


def make_kol(handle: str, source: str, **overrides) -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        source=source,
        **overrides,
    )


def make_request() -> MatchRequest:
    return MatchRequest(
        website_url="https://brand.example",
        facebook_url="https://www.facebook.com/brand",
    )


def test_merge_kols_dedupes_by_lowercase_handle_and_prefers_apify():
    cache = make_kol("Creator", "cache", followers=100)
    seed = make_kol("creator", "seed", followers=200)
    apify = make_kol("CREATOR", "apify", followers=300)
    other = make_kol("someone-else", "cache")

    merged = merge_kols([cache, seed], [apify, other])

    by_handle = {kol.handle.lower(): kol for kol in merged}
    assert set(by_handle) == {"creator", "someone-else"}
    assert by_handle["creator"].source == "apify"
    assert by_handle["creator"].followers == 300


def test_merge_kols_prefers_seed_over_cache():
    cache = make_kol("creator", "cache")
    seed = make_kol("creator", "seed")

    assert merge_kols([cache], [seed])[0].source == "seed"
    assert merge_kols([seed], [cache])[0].source == "seed"


def test_match_weights_normalize_to_one():
    weights = MatchWeights(relevance=4, engagement=2, audience=2, tier=2)

    assert weights.relevance == pytest.approx(0.4)
    assert weights.engagement == pytest.approx(0.2)
    assert weights.audience == pytest.approx(0.2)
    assert weights.tier == pytest.approx(0.2)
    assert sum(weights.model_dump().values()) == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["relevance", "engagement", "audience", "tier"])
def test_match_weights_reject_negative(field):
    values = {"relevance": 1.0, "engagement": 1.0, "audience": 1.0, "tier": 1.0}
    values[field] = -0.1

    with pytest.raises(ValidationError):
        MatchWeights(**values)


def test_match_weights_reject_all_zero():
    with pytest.raises(ValidationError, match="must not all be zero"):
        MatchWeights(relevance=0, engagement=0, audience=0, tier=0)


def test_graph_loop_accumulates_discovered_and_filter_can_shrink():
    def discover(state: MatchState) -> dict:
        round_no = state.search_rounds + 1
        if round_no == 1:
            found = [
                make_kol("Creator", "cache", followers=50),
                make_kol("shared", "cache", followers=500),
            ]
        else:
            found = [
                make_kol("CREATOR", "apify", followers=5_000),
                make_kol("shared", "seed", followers=500),
                make_kol("fresh", "apify", followers=10),
            ]
        return {
            "discovered": found,
            "search_rounds": round_no,
            "errors": [f"discover round {round_no}"],
        }

    def filter_thai(state: MatchState) -> dict:
        kept = [kol for kol in state.discovered if kol.followers >= 500]
        return {"candidates": kept, "errors": ["filter_thai"]}

    def next_node(state: MatchState) -> str:
        return "discover" if state.search_rounds < 2 else "filter_thai"

    graph = StateGraph(MatchState)
    graph.add_node("discover", discover)
    graph.add_node("filter_thai", filter_thai)
    graph.add_edge(START, "discover")
    graph.add_conditional_edges("discover", next_node, ["discover", "filter_thai"])
    graph.add_edge("filter_thai", END)

    result = graph.compile().invoke({"request": make_request()})

    assert result["search_rounds"] == 2
    by_handle = {kol.handle.lower(): kol for kol in result["discovered"]}
    assert set(by_handle) == {"creator", "shared", "fresh"}
    # Reducer keeps the highest-priority source across rounds.
    assert by_handle["creator"].source == "apify"
    assert by_handle["creator"].followers == 5_000
    assert by_handle["shared"].source == "seed"

    # `candidates` overwrites (filter can shrink), `discovered` does not.
    assert [kol.handle for kol in result["candidates"]] == ["CREATOR", "shared"]
    assert len(result["candidates"]) < len(result["discovered"])
    assert result["errors"] == [
        "discover round 1",
        "discover round 2",
        "filter_thai",
    ]
