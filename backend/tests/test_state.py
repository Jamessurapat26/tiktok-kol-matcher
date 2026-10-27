import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.models import KolProfile, MatchRequest, MatchWeights, ScoreBreakdown
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


def test_merge_kols_coerces_dict_updates():
    cache = {
        "handle": "@Creator",
        "url": "https://www.tiktok.com/@creator",
        "source": "cache",
        "followers": 100,
    }
    apify = {
        "handle": " creator ",
        "url": "https://www.tiktok.com/@creator",
        "source": "apify",
        "followers": 300,
    }

    merged = merge_kols([cache], [apify])

    assert len(merged) == 1
    assert isinstance(merged[0], KolProfile)
    assert merged[0].source == "apify"
    assert merged[0].followers == 300


def test_merge_kols_prefers_seed_over_cache():
    cache = make_kol("creator", "cache")
    seed = make_kol("creator", "seed")

    assert merge_kols([cache], [seed])[0].source == "seed"
    assert merge_kols([seed], [cache])[0].source == "seed"


def test_merge_kols_equal_priority_keeps_the_earlier_row():
    earlier = make_kol("creator", "seed", followers=1)
    later = make_kol("CREATOR", "seed", followers=2, bio="later bio")

    merged = merge_kols([earlier], [later])

    assert merged[0].source == "seed"
    assert merged[0].followers == 1
    assert merged[0].handle == "creator"
    # Equal priority does not fill from the later row.
    assert merged[0].bio is None


def test_merge_kols_fills_missing_fields_from_lower_priority():
    seed = make_kol(
        "creator",
        "seed",
        followers=200,
        bio="from seed",
        language="th",
        nickname="Nick",
        region="TH",
        content_summary="seed topics",
        median_engagement_rate=0.05,
    )
    apify = make_kol("Creator", "apify", followers=300)

    merged = merge_kols([seed], [apify])

    assert len(merged) == 1
    winner = merged[0]
    assert winner.source == "apify"
    assert winner.handle == "Creator"
    assert winner.followers == 300
    assert winner.bio == "from seed"
    assert winner.language == "th"
    assert winner.nickname == "Nick"
    assert winner.region == "TH"
    assert winner.content_summary == "seed topics"
    assert winner.median_engagement_rate == 0.05


def test_merge_kols_fills_when_the_lower_priority_row_arrives_later():
    apify = make_kol("Creator", "apify", followers=300, bio="kept")
    cache = make_kol(
        "creator",
        "cache",
        followers=10,
        bio="not used",
        language="th",
        content_summary="cached topics",
    )

    winner = merge_kols([apify], [cache])[0]

    assert winner.source == "apify"
    assert winner.followers == 300
    assert winner.bio == "kept"
    assert winner.language == "th"
    assert winner.content_summary == "cached topics"


def test_merge_kols_does_not_treat_zero_as_missing():
    seed = make_kol("creator", "seed", followers=50, posts_per_week=4)
    apify = make_kol("creator", "apify", followers=0, posts_per_week=0)

    winner = merge_kols([seed], [apify])[0]

    assert winner.source == "apify"
    assert winner.followers == 0
    assert winner.posts_per_week == 0


def test_merge_kols_fills_none_fields_through_the_priority_chain():
    cache = make_kol("creator", "cache", bio="cache bio")
    seed = make_kol("creator", "seed", language="th")
    apify = make_kol("CREATOR", "apify", followers=10)

    winner = merge_kols([cache, seed], [apify])[0]

    assert winner.source == "apify"
    assert winner.handle == "CREATOR"
    assert winner.followers == 10
    assert winner.bio == "cache bio"
    assert winner.language == "th"


def test_merge_kols_fills_missing_fields_from_dict_rows():
    low = {
        "handle": "creator",
        "url": "https://www.tiktok.com/@creator",
        "source": "cache",
        "bio": "hello",
        "followers": 10,
    }
    high = {
        "handle": "Creator",
        "url": "https://www.tiktok.com/@Creator",
        "source": "apify",
        "followers": 20,
    }

    merged = merge_kols([low], [high])

    assert merged[0].source == "apify"
    assert merged[0].followers == 20
    assert merged[0].bio == "hello"


def test_handle_strips_at_sign_and_preserves_case():
    foo = KolProfile(
        handle="@Foo",
        url="https://www.tiktok.com/@Foo",
        source="apify",
    )
    bar = KolProfile(
        handle="  @Bar ",
        url="https://www.tiktok.com/@Bar",
        source="seed",
    )

    assert foo.handle == "Foo"
    assert bar.handle == "Bar"

    merged = merge_kols(
        [
            {
                "handle": "@Foo",
                "url": "https://www.tiktok.com/@Foo",
                "source": "apify",
            }
        ],
        [
            {
                "handle": "  @Bar ",
                "url": "https://www.tiktok.com/@Bar",
                "source": "seed",
            }
        ],
    )

    assert [kol.handle for kol in merged] == ["Foo", "Bar"]


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


def test_match_weights_defaults_fill_and_apply():
    weights = MatchWeights()
    assert sum(weights.model_dump().values()) == pytest.approx(1.0)

    request = make_request()
    assert sum(request.weights.model_dump().values()) == pytest.approx(1.0)

    scores = ScoreBreakdown(
        relevance=1,
        engagement=1,
        audience_fit=1,
        tier_fit=1,
        audience_confidence="high",
    )
    assert request.weights.apply(scores) == pytest.approx(1.0)


def test_match_weights_missing_fields_get_defaults_before_normalizing():
    weights = MatchWeights(relevance=1)

    assert sum(weights.model_dump().values()) == pytest.approx(1.0)
    assert weights.engagement == pytest.approx(0.25 / 1.6)
    assert weights.audience == pytest.approx(0.2 / 1.6)
    assert weights.tier == pytest.approx(0.15 / 1.6)


def test_match_weights_assignment_raises():
    weights = MatchWeights()

    with pytest.raises(ValidationError):
        weights.relevance = 1.0

    assert sum(weights.model_dump().values()) == pytest.approx(1.0)


def test_match_weights_model_copy_revalidates_update():
    weights = MatchWeights()

    copied = weights.model_copy(update={"relevance": 9})

    assert copied is not weights
    assert sum(copied.model_dump().values()) == pytest.approx(1.0)
    assert copied.relevance == pytest.approx(9 / 9.6)


def test_match_weights_model_validate_normalizes():
    weights = MatchWeights.model_validate(
        {"relevance": 4, "engagement": 2, "audience": 2, "tier": 2}
    )

    assert weights.relevance == pytest.approx(0.4)
    assert weights.engagement == pytest.approx(0.2)
    assert weights.audience == pytest.approx(0.2)
    assert weights.tier == pytest.approx(0.2)
    assert sum(weights.model_dump().values()) == pytest.approx(1.0)


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


def test_graph_node_returning_dicts_is_coerced_by_reducer():
    def discover(state: MatchState) -> dict:
        return {
            "discovered": [
                {
                    "handle": "@Creator",
                    "url": "https://www.tiktok.com/@creator",
                    "source": "cache",
                    "followers": 100,
                },
                {
                    "handle": "creator",
                    "url": "https://www.tiktok.com/@creator",
                    "source": "apify",
                    "followers": 300,
                },
            ]
        }

    graph = StateGraph(MatchState)
    graph.add_node("discover", discover)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", END)

    result = graph.compile().invoke({"request": make_request()})

    assert len(result["discovered"]) == 1
    assert all(isinstance(kol, KolProfile) for kol in result["discovered"])
    assert result["discovered"][0].source == "apify"
    assert result["discovered"][0].followers == 300


def test_graph_accumulates_search_queries_across_rounds():
    def generate_queries(state: MatchState) -> dict:
        round_no = state.search_rounds + 1
        return {
            "search_queries": [f"query {round_no}"],
            "search_rounds": round_no,
        }

    def next_node(state: MatchState) -> str:
        return "generate_queries" if state.search_rounds < 3 else END

    graph = StateGraph(MatchState)
    graph.add_node("generate_queries", generate_queries)
    graph.add_edge(START, "generate_queries")
    graph.add_conditional_edges(
        "generate_queries", next_node, ["generate_queries", END]
    )

    result = graph.compile().invoke({"request": make_request()})

    assert result["search_queries"] == ["query 1", "query 2", "query 3"]

