from datetime import UTC, datetime, timedelta

from langgraph.graph import END, START, StateGraph

from app.models import KolProfile, MatchRequest, VideoStats
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


def test_merge_kols_returns_the_union_of_videos_newest_first():
    def video(suffix: str, days: int | None) -> VideoStats:
        posted_at = None if days is None else datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=days)
        return VideoStats(
            url=f"https://www.tiktok.com/@creator/video/{suffix}",
            views=100,
            likes=10,
            comments=1,
            shares=1,
            posted_at=posted_at,
            caption=f"video {suffix}",
        )

    older = make_kol(
        "creator",
        "cache",
        videos=[video("shared", 2), video("old", 3), video("shared", 2)],
    )
    newer = make_kol(
        "CREATOR",
        "apify",
        videos=[video("middle", 1), video("newest", 5), video("undated", None)],
    )

    winner = merge_kols([older], [newer])[0]

    assert winner.source == "apify"
    assert [item.url.rsplit("/", 1)[-1] for item in winner.videos] == [
        "newest",
        "old",
        "shared",
        "middle",
        "undated",
    ]


def test_merge_kols_unions_videos_for_equal_priority_rows():
    first = make_kol(
        "creator",
        "seed",
        followers=1,
        videos=[
            VideoStats(
                url="https://www.tiktok.com/@creator/video/1",
                posted_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    )
    later = make_kol(
        "Creator",
        "seed",
        followers=2,
        videos=[
            VideoStats(
                url="https://www.tiktok.com/@creator/video/2",
                posted_at=datetime(2026, 1, 2, tzinfo=UTC),
            )
        ],
    )

    winner = merge_kols([first], [later])[0]

    assert winner.followers == 1
    assert [item.url.rsplit("/", 1)[-1] for item in winner.videos] == ["2", "1"]


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

    def hard_filter(state: MatchState) -> dict:
        kept = [kol for kol in state.discovered if kol.followers >= 500]
        return {"candidates": kept, "errors": ["hard_filter"]}

    def next_node(state: MatchState) -> str:
        return "discover" if state.search_rounds < 2 else "hard_filter"

    graph = StateGraph(MatchState)
    graph.add_node("discover", discover)
    graph.add_node("hard_filter", hard_filter)
    graph.add_edge(START, "discover")
    graph.add_conditional_edges("discover", next_node, ["discover", "hard_filter"])
    graph.add_edge("hard_filter", END)

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
        "hard_filter",
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

