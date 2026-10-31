import json
import operator
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.brand_match import normalize_for_match
from app.graph import build_graph, initial_state
from app.integrations import mock_integrations
from app.integrations.kol_enricher import MockKolEnricher
from app.integrations.llm import MockBrandAnalyzer
from app.matching_config import DEFAULT_CONFIG
from app.models import KolProfile, MatchRequest
from app.nodes import _guarded, make_nodes
from app.state import MatchState

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "kols.json"
_HARD_FILTERED_HANDLES = {
    "examplebrand.th",
    "otherlens_official",
    "gadgetshop_bkk",
    "gamernews.th",
    "inactivegamer.tan",
    "tinycreator.num",
    "megastar.gamer",
    "screenreview.joe",
}


def make_request(top_k: int = 10) -> MatchRequest:
    return MatchRequest(
        website_url="https://example-brand.co.th",
        facebook_url="https://www.facebook.com/examplebrand",
        top_k=top_k,
    )


def run_stream(graph, request):
    """Collect node names in execution order and the final state."""
    names: list[str] = []
    final: dict = {}
    for mode, payload in graph.stream(
        initial_state(request), stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            names.extend(payload.keys())
        else:
            final = payload
    return names, final


def make_fixture(handle: str, group: int, **overrides) -> dict:
    fixture = {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle.lstrip('@')}",
        "nickname": handle,
        "followers": 50_000,
        "bio": "Mock Thai gaming creator.",
        "source": "apify",
        "text_language": "th",
        "location_created": "TH",
        "topics": ["gaming", "tech"],
        "query_group": group,
    }
    fixture.update(overrides)
    return fixture


def test_default_pipeline_end_to_end():
    request = make_request(top_k=10)
    final = build_graph().invoke(initial_state(request))

    assert final["stage"] == "done"
    assert final["errors"] == []
    assert len(final["results"]) == 10
    assert len(final["ranked"]) > len(final["results"])
    assert {item.kol.handle for item in final["results"]} <= {
        item.kol.handle for item in final["ranked"]
    }

    finals = [item.final for item in final["ranked"]]
    assert finals == sorted(finals, reverse=True)
    for item in final["ranked"]:
        assert item.final == pytest.approx(item.scores.F * item.quality)
        assert item.profile_url.startswith("https://www.tiktok.com/@")
        assert item.reason

    # MMR picks a diverse top-k, so the result order need not be final-desc.
    assert len({item.kol.handle for item in final["results"]}) == 10


def test_loop_stops_at_max_rounds_even_with_too_few_candidates():
    fixtures = [
        make_fixture(f"kol-{group}-{index}", group)
        for group in (1, 2)
        for index in range(3)
    ]
    graph = build_graph(
        mock_integrations(kol_fixtures=fixtures),
        min_candidates=30,
        cfg=replace(DEFAULT_CONFIG, max_search_queries=20),
    )

    names, final = run_stream(graph, make_request())

    assert final["search_rounds"] == 3
    assert names.count("generate_queries") == 3
    assert final["stage"] == "done"

    queries = final["search_queries"]
    assert "blue light glasses thailand" in queries  # round 1
    assert "gaming creator thailand" in queries  # round 2
    normalized = [query.strip().casefold() for query in queries]
    assert len(set(normalized)) == len(normalized)


def test_loop_stops_early_when_enough_candidates():
    final = build_graph(min_candidates=5).invoke(initial_state(make_request()))

    assert final["search_rounds"] == 1
    assert final["stage"] == "done"


def non_thai_fixture(handle: str, group: int) -> dict:
    return make_fixture(
        handle,
        group,
        text_language="en",
        location_created="US",
        bio="English gaming creator.",
    )


def test_discovery_keeps_searching_while_thai_creators_are_short():
    # Round 1 is mostly non-Thai: 5 Thai out of 40 must not stop discovery.
    fixtures = [make_fixture(f"thai-1-{index}", 1) for index in range(5)]
    fixtures += [non_thai_fixture(f"foreign-1-{index}", 1) for index in range(35)]
    fixtures += [make_fixture(f"thai-2-{index}", 2) for index in range(30)]

    graph = build_graph(mock_integrations(kol_fixtures=fixtures), min_candidates=30)
    names, final = run_stream(graph, make_request())

    assert final["search_rounds"] == 2
    assert names.count("generate_queries") == 2
    assert len(final["candidates"]) >= 30
    assert final["stage"] == "done"


def test_discovery_stops_after_one_round_when_thai_target_is_met():
    fixtures = [make_fixture(f"thai-{index}", 1) for index in range(30)]

    graph = build_graph(mock_integrations(kol_fixtures=fixtures), min_candidates=30)
    names, final = run_stream(graph, make_request())

    assert final["search_rounds"] == 1
    assert names.count("generate_queries") == 1
    assert final["stage"] == "done"


def test_discovery_never_exceeds_max_search_rounds():
    fixtures = [make_fixture(f"thai-1-{index}", 1) for index in range(5)]
    fixtures += [non_thai_fixture(f"foreign-1-{index}", 1) for index in range(35)]
    fixtures += [
        non_thai_fixture(f"foreign-{group}-{index}", group)
        for group in (2, 3)
        for index in range(10)
    ]

    graph = build_graph(
        mock_integrations(kol_fixtures=fixtures),
        min_candidates=30,
        max_search_rounds=3,
        cfg=replace(DEFAULT_CONFIG, max_search_queries=20),
    )
    names, final = run_stream(graph, make_request())

    assert final["search_rounds"] == 3
    assert names.count("generate_queries") == 3
    assert final["stage"] == "done"


def test_discovery_stops_at_the_query_cap_even_with_too_few_candidates():
    fixtures = [
        make_fixture(f"kol-{group}-{index}", group)
        for group in (1, 2)
        for index in range(3)
    ]

    graph = build_graph(
        mock_integrations(kol_fixtures=fixtures),
        min_candidates=30,
    )
    names, final = run_stream(graph, make_request())

    assert DEFAULT_CONFIG.max_search_queries == 10
    assert final["search_rounds"] == 2
    assert names.count("generate_queries") == 2
    assert len(final["search_queries"]) == 10
    assert final["stage"] == "done"


def test_parallel_scrapers_both_populate_state():
    graph = build_graph()
    names, final = run_stream(graph, make_request())

    assert set(names[:2]) == {"scrape_website", "scrape_facebook"}
    assert names.count("analyze_brand") == 1
    assert names.index("analyze_brand") > names.index("scrape_website")
    assert names.index("analyze_brand") > names.index("scrape_facebook")
    assert final["website_content"]
    assert final["facebook_page"] is not None

    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    assert ("scrape_website", "research_brand") in edges
    assert ("scrape_facebook", "research_brand") in edges
    assert ("research_brand", "analyze_brand") in edges


def test_hard_filter_shrinks_discovered():
    final = build_graph().invoke(initial_state(make_request()))

    assert len(final["candidates"]) < len(final["discovered"])
    for kol in final["candidates"]:
        assert (kol.language or "").casefold() in {"th", "thai"} or (
            kol.region or ""
        ).upper() == "TH"
    assert any(
        (kol.language or "").casefold() not in {"th", "thai"}
        and (kol.region or "").upper() != "TH"
        for kol in final["discovered"]
    )
    assert final["rejected"]


def test_dedupe_across_sources():
    total_records = len(json.loads(_KOL_FIXTURE.read_text(encoding="utf-8")))
    final = build_graph().invoke(initial_state(make_request()))

    handles = [kol.handle.strip().lower().lstrip("@") for kol in final["discovered"]]
    assert len(set(handles)) == len(handles)
    assert len(final["discovered"]) < total_records


def test_ranked_contains_every_scored_candidate_and_is_deterministic():
    request = make_request(top_k=3)

    first = build_graph().invoke(initial_state(request))
    second = build_graph().invoke(initial_state(request))

    assert len(first["ranked"]) == len(first["candidates"])
    assert len(first["results"]) == 3
    assert [item.kol.handle for item in first["results"]] == [
        item.kol.handle for item in second["results"]
    ]


def test_website_failure_uses_facebook_and_does_not_fail_the_job():
    class BoomReader:
        def read(self, url: str) -> str:
            raise RuntimeError("boom")

    integrations = replace(mock_integrations(), web_reader=BoomReader())
    names, final = run_stream(build_graph(integrations), make_request())

    assert final["stage"] == "done"
    assert "scrape_website: boom" in final["errors"]
    assert "analyze_brand: missing website_content" in final["errors"]
    assert final["brand_sources"] == ["website: skipped", "facebook: used"]
    assert final["brand_profile"] is not None
    assert "analyze_brand" in names
    assert "generate_queries" in names


def test_both_brand_sources_failing_is_fatal():
    class BoomReader:
        def read(self, url: str) -> str:
            raise RuntimeError("website down")

    class BoomFacebook:
        def fetch_page(self, url: str):
            raise RuntimeError("facebook down")

    integrations = replace(
        mock_integrations(),
        web_reader=BoomReader(),
        facebook=BoomFacebook(),
    )
    names, final = run_stream(build_graph(integrations), make_request())

    assert final["stage"] == "failed"
    assert (
        "analyze_brand: no usable brand source "
        "(website: website down, facebook: facebook down)"
    ) in final["errors"]
    assert names[-1] == "analyze_brand"
    assert "generate_queries" not in names


def test_facebook_scrape_failure_is_a_warning():
    class BoomFacebook:
        def fetch_page(self, url: str):
            raise RuntimeError("fb down")

    integrations = replace(mock_integrations(), facebook=BoomFacebook())
    final = build_graph(integrations).invoke(initial_state(make_request()))

    assert final["stage"] == "done"
    assert any(error.startswith("scrape_facebook:") for error in final["errors"])
    assert "analyze_brand: missing facebook_page" in final["errors"]


def test_search_failure_stops_without_retrying():
    class BoomSearch:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, queries: list[str], round: int) -> list[dict]:
            self.calls += 1
            raise RuntimeError("search down")

    search = BoomSearch()
    integrations = replace(mock_integrations(), kol_search=search)
    names, final = run_stream(build_graph(integrations), make_request())

    assert search.calls == 1
    assert final["stage"] == "failed"
    assert final["search_rounds"] == 1
    assert any(
        "discover_kols: search failed: search down" in error
        for error in final["errors"]
    )
    assert "hard_filter" not in names


def test_search_failure_after_a_successful_round_is_a_warning():
    class FailSecondRound:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.calls = 0

        def search(self, queries: list[str], round: int) -> list[dict]:
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("round 2 down")
            return self.inner.search(queries, round)

    base = mock_integrations()
    search = FailSecondRound(base.kol_search)
    integrations = replace(base, kol_search=search)
    final = build_graph(
        integrations,
        min_candidates=30,
        max_search_rounds=2,
    ).invoke(initial_state(make_request()))

    assert search.calls == 2
    assert final["stage"] == "done"
    assert final["discovered"]
    assert any("discover_kols: search failed:" in error for error in final["errors"])


def test_invalid_discovery_row_is_skipped():
    fixtures = json.loads(_KOL_FIXTURE.read_text(encoding="utf-8"))
    fixtures.append(make_fixture("this-row-is-invalid", 1, source="not-a-source"))
    baseline = build_graph().invoke(initial_state(make_request()))
    final = build_graph(mock_integrations(kol_fixtures=fixtures)).invoke(
        initial_state(make_request())
    )

    assert final["stage"] == "done"
    assert {kol.handle for kol in final["discovered"]} == {
        kol.handle for kol in baseline["discovered"]
    }
    assert "this-row-is-invalid" not in {kol.handle for kol in final["discovered"]}
    assert any("discover_kols: skipped row" in error for error in final["errors"])


def test_discover_searches_only_the_latest_round_queries():
    class RecordingSearch:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.calls: list[tuple[int, list[str]]] = []

        def search(self, queries: list[str], round: int) -> list[dict]:
            self.calls.append((round, list(queries)))
            return self.inner.search(queries, round)

    base = mock_integrations()
    search = RecordingSearch(base.kol_search)
    integrations = replace(base, kol_search=search)
    final = build_graph(
        integrations,
        min_candidates=30,
        max_search_rounds=2,
    ).invoke(initial_state(make_request()))

    assert [round_no for round_no, _ in search.calls] == [1, 2]
    round_1 = search.calls[0][1]
    round_2 = search.calls[1][1]
    assert round_1 and round_2
    assert set(round_1).isdisjoint(round_2)
    assert "blue light glasses thailand" in round_1
    assert "blue light glasses thailand" not in round_2
    assert "gaming creator thailand" in round_2
    assert round_2 == final["last_queries"]
    assert final["search_queries"] == round_1 + round_2


def test_enricher_exception_keeps_the_kol_and_continues():
    class Boom:
        def enrich(self, kol):
            if kol.handle == "Boom":
                raise RuntimeError("profile down")
            return {"posts_per_week": 1.0}

    fixtures = [
        make_fixture("Boom", 1),
        make_fixture("Fine", 1),
    ]
    integrations = replace(mock_integrations(kol_fixtures=fixtures), enricher=Boom())
    final = build_graph(integrations, min_candidates=1).invoke(
        initial_state(make_request())
    )

    by_handle = {kol.handle: kol for kol in final["candidates"]}
    assert final["stage"] == "done"
    assert by_handle["Boom"].posts_per_week == 0
    assert by_handle["Fine"].posts_per_week == 1.0
    assert any(
        "enrich_kols: failed for Boom: profile down" in error
        for error in final["errors"]
    )


def test_invalid_enrichment_keeps_the_original_kol():
    class SelectiveEnricher:
        def enrich(self, kol):
            if kol.handle == "BadSource":
                return {"stats_source": "bogus"}
            if kol.handle == "BadFollowers":
                return {"followers": -5}
            return {"posts_per_week": 2.0}

    fixtures = [
        make_fixture("BadSource", 1, followers=10_000),
        make_fixture("BadFollowers", 1, followers=20_000),
        make_fixture("GoodKol", 1, followers=30_000),
    ]
    integrations = replace(
        mock_integrations(kol_fixtures=fixtures),
        enricher=SelectiveEnricher(),
    )
    final = build_graph(integrations, min_candidates=1).invoke(
        initial_state(make_request())
    )

    by_handle = {kol.handle: kol for kol in final["candidates"]}
    assert final["stage"] == "done"
    assert by_handle["BadSource"].stats_source == "search"
    assert by_handle["BadSource"].followers == 10_000
    assert by_handle["BadFollowers"].followers == 20_000
    assert by_handle["BadFollowers"].posts_per_week == 0
    assert by_handle["GoodKol"].posts_per_week == 2.0
    assert any(
        "enrich_kols: rejected enrichment for BadSource" in error
        for error in final["errors"]
    )
    assert any(
        "enrich_kols: rejected enrichment for BadFollowers" in error
        for error in final["errors"]
    )


def _engagement_fixture(handle: str, likes: int) -> dict:
    return make_fixture(
        handle,
        1,
        followers=50_000,
        bio="Thai gaming creator reviewing blue-light glasses.",
        videos=[
            {
                "url": f"https://www.tiktok.com/@{handle}/video/{index}",
                "views": 10_000,
                "likes": likes,
                "comments": 20,
                "shares": 5,
                "days_ago": index + 1,
                "caption": "รีวิวแว่นเกมมิ่ง gaming glasses #eyewear",
            }
            for index in range(5)
        ],
    )


def test_higher_engagement_ranks_first_when_handles_would_not():
    fixtures = [
        _engagement_fixture("aaa_low", likes=20),
        _engagement_fixture("zzz_high", likes=900),
    ]
    integrations = mock_integrations(kol_fixtures=fixtures)
    final = build_graph(
        integrations,
        min_candidates=1,
    ).invoke(initial_state(make_request()))

    ranked = final["ranked"]
    assert [item.kol.handle for item in ranked] == ["zzz_high", "aaa_low"]
    assert ranked[0].scores.E > ranked[1].scores.E
    assert ranked[0].final > ranked[1].final


def test_no_thai_kols_is_fatal():
    fixtures = [
        make_fixture("foreign", 1, text_language="en", location_created="US")
    ]
    names, final = run_stream(
        build_graph(mock_integrations(kol_fixtures=fixtures), min_candidates=1),
        make_request(),
    )

    assert final["stage"] == "failed"
    assert any(
        "hard_filter: no candidates passed the hard filters" in error
        for error in final["errors"]
    )
    assert "enrich_kols" not in names


def test_enricher_prefers_higher_priority_fixture_and_keeps_ties():
    def fixture(handle: str, source: str, caption: str) -> dict:
        return {
            "handle": handle,
            "source": source,
            "profile_videos": [
                {
                    "url": f"https://www.tiktok.com/@creator/video/{source}",
                    "views": 100,
                    "likes": 10,
                    "comments": 1,
                    "shares": 1,
                    "days_ago": 2,
                    "caption": caption,
                }
            ],
        }

    apify = fixture("@Creator", "apify", "from apify #gaming")
    seed = fixture("creator", "seed", "from seed #eyewear")
    kol = KolProfile(
        handle="creator",
        url="https://www.tiktok.com/@creator",
        source="cache",
    )

    for fixtures in ([apify, seed], [seed, apify]):
        update = MockKolEnricher(fixtures).enrich(kol)
        assert update["stats_source"] == "profile"
        assert update["videos"][0]["url"].endswith("/apify")
        assert "from apify" in update["content_summary"]

    earlier = fixture("creator", "seed", "earlier #gaming")
    later = fixture("CREATOR", "seed", "later #eyewear")
    update = MockKolEnricher([earlier, later]).enrich(kol)
    assert update["videos"][0]["url"].endswith("/seed")
    assert "earlier" in update["content_summary"]


class _CountingEnricher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enrich(self, kol: KolProfile) -> dict:
        self.calls.append(kol.handle)
        return {}


def test_config_enrich_top_n_enriches_the_first_pass_leaders():
    enricher = _CountingEnricher()
    integrations = replace(mock_integrations(), enricher=enricher)
    graph = build_graph(
        integrations,
        cfg=replace(DEFAULT_CONFIG, enrich_top_n=3),
    )

    final = graph.invoke(initial_state(make_request()))

    assert final["stage"] == "done"
    assert len(enricher.calls) == 3
    # candidates keep the first-pass order, so the first three are the leaders.
    assert [handle.lower() for handle in enricher.calls] == [
        kol.handle.lower() for kol in final["candidates"][:3]
    ]
    assert len(final["candidates"]) > 3


def test_default_enrichment_is_capped_by_the_config():
    enricher = _CountingEnricher()
    integrations = replace(mock_integrations(), enricher=enricher)
    final = build_graph(integrations).invoke(initial_state(make_request()))

    expected = min(len(final["candidates"]), DEFAULT_CONFIG.enrich_top_n)
    assert len(enricher.calls) == expected
    assert [handle.lower() for handle in enricher.calls] == [
        kol.handle.lower() for kol in final["candidates"][:expected]
    ]


def test_analyze_brand_merges_url_aliases_after_model_names():
    class _NamedAnalyzer(MockBrandAnalyzer):
        def analyze_brand(self, website_content, facebook_page, research=None):
            profile = super().analyze_brand(website_content, facebook_page)
            return profile.model_copy(
                update={"brand_names": ["ExampleBrand", "EB-24"]}
            )

    integrations = replace(mock_integrations(), llm=_NamedAnalyzer())
    nodes = make_nodes(integrations, min_candidates=1, max_search_rounds=1)
    request = MatchRequest(
        website_url="https://www.example-brand.co.th",
        facebook_url="https://www.facebook.com/ExampleBrandThailand",
    )

    update = nodes["analyze_brand"](
        MatchState(request=request, website_content="site text")
    )

    assert update["brand_profile"].brand_names == [
        "ExampleBrand",
        "EB-24",
        "ExampleBrandThailand",
    ]


def test_judge_is_one_batched_call_on_at_most_judge_top_n():
    class _CountingJudge(MockBrandAnalyzer):
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def judge(self, brief, kols):
            self.calls.append([kol.handle for kol in kols])
            return super().judge(brief, kols)

    analyzer = _CountingJudge()
    cfg = replace(DEFAULT_CONFIG, judge_top_n=5)
    final = build_graph(
        replace(mock_integrations(), llm=analyzer),
        cfg=cfg,
    ).invoke(initial_state(make_request(top_k=3)))

    assert final["stage"] == "done"
    assert len(analyzer.calls) == 1
    assert len(analyzer.calls[0]) == cfg.judge_top_n
    assert sum(1 for item in final["ranked"] if item.scores.judged) == 5


def test_judge_failure_leaves_everyone_unjudged_but_ranked():
    class _BoomJudge(MockBrandAnalyzer):
        def judge(self, brief, kols):
            raise RuntimeError("judge down")

    final = build_graph(
        replace(mock_integrations(), llm=_BoomJudge())
    ).invoke(initial_state(make_request()))

    assert final["stage"] == "done"
    assert final["ranked"]
    assert not any(item.scores.judged for item in final["ranked"])
    assert any("judge: judge down" in error for error in final["errors"])


def test_cost_log_records_every_paid_stage():
    final = build_graph().invoke(initial_state(make_request()))
    lines = final["cost_log"]

    assert "llm: analyze_brand 1 call" in lines
    assert "llm: generate_queries round 1 1 call" in lines
    assert any(
        line.startswith("search round 1: est $")
        and line.endswith("(8 queries)")
        for line in lines
    )
    assert any(
        line.startswith("enrich: est $")
        and line.endswith("(20 profiles)")
        for line in lines
    )
    assert any(
        line.startswith("llm: judge 1 call (")
        and line.endswith(" creators)")
        for line in lines
    )
    assert sum(1 for line in lines if line.startswith("search round ")) == 2


def test_mmr_keeps_near_duplicate_creators_out_of_the_same_top_ten():
    final = build_graph().invoke(initial_state(make_request(top_k=10)))

    handles = {item.kol.handle for item in final["results"]}
    duplicates = {"gamingduo.peak", "gamingduo.peakteam"}
    assert len(handles & duplicates) <= 1


def test_demo_results_are_top_k_and_never_include_the_brand_handle():
    final = build_graph().invoke(initial_state(make_request(top_k=10)))

    assert final["stage"] == "done"
    assert len(final["results"]) == 10
    result_handles = {item.kol.handle for item in final["results"]}
    assert "examplebrand.th" not in result_handles
    assert result_handles.isdisjoint(_HARD_FILTERED_HANDLES)


def test_generate_queries_drops_brand_name_queries():
    class _BrandQueryAnalyzer(MockBrandAnalyzer):
        def generate_queries(self, brand, previous_queries, round):
            return [
                "gaming glasses thailand",
                "ExampleBrand review",
                "example-brand unboxing",
            ]

    integrations = replace(mock_integrations(), llm=_BrandQueryAnalyzer())
    nodes = make_nodes(integrations, min_candidates=1, max_search_rounds=1)
    state = MatchState(
        request=make_request(),
        brand_profile=MockBrandAnalyzer().analyze_brand("website", None),
    )

    update = nodes["generate_queries"](state)

    assert update["search_queries"] == ["gaming glasses thailand"]
    assert update["last_queries"] == ["gaming glasses thailand"]
    assert update["errors"] == [
        "generate_queries: dropped 2 brand-name queries"
    ]


def test_demo_graph_hard_filters_and_recall_guard():
    final = build_graph().invoke(initial_state(make_request(top_k=50)))

    assert final["stage"] == "done"
    result_handles = {item.kol.handle for item in final["results"]}
    assert result_handles.isdisjoint(_HARD_FILTERED_HANDLES)
    assert _HARD_FILTERED_HANDLES.isdisjoint(
        {kol.handle for kol in final["candidates"]}
    )
    assert final["rejected"]

    for query in final["search_queries"]:
        assert "examplebrand" not in normalize_for_match(query)


class _GuardState(BaseModel):
    stage: str = "received"
    answer: str | None = None
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)


def _single_node_graph(node, **compile_kwargs):
    builder = StateGraph(_GuardState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(**compile_kwargs)


def test_guarded_node_lets_interrupt_pause_and_resume():
    @_guarded("ask")
    def ask(state: _GuardState) -> dict:
        return {"answer": interrupt("approve?")}

    graph = _single_node_graph(ask, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "guarded-interrupt"}}

    paused = graph.invoke(_GuardState(), config)

    assert [item.value for item in paused["__interrupt__"]] == ["approve?"]
    assert paused["stage"] == "received"
    assert paused["errors"] == []

    resumed = graph.invoke(Command(resume="yes"), config)

    assert resumed["answer"] == "yes"
    assert resumed["stage"] == "received"
    assert resumed["errors"] == []


def test_guarded_node_still_records_ordinary_exceptions():
    @_guarded("boom")
    def boom(state: _GuardState) -> dict:
        raise ValueError("broken")

    final = _single_node_graph(boom).invoke(_GuardState())

    assert final["stage"] == "failed"
    assert final["errors"] == ["boom: broken"]
