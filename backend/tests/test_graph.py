import json
from dataclasses import replace
from pathlib import Path

from app.graph import build_graph, initial_state
from app.integrations import mock_integrations
from app.integrations.kol_enricher import MockKolEnricher
from app.integrations.llm import MockBrandAnalyzer
from app.models import KolProfile, MatchRequest, ScoreBreakdown
from app.nodes import make_nodes
from app.state import MatchState

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "kols.json"


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
    assert final["results"] == final["ranked"][:10]

    scores = [item.total_score for item in final["results"]]
    assert scores == sorted(scores, reverse=True)
    for item in final["results"]:
        assert item.total_score == request.weights.apply(item.scores)


def test_loop_stops_at_max_rounds_even_with_too_few_candidates():
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
    )
    names, final = run_stream(graph, make_request())

    assert final["search_rounds"] == 3
    assert names.count("generate_queries") == 3
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
    assert ("scrape_website", "analyze_brand") in edges
    assert ("scrape_facebook", "analyze_brand") in edges


def test_filter_thai_shrinks_discovered():
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


def test_dedupe_across_sources():
    total_records = len(json.loads(_KOL_FIXTURE.read_text(encoding="utf-8")))
    final = build_graph().invoke(initial_state(make_request()))

    handles = [kol.handle.strip().lower().lstrip("@") for kol in final["discovered"]]
    assert len(set(handles)) == len(handles)
    assert len(final["discovered"]) < total_records


def test_prefilter_cuts_shortlist_and_is_deterministic():
    request = make_request(top_k=3)
    graph = build_graph(prefilter_n=5)

    first = graph.invoke(initial_state(request))
    second = graph.invoke(initial_state(request))

    assert len(first["shortlisted"]) == 5
    assert len(first["ranked"]) == 5
    assert len(first["results"]) == 3
    assert [item.kol.handle for item in first["results"]] == [
        item.kol.handle for item in second["results"]
    ]


def test_failing_integration_is_reported_not_raised():
    class BoomReader:
        def read(self, url: str) -> str:
            raise RuntimeError("boom")

    integrations = replace(mock_integrations(), web_reader=BoomReader())
    names, final = run_stream(build_graph(integrations), make_request())

    assert final["stage"] == "failed"
    assert "scrape_website: boom" in final["errors"]
    assert "analyze_brand: missing website_content" in final["errors"]
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
    assert "filter_thai" not in names


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
            return {"median_engagement_rate": 0.04, "posts_per_week": 1.0}

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
    assert by_handle["Boom"].median_engagement_rate is None
    assert by_handle["Fine"].median_engagement_rate == 0.04
    assert any(
        "enrich_kols: failed for Boom: profile down" in error
        for error in final["errors"]
    )


def test_invalid_enrichment_keeps_the_original_kol():
    class SelectiveEnricher:
        def enrich(self, kol):
            if kol.handle == "BadRate":
                return {"median_engagement_rate": 1.5}
            if kol.handle == "BadFollowers":
                return {"followers": -5}
            return {"median_engagement_rate": 0.05, "posts_per_week": 2.0}

    fixtures = [
        make_fixture("BadRate", 1, followers=10_000),
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
    assert by_handle["BadRate"].median_engagement_rate is None
    assert by_handle["BadRate"].followers == 10_000
    assert by_handle["BadFollowers"].followers == 20_000
    assert by_handle["BadFollowers"].median_engagement_rate is None
    assert by_handle["GoodKol"].median_engagement_rate == 0.05
    assert any(
        "enrich_kols: rejected enrichment for BadRate" in error
        for error in final["errors"]
    )
    assert any(
        "enrich_kols: rejected enrichment for BadFollowers" in error
        for error in final["errors"]
    )


def test_higher_engagement_ranks_first_when_handles_would_not():
    fixtures = [
        make_fixture("aaa_low", 1, followers=50_000, engagement_rate=0.09),
        make_fixture("zzz_high", 1, followers=50_000, engagement_rate=0.14),
    ]
    final = build_graph(
        mock_integrations(kol_fixtures=fixtures),
        min_candidates=1,
    ).invoke(initial_state(make_request()))

    ranked = final["ranked"]
    assert [item.kol.handle for item in ranked] == ["zzz_high", "aaa_low"]
    assert ranked[0].scores.engagement > ranked[1].scores.engagement
    assert ranked[0].total_score > ranked[1].total_score


def test_no_thai_kols_is_fatal():
    fixtures = [
        make_fixture("foreign", 1, text_language="en", location_created="US")
    ]
    names, final = run_stream(
        build_graph(mock_integrations(kol_fixtures=fixtures), min_candidates=1),
        make_request(),
    )

    assert final["stage"] == "failed"
    assert any("filter_thai: no Thai KOLs found" in error for error in final["errors"])
    assert "enrich_kols" not in names


def test_enricher_prefers_higher_priority_fixture_and_keeps_ties():
    apify = {
        "handle": "@Creator",
        "source": "apify",
        "engagement_rate": 0.11,
        "topics": ["from-apify"],
    }
    seed = {
        "handle": "creator",
        "source": "seed",
        "engagement_rate": 0.02,
        "topics": ["from-seed"],
    }
    kol = KolProfile(
        handle="creator",
        url="https://www.tiktok.com/@creator",
        source="cache",
    )

    for fixtures in ([apify, seed], [seed, apify]):
        update = MockKolEnricher(fixtures).enrich(kol)
        assert update["median_engagement_rate"] == 0.11
        assert "from-apify" in update["content_summary"]

    earlier = {
        "handle": "creator",
        "source": "seed",
        "engagement_rate": 0.03,
        "topics": ["earlier"],
    }
    later = {
        "handle": "CREATOR",
        "source": "seed",
        "engagement_rate": 0.09,
        "topics": ["later"],
    }
    update = MockKolEnricher([earlier, later]).enrich(kol)
    assert update["median_engagement_rate"] == 0.03
    assert "earlier" in update["content_summary"]


class _CountingEnricher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enrich(self, kol: KolProfile) -> dict:
        self.calls.append(kol.handle)
        return {}


def test_enrich_limit_caps_enrichment_to_largest_accounts():
    enricher = _CountingEnricher()
    integrations = replace(mock_integrations(), enricher=enricher)
    graph = build_graph(integrations, enrich_limit=3)

    final = graph.invoke(initial_state(make_request()))

    assert final["stage"] == "done"
    assert len(enricher.calls) == 3
    by_followers = sorted(final["candidates"], key=lambda k: -k.followers)
    assert {h.lower() for h in enricher.calls} == {
        k.handle.lower() for k in by_followers[:3]
    }
    assert len(final["candidates"]) > 3


def test_enrich_limit_default_enriches_everyone():
    enricher = _CountingEnricher()
    integrations = replace(mock_integrations(), enricher=enricher)
    final = build_graph(integrations).invoke(initial_state(make_request()))
    assert len(enricher.calls) == len(final["candidates"])


def test_analyze_brand_merges_url_aliases_after_model_names():
    class _NamedAnalyzer(MockBrandAnalyzer):
        def analyze_brand(self, website_content, facebook_page):
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


def test_llm_rerank_caps_relevance_when_the_brand_is_not_named():
    othermart_caption = (
        "พามาล่าแสตมป์ 🎁 แลกของที่ OtherMart กันครับทุกคน!! "
        "ช่วงนี้เค้ามีกิจกรรมสะสม E-Stamp แลกของได้เยอะมากกก "
        "#OtherMart #ขุมทรัพย์นักช้อป2569"
    )

    class _HighRelevanceAnalyzer(MockBrandAnalyzer):
        def rerank(self, brand, kols, *, similarities=None):
            scores = ScoreBreakdown(
                relevance=0.95,
                engagement=0.5,
                audience_fit=1.0,
                tier_fit=1.0,
                audience_confidence="high",
            )
            return [(kol, scores, "Strong fit.") for kol in kols]

    brand = MockBrandAnalyzer().analyze_brand("website", None).model_copy(
        update={"brand_names": ["ExampleBrand", "EB-24", "เอ็กแซมเปิล"]}
    )
    request = MatchRequest(
        website_url="https://www.example-brand.co.th",
        facebook_url="https://www.facebook.com/ExampleBrandThailand",
        top_k=2,
    )
    other = KolProfile(
        handle="OtherMartCreator",
        url="https://www.tiktok.com/@OtherMartCreator",
        nickname="นักช้อป",
        followers=50_000,
        language="th",
        region="TH",
        median_engagement_rate=0.05,
        content_summary=othermart_caption,
        source="seed",
    )
    example = KolProfile(
        handle="ExampleCreator",
        url="https://www.tiktok.com/@ExampleCreator",
        nickname="นักรีวิว",
        followers=50_000,
        language="th",
        region="TH",
        median_engagement_rate=0.05,
        content_summary="โปร ExampleBrand ข้าว+น้ำ 69.- #รีวิวของกิน",
        source="seed",
    )
    state = MatchState(
        request=request,
        brand_profile=brand,
        shortlisted=[other, example],
        stage="embedding_prefilter",
    )

    update = make_nodes(
        replace(mock_integrations(), llm=_HighRelevanceAnalyzer()),
        min_candidates=1,
        max_search_rounds=1,
    )["llm_rerank"](state)

    ranked = update["ranked"]
    by_handle = {item.kol.handle: item for item in ranked}
    capped = by_handle["OtherMartCreator"]
    named = by_handle["ExampleCreator"]

    assert capped.scores.relevance == 0.6
    assert capped.total_score == request.weights.apply(capped.scores)
    assert capped.reason.endswith("Brand not named in sampled posts.")
    assert named.scores.relevance == 0.95
    assert [item.kol.handle for item in ranked] == [
        "ExampleCreator",
        "OtherMartCreator",
    ]
