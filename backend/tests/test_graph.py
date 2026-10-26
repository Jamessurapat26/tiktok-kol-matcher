import json
from dataclasses import replace
from pathlib import Path

from app.graph import build_graph, initial_state
from app.integrations import mock_integrations
from app.models import MatchRequest

_KOL_FIXTURE = Path(__file__).resolve().parents[1] / "app" / "fixtures" / "kols.json"


def make_request(top_k: int = 10) -> MatchRequest:
    return MatchRequest(
        website_url="https://ophtus.com",
        facebook_url="https://www.facebook.com/ophtus",
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

    assert final["search_rounds"] == 2
    assert names.count("generate_queries") == 2
    assert final["stage"] == "done"

    queries = final["search_queries"]
    assert "blue light glasses thailand" in queries  # round 1
    assert "#Ophtus tiktok" in queries  # round 2
    normalized = [query.strip().casefold() for query in queries]
    assert len(set(normalized)) == len(normalized)


def test_loop_stops_early_when_enough_candidates():
    final = build_graph(min_candidates=5).invoke(initial_state(make_request()))

    assert final["search_rounds"] == 1
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
    graph = build_graph(prefilter_n=5)

    first = graph.invoke(initial_state(make_request()))
    second = graph.invoke(initial_state(make_request()))

    assert len(first["shortlisted"]) == 5
    assert len(first["ranked"]) == 5
    assert [item.kol.handle for item in first["results"]] == [
        item.kol.handle for item in second["results"]
    ]


def test_failing_integration_is_reported_not_raised():
    class BoomReader:
        def read(self, url: str) -> str:
            raise RuntimeError("boom")

    integrations = replace(mock_integrations(), web_reader=BoomReader())
    final = build_graph(integrations).invoke(initial_state(make_request()))

    assert "scrape_website: boom" in final["errors"]
