from dataclasses import replace

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from app.api.settings import Settings
from app.cli import main
from app.graph import build_graph, initial_state
from app.integrations import build_integrations, mock_integrations
from app.integrations.brand_research import (
    BrandResearch,
    MockBrandResearcher,
    WebSearchConfigError,
    WebSearchError,
)
from app.integrations.brand_research_live import (
    JinaWebSearch,
    LiveBrandResearcher,
    SearchHit,
    build_research_agent,
)
from app.integrations.llm import MockBrandAnalyzer
from app.integrations.llm_live import LlmSettings
from app.models import MatchRequest
from app.prompts import brand_messages


class _ToolModel(GenericFakeChatModel):
    """A scripted model that records each call's messages and bound tools."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self.bind(bound_tools=[tool.name for tool in tools])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append((messages, kwargs.pop("bound_tools", None)))
        return super()._generate(messages, stop, run_manager, **kwargs)


def _model(*replies: AIMessage) -> _ToolModel:
    return _ToolModel(messages=iter(replies), calls=[])


def _search_call(*queries: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": query}, "id": f"call-{query}"}
            for query in queries
        ],
    )


class _Search:
    def __init__(self, hits=None, error=None):
        self.queries: list[str] = []
        self._hits = (
            [SearchHit("Rival", "https://example.com/rival", "A rival brand")]
            if hits is None
            else hits
        )
        self._error = error

    def search(self, query):
        self.queries.append(query)
        if self._error:
            raise self._error
        return self._hits


def _researcher(model, search, **kwargs) -> LiveBrandResearcher:
    return LiveBrandResearcher(
        LlmSettings(api_key="secret"), model=model, search=search, **kwargs
    )


def _research(researcher: LiveBrandResearcher) -> BrandResearch:
    return researcher.research("https://brand.example", "We sell glasses.", None)


def test_agent_searches_then_writes_notes():
    search = _Search()
    model = _model(
        _search_call("glasses rivals"),
        AIMessage(content="done"),
        AIMessage(content="Rival is a rival."),
    )

    research = _research(_researcher(model, search))

    assert search.queries == ["glasses rivals"]
    assert research == BrandResearch("Rival is a rival.", searches=1, llm_calls=3)


def test_search_results_go_back_to_the_model_in_the_loop():
    model = _model(_search_call("q"), AIMessage(content="done"), AIMessage(content="notes"))

    final = build_research_agent(model, _Search()).invoke(
        {"messages": [{"role": "user", "content": "go"}]}
    )

    tool_message = next(m for m in final["messages"] if isinstance(m, ToolMessage))
    assert "https://example.com/rival" in tool_message.content
    assert tool_message.artifact == ["https://example.com/rival"]
    assert tool_message in model.calls[1][0]


def test_notes_step_has_no_tools_and_no_tool_messages():
    model = _model(_search_call("q"), AIMessage(content="done"), AIMessage(content="notes"))

    _research(_researcher(model, _Search()))

    assert [bound for _, bound in model.calls] == [["web_search"], ["web_search"], None]
    messages = model.calls[-1][0]
    assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", None) for m in messages)
    assert "https://example.com/rival" in messages[-1].text
    assert "We sell glasses." in messages[-1].text


def test_extra_tool_calls_in_one_reply_are_dropped_at_the_budget():
    search = _Search()
    model = _model(_search_call("a", "b", "c"), AIMessage(content="notes"))

    research = _research(_researcher(model, search, max_searches=2))

    assert search.queries == ["a", "b"]
    assert research == BrandResearch("notes", searches=2, llm_calls=2)


def test_loop_ends_when_the_budget_is_spent():
    search = _Search()
    model = _model(_search_call("a"), _search_call("b"), AIMessage(content="notes"))

    research = _research(_researcher(model, search, max_searches=2))

    assert search.queries == ["a", "b"]
    assert [bound for _, bound in model.calls] == [["web_search"], ["web_search"], None]
    assert research == BrandResearch("notes", searches=2, llm_calls=3)


def test_answer_without_any_search_is_not_used_as_notes():
    model = _model(AIMessage(content="From memory: X is big."))

    assert _research(_researcher(model, _Search())) == BrandResearch(
        None, searches=0, llm_calls=1
    )


def test_failed_searches_go_back_to_the_model_and_give_no_notes():
    search = _Search(error=WebSearchError("Jina search failed: HTTP 402"))
    model = _model(_search_call("a"), AIMessage(content="X is big, I recall."))

    research = _research(_researcher(model, search))

    assert search.queries == ["a"]
    assert "HTTP 402" in model.calls[1][0][-1].text
    assert research == BrandResearch(None, searches=1, llm_calls=2)


def test_empty_results_give_no_notes():
    model = _model(_search_call("a"), AIMessage(content="done"))

    assert _research(_researcher(model, _Search(hits=[]))).notes is None


def test_unexpected_tool_bug_is_not_hidden_from_the_caller():
    search = _Search(error=TypeError("bug"))
    with pytest.raises(TypeError, match="bug"):
        _research(_researcher(_model(_search_call("a")), search))


def _jina(handler) -> JinaWebSearch:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JinaWebSearch(api_key="jina-key", client=client, backoff=0)


def test_jina_search_sends_the_query_and_maps_rows():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"title": "T", "url": "https://a.example", "description": "S"},
                    {"title": "no url"},
                ]
            },
        )

    hits = _jina(handler).search("แว่น คู่แข่ง")

    assert hits == [SearchHit("T", "https://a.example", "S")]
    assert seen[0].url.params["q"] == "แว่น คู่แข่ง"
    assert seen[0].headers["authorization"] == "Bearer jina-key"
    assert seen[0].headers["x-respond-with"] == "no-content"


def test_jina_search_retries_transient_status_then_succeeds():
    responses = [httpx.Response(503), httpx.Response(200, json={"data": []})]

    assert _jina(lambda request: responses.pop(0)).search("q") == []
    assert responses == []


def test_jina_search_does_not_retry_a_payment_error():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(402)

    with pytest.raises(WebSearchError, match="HTTP 402"):
        _jina(handler).search("q")
    assert len(calls) == 1


def test_jina_search_needs_a_key(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with pytest.raises(WebSearchConfigError, match="JINA_API_KEY"):
        JinaWebSearch()


def test_live_search_flag_builds_the_live_researcher(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "jina-key")
    bundle = build_integrations(live_search=True, llm_api_key="llm-key")
    assert isinstance(bundle.researcher, LiveBrandResearcher)


def test_default_bundle_does_not_search():
    assert isinstance(mock_integrations().researcher, MockBrandResearcher)


def test_brand_messages_add_notes_only_when_present():
    without = brand_messages("site", None)[1]["content"]
    with_notes = brand_messages("site", None, "Rival sells more.")[1]["content"]

    assert "Web research notes" not in without
    assert "Web research notes\nRival sells more." in with_notes


class _SpyAnalyzer(MockBrandAnalyzer):
    def __init__(self):
        self.research = "unset"

    def analyze_brand(self, website_content, facebook_page, research=None):
        self.research = research
        return super().analyze_brand(website_content, facebook_page)


class _FixedResearcher:
    def __init__(self, result=None, error=None):
        self.calls = 0
        self._result = result
        self._error = error

    def research(self, website_url, website_content, facebook_page):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


def _run(researcher, **replaced):
    analyzer = _SpyAnalyzer()
    integrations = replace(
        mock_integrations(), llm=analyzer, researcher=researcher, **replaced
    )
    request = MatchRequest(
        website_url="https://brand.example",
        facebook_url="https://facebook.com/brand",
    )
    return analyzer, build_graph(integrations).invoke(initial_state(request))


def test_graph_gives_notes_to_the_brand_analysis():
    analyzer, final = _run(
        _FixedResearcher(BrandResearch("Rival sells more.", searches=2, llm_calls=3))
    )

    assert analyzer.research == "Rival sells more."
    assert final["brand_sources"] == [
        "website: used",
        "facebook: used",
        "web research: used",
    ]
    assert "llm: research_brand 3 calls" in final["cost_log"]
    assert "web search: 2 queries" in final["cost_log"]
    assert final["stage"] == "done"


def test_research_failure_is_a_warning_and_the_search_continues():
    analyzer, final = _run(_FixedResearcher(error=RuntimeError("provider down")))

    assert analyzer.research is None
    assert "research_brand: provider down" in final["errors"]
    assert final["stage"] == "done"


def test_no_web_evidence_is_a_warning():
    analyzer, final = _run(_FixedResearcher(BrandResearch(None, searches=1, llm_calls=2)))

    assert analyzer.research is None
    assert "research_brand: warning: no web evidence found" in final["errors"]
    assert final["stage"] == "done"


def test_mock_research_adds_no_warning_or_cost():
    _, final = _run(MockBrandResearcher())

    assert not any("research_brand" in line for line in final["errors"])
    assert not any("research_brand" in line for line in final["cost_log"])


def test_research_is_skipped_when_no_brand_source_was_read():
    class _Fails:
        def read(self, url):
            raise RuntimeError("blocked")

        def fetch_page(self, url):
            raise RuntimeError("blocked")

    researcher = _FixedResearcher(BrandResearch("unused"))
    _, final = _run(researcher, web_reader=_Fails(), facebook=_Fails())

    assert researcher.calls == 0
    assert final["stage"] == "failed"
    assert any(line.startswith("analyze_brand: no usable") for line in final["errors"])


def test_live_search_is_unavailable_without_either_key():
    both = Settings(demo_mode=False, llm_api_key="k", jina_api_key="j", _env_file=None)
    no_jina = Settings(demo_mode=False, llm_api_key="k", jina_api_key="", _env_file=None)
    no_llm = Settings(
        demo_mode=False,
        llm_api_key="",
        nvidia_api_key="",
        openai_api_key="",
        jina_api_key="j",
        _env_file=None,
    )

    assert both.default_flags().live_search is True
    assert no_jina.unavailable()["live_search"] == "JINA_API_KEY is not set"
    assert "LLM_API_KEY" in no_llm.unavailable()["live_search"]
    assert no_jina.default_flags().live_search is False


def test_cli_live_search_without_jina_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "k")

    code = main(
        [
            "match",
            "--website",
            "https://brand.example",
            "--facebook",
            "https://facebook.com/brand",
            "--live-search",
        ]
    )

    assert code == 2
    assert "--live-search requires JINA_API_KEY" in capsys.readouterr().err
