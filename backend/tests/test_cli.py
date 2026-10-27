import json
import sys
from dataclasses import replace

import pytest
from langgraph.errors import GraphRecursionError

from app.cli import main
from app.graph import build_graph
from app.integrations import mock_integrations
from app.integrations.kol_enricher_live import ApifyKolEnricher
from app.integrations.kol_search_live import ApifyKolSearch
from app.integrations.web_reader import MockWebReader
from app.integrations.web_reader_live import CachedWebReader

_WEBSITE = "https://example-brand.co.th"
_FACEBOOK = "https://www.facebook.com/examplebrand"


def test_json_output(capsys):
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--top-k",
            "5",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["stage"] == "done"
    assert len(payload["results"]) == 5
    assert {"discovered", "candidates", "shortlisted", "ranked"} <= set(
        payload["counts"]
    )


def test_human_output_prints_table_and_progress(capsys):
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "rank" in captured.out
    assert "[ 1]" in captured.err
    assert "scrape_" in captured.err
    # Merged unique totals, not the raw row count from the node update.
    assert "discovered=22 (+22 new)" in captured.err
    assert "discovered=36 (+14 new)" in captured.err
    assert "discovered+" not in captured.err
    assert "@" not in captured.out


def test_invalid_url_returns_exit_code_2(capsys):
    exit_code = main(
        [
            "match",
            "--website",
            "not-a-url",
            "--facebook",
            _FACEBOOK,
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err


def test_negative_max_rounds_returns_exit_code_2(monkeypatch, capsys):
    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)

    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--max-rounds",
            "-1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--max-rounds" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_search_failure_exits_1_without_traceback(monkeypatch, capsys):
    class BoomSearch:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, queries, round):
            self.calls += 1
            raise RuntimeError("search down")

    search = BoomSearch()
    integrations = replace(mock_integrations(), kol_search=search)

    monkeypatch.setattr(
        "app.cli.build_graph",
        lambda *args, **kwargs: build_graph(integrations, *args, **kwargs),
    )

    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert search.calls == 1
    assert "discover_kols: search failed: search down" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_website_scrape_failure_exits_1(monkeypatch, capsys):
    class BoomReader:
        def read(self, url: str) -> str:
            raise RuntimeError("boom")

    integrations = replace(mock_integrations(), web_reader=BoomReader())
    monkeypatch.setattr(
        "app.cli.build_graph",
        lambda *args, **kwargs: build_graph(integrations, *args, **kwargs),
    )

    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scrape_website: boom" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_graph_recursion_error_exits_1_without_traceback(monkeypatch, capsys):
    class BoomGraph:
        def stream(self, *args, **kwargs):
            raise GraphRecursionError(
                "recursion limit of 25 reached without hitting a stop condition"
            )

    monkeypatch.setattr("app.cli.build_graph", lambda *args, **kwargs: BoomGraph())

    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: pipeline exceeded the recursion limit" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_help_mentions_live_web(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["match", "--help"])

    assert exc.value.code == 0
    assert "--live-web" in capsys.readouterr().out


def test_live_web_flag_uses_cached_reader_without_running_it(monkeypatch, capsys):
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["integrations"] = integrations
        captured["kwargs"] = kwargs
        captured["kwargs"] = kwargs

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter(
                    [("values", {"stage": "done", "results": [], "errors": []})]
                )

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--top-k",
            "5",
            "--json",
            "--live-web",
        ]
    )

    assert exit_code == 0
    assert isinstance(captured["integrations"].web_reader, CachedWebReader)
    assert isinstance(captured["integrations"].kol_search, type(mock_integrations().kol_search))
    assert captured["kwargs"]["min_candidates"]
    assert json.loads(capsys.readouterr().out)["stage"] == "done"


def test_help_mentions_live_kols(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["match", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--live-kols" in help_text
    assert "--max-results-per-query" in help_text
    assert "APIFY_TOKEN" in help_text


def test_live_kols_without_token_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--live-kols",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "APIFY_TOKEN" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""

    monkeypatch.setenv("APIFY_TOKEN", "   ")
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--live-kols",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "APIFY_TOKEN" in captured.err


def test_live_kols_rejects_non_positive_max_results(monkeypatch, capsys):
    monkeypatch.setenv("APIFY_TOKEN", "cli-test-token")

    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--live-kols",
            "--max-results-per-query",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--max-results-per-query" in captured.err
    assert "cli-test-token" not in captured.err
    assert "cli-test-token" not in captured.out


def test_live_kols_flag_wires_apify_without_running_it(monkeypatch, capsys):
    monkeypatch.setenv("APIFY_TOKEN", "cli-test-token")
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["integrations"] = integrations
        captured["kwargs"] = kwargs

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter(
                    [("values", {"stage": "done", "results": [], "errors": []})]
                )

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--top-k",
            "5",
            "--json",
            "--live-kols",
            "--max-results-per-query",
            "4",
            "--max-enrich",
            "3",
        ]
    )

    captured_io = capsys.readouterr()
    assert exit_code == 0
    assert isinstance(captured["integrations"].web_reader, MockWebReader)
    assert isinstance(captured["integrations"].kol_search, ApifyKolSearch)
    assert isinstance(captured["integrations"].enricher, ApifyKolEnricher)
    assert captured["integrations"].kol_search.max_results == 4
    assert captured["integrations"].enricher.max_videos == 4
    assert captured["kwargs"]["enrich_limit"] == 3
    assert "cli-test-token" not in captured_io.out
    assert "cli-test-token" not in captured_io.err
    assert json.loads(captured_io.out)["stage"] == "done"


def test_max_enrich_must_be_positive(capsys):
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--max-enrich",
            "0",
        ]
    )
    assert exit_code == 2
    assert "--max-enrich" in capsys.readouterr().err


def test_help_mentions_live_embed(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["match", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--live-embed" in help_text
    assert "--prefilter-limit" in help_text
    assert "google/embeddinggemma-2" in help_text
    assert "Ollama" not in help_text
    assert "OLLAMA" not in help_text


def test_prefilter_limit_validation(monkeypatch, capsys):
    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--prefilter-limit",
            "0",
        ]
    )
    assert exit_code == 2
    assert "--prefilter-limit" in capsys.readouterr().err

    monkeypatch.setenv("PREFILTER_LIMIT", "nope")
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
        ]
    )
    assert exit_code == 2
    assert "PREFILTER_LIMIT" in capsys.readouterr().err


def test_prefilter_limit_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("PREFILTER_LIMIT", "12")
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["kwargs"] = kwargs

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter([("values", {"stage": "done", "results": [], "errors": []})])

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--json",
            "--prefilter-limit",
            "8",
            "--top-k",
            "40",
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["prefilter_n"] == 8

    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--json",
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["prefilter_n"] == 12


def test_live_embed_flag_does_not_load_the_model(monkeypatch, capsys):
    from app.integrations.embeddings_live import SentenceTransformerEmbedder
    from app.integrations.llm import MockBrandAnalyzer

    for name in ("EMBED_MODEL", "EMBED_DEVICE", "EMBED_DIM", "EMBED_DTYPE"):
        monkeypatch.delenv(name, raising=False)
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["integrations"] = integrations
        captured["kwargs"] = kwargs

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter([("values", {"stage": "done", "results": [], "errors": []})])

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--top-k",
            "5",
            "--json",
            "--live-embed",
        ]
    )
    captured_io = capsys.readouterr()
    embedder = captured["integrations"].embedder
    assert exit_code == 0
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert isinstance(captured["integrations"].llm, MockBrandAnalyzer)
    assert isinstance(captured["integrations"].web_reader, MockWebReader)
    assert embedder.model_name == "google/embeddinggemma-2"
    assert embedder.device == "cpu"
    assert embedder.dim == 768
    assert embedder._model is None
    assert captured["kwargs"]["prefilter_n"] == 25
    assert "OLLAMA" not in captured_io.out
    assert "OLLAMA" not in captured_io.err


def test_live_embed_without_the_extra_exits_2_before_any_integration(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    def fail(*args, **kwargs):
        raise AssertionError("integrations and graph should not be built")

    monkeypatch.setattr("app.cli.build_integrations", fail)
    monkeypatch.setattr("app.cli.build_graph", fail)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--live-embed",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "uv sync --extra embed" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_bad_embed_dim_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("EMBED_DIM", "300")

    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            _WEBSITE,
            "--facebook",
            _FACEBOOK,
            "--live-embed",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "EMBED_DIM" in captured.err
    assert "Traceback" not in captured.err
