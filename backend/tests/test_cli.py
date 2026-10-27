import json
from dataclasses import replace

from langgraph.errors import GraphRecursionError

from app.cli import main
from app.graph import build_graph
from app.integrations import mock_integrations

_WEBSITE = "https://ophtus.com"
_FACEBOOK = "https://www.facebook.com/ophtus"


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
