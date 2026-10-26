import json

from app.cli import main

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
