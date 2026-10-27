"""Offline tests for the Apify TikTok search and profile actors.

Fixtures are trimmed from the documented dataset items on the actor store pages
(CDN media URLs removed). They are not live captures.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app.integrations import ApifyConfigError, build_integrations
from app.integrations.apify import ApifyAuthError, ApifyError
from app.integrations.embeddings import MockEmbedder
from app.integrations.facebook import MockFacebookScraper
from app.integrations.kol_enricher_live import (
    ApifyKolEnricher,
    content_summary,
    default_apify_cache_dir,
    engagement_rate,
    enrich_from_items,
    posts_per_week,
)
from app.integrations.kol_search_live import (
    ApifyKolSearch,
    classify_query,
    map_search_items,
)
from app.integrations.llm import MockBrandAnalyzer
from app.integrations.web_reader import MockWebReader
from app.integrations.web_reader_live import CachedWebReader
from app.models import KolProfile, MatchRequest
from app.nodes import make_nodes
from app.state import MatchState

_TOKEN = "super-secret-token"
_SEARCH_FIXTURE = (
    Path(__file__).resolve().parents[1] / "app" / "fixtures" / "apify_search_sample.json"
)
_PROFILE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "app" / "fixtures" / "apify_profile_sample.json"
)
_BACKEND = Path(__file__).resolve().parents[1]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _search_sample() -> list[dict]:
    return json.loads(_SEARCH_FIXTURE.read_text(encoding="utf-8"))


def _profile_sample() -> list[dict]:
    return json.loads(_PROFILE_FIXTURE.read_text(encoding="utf-8"))


def _search(handler, **kwargs) -> ApifyKolSearch:
    kwargs.setdefault("max_results", 10)
    kwargs.setdefault("backoff", 0)
    kwargs.setdefault("timeout", 30)
    kwargs.setdefault("max_charge_usd", 0.5)
    return ApifyKolSearch(token=_TOKEN, client=_client(handler), **kwargs)


def _enricher(handler, **kwargs) -> ApifyKolEnricher:
    kwargs.setdefault("max_videos", 10)
    kwargs.setdefault("backoff", 0)
    kwargs.setdefault("timeout", 30)
    kwargs.setdefault("max_charge_usd", 0.5)
    return ApifyKolEnricher(token=_TOKEN, client=_client(handler), **kwargs)


def _kol(handle: str = "apifytech", followers: int = 852) -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle.lstrip('@')}",
        followers=followers,
        source="apify",
    )


def test_search_fixture_maps_authors_and_keeps_bad_rows():
    rows = map_search_items(_search_sample())

    profiles = [row for row in rows if not row.get("errorCode")]
    errors = [row for row in rows if row.get("errorCode")]

    assert [row["handle"] for row in profiles] == ["gretalynnhihi", "shaiie_foeva"]
    greta = profiles[0]
    assert greta["url"] == "https://www.tiktok.com/@gretalynnhihi"
    assert greta["followers"] == 51200
    assert greta["nickname"] == "Greta Lynn"
    assert "hellouu" in greta["bio"]
    assert "fashion" in greta["bio"]
    assert "\n" not in greta["bio"]
    assert greta["text_language"] == "en"
    # The duplicate video is the one that carries locationCreated.
    assert greta["location_created"] == "US"
    assert greta["source"] == "apify"

    shaiie = profiles[1]
    assert shaiie["followers"] == 2_200_000
    assert shaiie["bio"] == "SHOP THE E-BOOK OUT NOW"
    assert shaiie["location_created"] == "GB"
    assert "text_language" not in shaiie

    assert [row["errorCode"] for row in errors] == ["MISSING_AUTHOR", "PROFILE_PRIVATE"]


def test_flat_author_meta_keys_are_mapped():
    row = map_search_items(
        [
            {
                "authorMeta.name": "flat_user",
                "authorMeta.fans": "1,250",
                "authorMeta.signature": "line one\nline two",
                "textLanguage": "th",
                "locationCreated": "th",
            }
        ]
    )[0]

    assert row["handle"] == "flat_user"
    assert row["followers"] == 1250
    assert row["bio"] == "line one line two"
    assert row["text_language"] == "th"
    assert row["location_created"] == "TH"
    assert row["url"] == "https://www.tiktok.com/@flat_user"
    assert row["source"] == "apify"


def test_classify_query_splits_hashtags_from_keyword_search():
    assert classify_query("  #Gaming ") == ("hashtags", ["Gaming"])
    assert classify_query("#ExampleBrand tiktok") == ("hashtags", ["ExampleBrand"])
    assert classify_query("#a #b tiktok") == ("hashtags", ["a", "b"])
    assert classify_query("#fyp gaming thailand") == ("search", "#fyp gaming thailand")
    assert classify_query("blue light glasses thailand") == (
        "search",
        "blue light glasses thailand",
    )


def test_search_posts_one_run_per_query_without_putting_the_token_in_the_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=[])

    rows = _search(handler, max_results=4).search(
        ["blue light glasses", "#ExampleBrand tiktok", "  "],
        round=2,
    )

    assert rows == []
    assert len(seen) == 2
    assert all(
        request.url.path
        == "/v2/actors/clockworks~tiktok-scraper/run-sync-get-dataset-items"
        for request in seen
    )
    for request in seen:
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        assert "token" not in request.url.params
        assert _TOKEN not in str(request.url)
        assert request.url.params["maxItems"] == "4"
        assert request.url.params["limit"] == "4"
        assert request.url.params["timeout"] == "30"
        assert float(request.url.params["maxTotalChargeUsd"]) == 0.5
        assert request.url.params["restartOnError"] == "false"

    keyword = json.loads(seen[0].content)
    hashtag = json.loads(seen[1].content)
    assert keyword["searchQueries"] == ["blue light glasses"]
    assert keyword["searchSection"] == "/video"
    assert keyword["resultsPerPage"] == 4
    assert keyword["shouldDownloadVideos"] is False
    assert keyword["shouldDownloadCovers"] is False
    assert keyword["downloadSubtitlesOptions"] == "NEVER_DOWNLOAD_SUBTITLES"
    assert keyword["commentsPerPost"] == 0
    assert hashtag["hashtags"] == ["ExampleBrand"]
    assert "searchQueries" not in hashtag


def test_search_dedupes_authors_across_queries_and_records_a_failed_query():
    sample = _search_sample()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if "hashtags" in body:
            return httpx.Response(500, text=f"boom {_TOKEN}")
        return httpx.Response(201, json=sample)

    rows = _search(handler, max_retries=1).search(["ootd", "#glasses"], round=1)

    # The 500 is retried once; the keyword query is not.
    assert calls["n"] == 3
    profiles = [row for row in rows if not row.get("errorCode")]
    failed = [row for row in rows if row.get("errorCode") == "APIFY_REQUEST_FAILED"]
    assert [row["handle"] for row in profiles] == ["gretalynnhihi", "shaiie_foeva"]
    assert len(failed) == 1
    assert _TOKEN not in failed[0]["error"]
    assert "[redacted]" in failed[0]["error"]


def test_search_does_not_retry_auth_or_client_errors_and_redacts_the_token():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            401,
            json={"error": {"type": "token-not-found", "message": f"bad {_TOKEN}"}},
        )

    with pytest.raises(ApifyAuthError) as exc_info:
        _search(handler).search(["glasses"], round=1)

    assert calls["n"] == 1
    message = str(exc_info.value)
    assert "rejected the token" in message
    assert "HTTP 401" in message
    assert _TOKEN not in message
    assert isinstance(exc_info.value, ApifyError)

    calls["n"] = 0

    def bad_input(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            400,
            json={"error": {"type": "invalid-input", "message": f"nope {_TOKEN}"}},
        )

    rows = _search(bad_input).search(["glasses"], round=1)

    assert calls["n"] == 1
    assert rows[0]["errorCode"] == "APIFY_REQUEST_FAILED"
    assert _TOKEN not in rows[0]["error"]
    assert "[redacted]" in rows[0]["error"]


def test_search_retries_rate_limit_then_maps_profiles():
    calls = {"n": 0}
    sample = _search_sample()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=sample)

    rows = _search(handler).search(["ootd"], round=1)

    assert calls["n"] == 2
    assert [row["handle"] for row in rows if not row.get("errorCode")] == [
        "gretalynnhihi",
        "shaiie_foeva",
    ]


def test_search_does_not_retry_timeouts():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("timed out")

    rows = _search(handler, max_retries=2).search(["ootd"], round=1)

    assert calls["n"] == 1
    assert rows[0]["errorCode"] == "APIFY_REQUEST_FAILED"
    assert "timed out" in rows[0]["error"]
    assert _TOKEN not in rows[0]["error"]


def test_discover_records_bad_rows_from_the_search_fixture():
    sample = _search_sample()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json=sample)

    from dataclasses import replace

    from app.integrations import mock_integrations

    integrations = replace(
        mock_integrations(),
        kol_search=_search(handler),
    )
    update = make_nodes(integrations, min_candidates=1, max_search_rounds=1)[
        "discover_kols"
    ](
        MatchState(
            request=MatchRequest(
                website_url="https://example-brand.co.th",
                facebook_url="https://www.facebook.com/examplebrand",
            ),
            last_queries=["ootd"],
        )
    )

    by_handle = {kol.handle: kol for kol in update["discovered"]}
    assert set(by_handle) == {"gretalynnhihi", "shaiie_foeva"}
    assert by_handle["gretalynnhihi"].language == "en"
    assert by_handle["gretalynnhihi"].region == "US"
    assert by_handle["gretalynnhihi"].followers == 51200
    assert by_handle["gretalynnhihi"].source == "apify"
    assert by_handle["shaiie_foeva"].language is None
    assert by_handle["shaiie_foeva"].region == "GB"
    recorded = " ".join(update["errors"])
    assert "MISSING_AUTHOR" in recorded
    assert "PROFILE_PRIVATE" in recorded
    assert _TOKEN not in recorded


def test_profile_fixture_engagement_median_cadence_and_summary():
    # Unpinned videos: 0.14 (views), 0.05 (views), 852/852 followers (no views).
    # Median is 0.14. Timestamps span 14 days, so (3 - 1) / 2 weeks = 1.
    # The pinned store sample is excluded from the median and the summary.
    update = enrich_from_items(_profile_sample(), followers=0)

    assert update["median_engagement_rate"] == 0.14
    assert update["posts_per_week"] == 1.0
    assert "desk setup tour" in update["content_summary"]
    assert "#eyewear" in update["content_summary"]
    assert "Hashtags: #gadgets, #eyewear" in update["content_summary"]
    assert "programmable" not in update["content_summary"]


def test_engagement_rate_uses_views_then_followers_and_caps_the_median():
    assert engagement_rate(
        {"diggCount": 10, "commentCount": 0, "shareCount": 0, "playCount": 200},
        followers=10_000,
    ) == 0.05
    assert (
        engagement_rate(
            {"diggCount": 10, "commentCount": 5, "shareCount": 5},
            followers=100,
        )
        == 0.2
    )
    assert (
        engagement_rate(
            {"diggCount": 10, "commentCount": 0, "shareCount": 0, "playCount": 0},
            followers=100,
        )
        == 0.1
    )
    assert engagement_rate({"text": "no counts"}, followers=100) is None

    capped = enrich_from_items(
        [
            {
                "diggCount": 50,
                "commentCount": 0,
                "shareCount": 0,
                "playCount": 10,
                "text": "viral",
                "createTime": 1_700_000_000,
            }
        ],
        followers=1,
    )
    assert capped["median_engagement_rate"] == 1.0
    assert "posts_per_week" not in capped

    even = enrich_from_items(
        [
            {
                "diggCount": 10,
                "commentCount": 0,
                "shareCount": 0,
                "playCount": 100,
                "text": "a",
                "createTime": 1_700_000_000,
            },
            {
                "diggCount": 30,
                "commentCount": 0,
                "shareCount": 0,
                "playCount": 100,
                "text": "b",
                "createTime": 1_700_000_000 + 7 * 24 * 60 * 60,
            },
        ],
        followers=1,
    )
    assert even["median_engagement_rate"] == 0.2
    assert even["posts_per_week"] == 1.0


def test_posts_per_week_reads_iso_timestamps_and_summary_is_bounded():
    assert posts_per_week([{"createTime": 10}]) is None
    assert (
        posts_per_week(
            [
                {"createTimeISO": "2024-01-01T00:00:00.000Z"},
                {"createTimeISO": "2024-01-15T00:00:00.000Z"},
            ]
        )
        == 0.5
    )

    summary = content_summary(
        [{"text": "b" * 200, "hashtags": [{"name": f"tag{i}"}]} for i in range(5)]
    )
    assert summary is not None
    assert len(summary) <= 400
    assert summary.startswith("Recent posts:")


def test_enricher_request_and_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert (
            request.url.path
            == "/v2/actors/clockworks~tiktok-profile-scraper/run-sync-get-dataset-items"
        )
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        assert _TOKEN not in str(request.url)
        body = json.loads(request.content)
        assert body["profiles"] == ["apifytech"]
        assert body["resultsPerPage"] == 4
        assert body["profileSorting"] == "latest"
        assert body["excludePinnedPosts"] is True
        assert body["shouldDownloadVideos"] is False
        assert body["commentsPerPost"] == 0
        payload = _profile_sample()
        payload[1] = {**payload[1], "text": "แว่นกันแสง #gadgets"}
        return httpx.Response(201, json=payload)

    enricher = _enricher(handler, max_videos=4, cache_dir=tmp_path, clock=lambda: 1_000.0)
    first = enricher.enrich(_kol("@ApifyTech"))
    second = enricher.enrich(_kol("apifytech"))

    assert calls["n"] == 1
    assert second == first
    assert first["median_engagement_rate"] == 0.14
    assert "แว่นกันแสง" in first["content_summary"]
    raw = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "แว่นกันแสง" in raw
    cached = json.loads(raw)
    assert cached["handle"] == "apifytech"
    assert cached["fetched_at"] == 1_000.0
    assert cached["update"]["posts_per_week"] == 1.0


def test_enricher_cache_expiry_and_corrupt_file(tmp_path):
    now = {"t": 1_000.0}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        videos = [
            {
                "text": f"version {calls['n']}",
                "diggCount": 10,
                "commentCount": 0,
                "shareCount": 0,
                "playCount": 100,
                "createTime": 1_700_000_000,
                "isPinned": False,
            }
        ]
        return httpx.Response(201, json=videos)

    enricher = _enricher(
        handler,
        cache_dir=tmp_path,
        ttl_seconds=10,
        clock=lambda: now["t"],
    )
    first = enricher.enrich(_kol())
    now["t"] = 1_010.0
    assert enricher.enrich(_kol()) == first
    assert calls["n"] == 1

    now["t"] = 1_010.1
    refreshed = enricher.enrich(_kol())
    assert calls["n"] == 2
    assert refreshed["content_summary"] != first["content_summary"]

    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
    enricher.enrich(_kol())
    assert calls["n"] == 3


def test_private_profile_is_not_cached(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            201,
            json=[
                {
                    "url": "https://www.tiktok.com/@apifytech",
                    "input": "apifytech",
                    "error": "Profile is private",
                    "errorCode": "PROFILE_PRIVATE",
                }
            ],
        )

    enricher = _enricher(handler, cache_dir=tmp_path)
    with pytest.raises(ApifyError, match="PROFILE_PRIVATE"):
        enricher.enrich(_kol())

    assert calls["n"] == 1
    assert list(tmp_path.glob("*.json")) == []


def test_missing_token_raises_a_config_error(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    with pytest.raises(ApifyConfigError, match="APIFY_TOKEN") as exc_info:
        ApifyKolSearch()
    assert _TOKEN not in str(exc_info.value)

    with pytest.raises(ApifyConfigError):
        build_integrations(live_kols=True, apify_token="  ")


def test_live_kols_factory_uses_apify_and_keeps_other_services_mocked(monkeypatch, tmp_path):
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    bundle = build_integrations(
        live_kols=True,
        apify_token=_TOKEN,
        max_results_per_query=4,
    )
    both = build_integrations(
        live_web=True,
        live_kols=True,
        apify_token=_TOKEN,
    )

    assert isinstance(bundle.web_reader, MockWebReader)
    assert isinstance(bundle.kol_search, ApifyKolSearch)
    assert isinstance(bundle.enricher, ApifyKolEnricher)
    assert bundle.kol_search.max_results == 4
    assert bundle.enricher.max_videos == 4
    assert isinstance(bundle.facebook, MockFacebookScraper)
    assert isinstance(bundle.llm, MockBrandAnalyzer)
    assert isinstance(bundle.embedder, MockEmbedder)
    assert "MOCK" not in (bundle.kol_search.__class__.__doc__ or "")
    assert isinstance(both.web_reader, CachedWebReader)
    assert isinstance(both.kol_search, ApifyKolSearch)
    bundle.kol_search._apify._client.close()
    bundle.enricher._apify._client.close()
    both.kol_search._apify._client.close()
    both.enricher._apify._client.close()


def test_default_apify_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("KOL_CACHE_DIR", raising=False)
    cache = default_apify_cache_dir()
    repo_root = cache.parents[2]
    assert cache == repo_root / "data" / "cache" / "apify"
    assert (repo_root / "backend" / "pyproject.toml").is_file()

    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    assert default_apify_cache_dir() == tmp_path / "apify"


def test_smoke_script_exits_2_without_a_token(monkeypatch):
    env = os.environ.copy()
    env.pop("APIFY_TOKEN", None)
    result = subprocess.run(
        [sys.executable, str(_BACKEND / "scripts" / "apify_smoke.py")],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "APIFY_TOKEN" in result.stderr
    assert result.stdout == ""
