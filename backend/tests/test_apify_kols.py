"""Offline tests for the Apify TikTok search and profile actors.

Fixtures are trimmed from the documented dataset items on the actor store pages
(CDN media URLs removed). They are not live captures.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.integrations import ApifyConfigError, build_integrations
from app.integrations.apify import ApifyAuthError, ApifyError
from app.integrations.embeddings import MockEmbedder
from app.integrations.facebook import MockFacebookScraper
from app.integrations.kol_enricher import MockKolEnricher
from app.integrations.kol_enricher_live import (
    ApifyKolEnricher,
    content_summary,
    default_apify_cache_dir,
    enrich_from_items,
    posts_per_week,
)
from app.integrations.kol_search import MockKolSearch
from app.integrations.kol_search_live import (
    ApifyKolSearch,
    classify_query,
    map_search_item,
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


def test_search_fixture_keeps_per_video_stats_and_concatenates_per_author():
    rows = map_search_items(_search_sample())

    greta = next(row for row in rows if row.get("handle") == "gretalynnhihi")
    assert [video["url"] for video in greta["videos"]] == [
        "https://www.tiktok.com/@gretalynnhihi/video/7543693751290481942",
        "https://www.tiktok.com/@gretalynnhihi/video/7543693751290481999",
    ]
    first, second = greta["videos"]
    assert (first["views"], first["likes"], first["comments"], first["shares"]) == (
        145900,
        23400,
        46,
        145,
    )
    assert first["caption"] == "ootd #foruyou #outfit"
    assert second["caption"] == "second look #outfit"
    assert datetime.fromisoformat(first["posted_at"]) == datetime(
        2025, 8, 28, 17, 44, 35, tzinfo=UTC
    )
    assert datetime.fromisoformat(first["posted_at"]).tzinfo is not None

    shaiie = next(row for row in rows if row.get("handle") == "shaiie_foeva")
    assert len(shaiie["videos"]) == 1
    assert shaiie["videos"][0]["caption"] == ""


def test_map_search_item_builds_video_url_and_defaults_missing_counts():
    row = map_search_item(
        {
            "id": 7543693751290481999,
            "text": "  late night  gaming\n#eyewear ",
            "createTimeISO": "2025-08-28T16:53:20.000Z",
            "authorMeta": {"name": "demo_user", "fans": 10},
        }
    )

    video = row["videos"][0]
    assert video["url"] == (
        "https://www.tiktok.com/@demo_user/video/7543693751290481999"
    )
    assert (video["views"], video["likes"], video["comments"], video["shares"]) == (
        0,
        0,
        0,
        0,
    )
    assert video["caption"] == "late night gaming #eyewear"
    assert datetime.fromisoformat(video["posted_at"]).tzinfo is not None


def test_map_search_items_dedupes_videos_by_url():
    item = {
        "id": "1234567890123456789",
        "text": "same video",
        "createTime": 1_756_403_075,
        "webVideoUrl": "https://www.tiktok.com/@repeat/video/1234567890123456789",
        "authorMeta": {"name": "repeat", "fans": 100},
        "diggCount": 10,
        "commentCount": 1,
        "shareCount": 1,
        "playCount": 100,
    }

    row = map_search_items([item, item])[0]
    assert len(row["videos"]) == 1


def test_search_captions_are_kept_joined_and_capped():
    rows = map_search_items(_search_sample())
    greta = next(row for row in rows if row.get("handle") == "gretalynnhihi")
    shaiie = next(row for row in rows if row.get("handle") == "shaiie_foeva")

    assert greta["content_summary"] == "ootd #foruyou #outfit | second look #outfit"
    assert "content_summary" not in shaiie

    def item(handle: str, text: str) -> dict:
        return {"authorMeta": {"name": handle, "fans": 10}, "text": text}

    single = map_search_items([item("long_caption", "a" * 400)])[0]
    assert single["content_summary"] == "a" * 300

    joined = map_search_items(
        [
            item("caption_kol", "a" * 400),
            item("caption_kol", "b" * 400),
            item("caption_kol", "c" * 400),
        ]
    )[0]["content_summary"]
    assert joined == " | ".join(["a" * 300, "b" * 300, "c" * 300])[:500]
    assert len(joined) == 500

    distinct = map_search_items(
        [
            item("duplicate_captions", " same "),
            item("duplicate_captions", "same"),
            item("duplicate_captions", "next"),
            item("duplicate_captions", "third"),
            item("duplicate_captions", "fourth"),
        ]
    )[0]["content_summary"]
    assert distinct == "same | next | third"


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


def test_profile_fixture_cadence_and_summary():
    # Timestamps span 14 days, so (3 - 1) / 2 weeks = 1. The pinned store
    # sample is excluded from the summary.
    update = enrich_from_items(_profile_sample(), followers=0)

    assert "median_engagement_rate" not in update
    assert update["posts_per_week"] == 1.0
    assert "desk setup tour" in update["content_summary"]
    assert "#eyewear" in update["content_summary"]
    assert "Hashtags: #gadgets, #eyewear" in update["content_summary"]
    assert "programmable" not in update["content_summary"]
    assert update["stats_source"] == "profile"
    assert len(update["videos"]) == 3
    assert update["videos"][0]["url"] == (
        "https://www.tiktok.com/@apifytech/video/7353646097262202201"
    )
    assert update["videos"][0]["caption"] == "desk setup tour #gadgets"
    assert update["videos"][0]["views"] == 1000
    assert datetime.fromisoformat(update["videos"][0]["posted_at"]).tzinfo is not None


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
    assert "median_engagement_rate" not in first
    assert first["stats_source"] == "profile"
    assert first["videos"]
    assert "แว่นกันแสง" in first["content_summary"]
    raw = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "แว่นกันแสง" in raw
    cached = json.loads(raw)
    assert cached["handle"] == "apifytech"
    assert cached["fetched_at"] == 1_000.0
    assert cached["update"]["posts_per_week"] == 1.0
    assert cached["update"]["stats_source"] == "profile"
    assert cached["update"]["videos"]


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
    for name in ("TIKTOK_RESULTS_PER_QUERY", "TIKTOK_PROFILE_VIDEOS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TIKTOK_PROFILE_VIDEOS", "6")
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
    assert bundle.enricher.max_videos == 6
    assert both.kol_search.max_results == 15
    assert both.enricher.max_videos == 6
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


def test_live_sizes_come_from_env_in_the_actor_inputs(monkeypatch, tmp_path):
    for name in (
        "TIKTOK_RESULTS_PER_QUERY",
        "TIKTOK_PROFILE_VIDEOS",
        "APIFY_ENRICH_RESERVE_USD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIKTOK_RESULTS_PER_QUERY", "7")
    monkeypatch.setenv("TIKTOK_PROFILE_VIDEOS", "3")
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    seen: list[tuple[str, object, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (request.url.path, request.url.params, json.loads(request.content))
        )
        if "profile-scraper" in request.url.path:
            return httpx.Response(201, json=_profile_sample())
        return httpx.Response(201, json=_search_sample())

    bundle = build_integrations(live_kols=True, apify_token=_TOKEN)
    bundle.kol_search._apify._client.close()
    bundle.kol_search._apify._client = _client(handler)
    bundle.enricher._apify._client.close()
    bundle.enricher._apify._client = _client(handler)

    rows = bundle.kol_search.search(["ootd"], round=1)
    bundle.enricher.enrich(_kol("apifytech"))

    search_path, search_params, search_body = seen[0]
    profile_path, profile_params, profile_body = seen[1]
    assert "clockworks~tiktok-scraper" in search_path
    assert search_body["resultsPerPage"] == 7
    assert search_params["maxItems"] == "7"
    assert "clockworks~tiktok-profile-scraper" in profile_path
    assert profile_body["resultsPerPage"] == 3
    assert profile_params["maxItems"] == "3"
    assert any(row.get("handle") == "gretalynnhihi" for row in rows)
    assert bundle.kol_search._apify._reserve_usd == pytest.approx(0.70)

    bundle.kol_search._apify._client.close()
    bundle.enricher._apify._client.close()


def test_default_apify_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("KOL_CACHE_DIR", raising=False)
    cache = default_apify_cache_dir()
    repo_root = cache.parents[2]
    assert cache == repo_root / "data" / "cache" / "apify"
    assert (repo_root / "backend" / "pyproject.toml").is_file()

    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    assert default_apify_cache_dir() == tmp_path / "apify"


def test_mock_search_converts_days_ago_to_recent_utc_posted_at():
    fixtures = [
        {
            "handle": "demo_creator",
            "url": "https://www.tiktok.com/@demo_creator",
            "nickname": "Demo",
            "followers": 1000,
            "bio": "demo",
            "source": "apify",
            "text_language": "th",
            "location_created": "TH",
            "query_group": 1,
            "videos": [
                {
                    "url": "https://www.tiktok.com/@demo_creator/video/1234567890123456789",
                    "views": 100,
                    "likes": 10,
                    "comments": 1,
                    "shares": 2,
                    "days_ago": 3,
                    "caption": "demo #gaming",
                }
            ],
        }
    ]

    rows = MockKolSearch(fixtures).search(["demo"], round=1)

    video = rows[0]["videos"][0]
    assert "days_ago" not in video
    posted = datetime.fromisoformat(video["posted_at"])
    assert posted.tzinfo is not None
    assert 2 <= (datetime.now(UTC) - posted).total_seconds() / 86400 <= 4
    assert MockKolSearch(fixtures).search(["demo"], round=2) == []


def test_mock_enricher_returns_profile_videos_and_stats_source():
    fixtures = [
        {
            "handle": "demo_creator",
            "source": "apify",
            "profile_videos": [
                {
                    "url": "https://www.tiktok.com/@demo_creator/video/1",
                    "views": 100,
                    "likes": 5,
                    "comments": 1,
                    "shares": 1,
                    "days_ago": 2,
                    "caption": "first profile caption #gaming",
                },
                {
                    "url": "https://www.tiktok.com/@demo_creator/video/2",
                    "views": 200,
                    "likes": 20,
                    "comments": 2,
                    "shares": 2,
                    "days_ago": 16,
                    "caption": "second profile caption #eyewear",
                },
            ],
        }
    ]
    kol = _kol("demo_creator")

    update = MockKolEnricher(fixtures).enrich(kol)

    assert update["stats_source"] == "profile"
    assert len(update["videos"]) == 2
    assert update["posts_per_week"] == 0.5
    assert update["content_summary"].startswith("Recent posts:")
    assert "first profile caption" in update["content_summary"]
    assert "median_engagement_rate" not in update


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


def _avatar_item(**author) -> dict:
    return {
        "id": 7543693751290481001,
        "text": "gaming glasses",
        "createTimeISO": "2026-10-20T10:00:00.000Z",
        "authorMeta": {"name": "avatar_user", "fans": 10, **author},
    }


def test_map_search_item_prefers_the_original_avatar_url():
    row = map_search_item(
        _avatar_item(
            avatar="https://example.com/small.jpg",
            originalAvatarUrl="https://example.com/a.jpg",
        )
    )

    assert row["avatar_url"] == "https://example.com/a.jpg"


def test_map_search_item_falls_back_to_the_avatar_field():
    row = map_search_item(_avatar_item(avatar="https://example.com/a.jpg"))

    assert row["avatar_url"] == "https://example.com/a.jpg"


def test_map_search_item_skips_a_missing_or_non_http_avatar():
    assert "avatar_url" not in map_search_item(_avatar_item())
    assert "avatar_url" not in map_search_item(_avatar_item(avatar="not-a-url"))


def test_enrich_from_items_returns_the_profile_avatar_url():
    items = [
        {**_avatar_item(originalAvatarUrl="https://example.com/a.jpg"), "id": 1},
        {**_avatar_item(avatar="https://example.com/b.jpg"), "id": 2},
    ]

    update = enrich_from_items(items, followers=10)

    assert update["avatar_url"] == "https://example.com/a.jpg"


def test_kol_profile_without_avatar_url_defaults_to_none():
    kol = KolProfile.model_validate(
        {
            "handle": "noavatar",
            "url": "https://www.tiktok.com/@noavatar",
            "source": "apify",
        }
    )

    assert kol.avatar_url is None
