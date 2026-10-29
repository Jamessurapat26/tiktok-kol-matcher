import json
from dataclasses import replace

import httpx
import pytest

from app.cli import main
from app.integrations import ApifyConfigError, build_integrations, mock_integrations
from app.integrations.facebook import MockFacebookScraper
from app.integrations.facebook_live import (
    ApifyFacebookScraper,
    default_facebook_cache_dir,
    page_from_items,
    page_from_text,
)
from app.integrations.llm import MockBrandAnalyzer
from app.integrations.apify import ApifyError
from app.integrations.web_reader import MockWebReader, WebReadError
from app.integrations.web_reader_live import CachedWebReader
from app.models import MAX_POST_CHARS, MAX_RECENT_POSTS, FacebookPage, MatchRequest
from app.nodes import make_nodes
from app.prompts import MAX_PROMPT_POSTS, MAX_PROMPT_POST_CHARS, brand_messages
from app.state import MatchState


def _page(**overrides) -> FacebookPage:
    data = {
        "page_name": "ExampleBrand",
        "title": "ExampleBrand",
        "url": "https://www.facebook.com/examplebrand",
    }
    data.update(overrides)
    return FacebookPage.model_validate(data)


def test_recent_posts_are_trimmed_clipped_and_capped():
    posts = ["  first\n\npost  ", "   ", "x" * (MAX_POST_CHARS + 50)]
    posts += [f"post {n}" for n in range(30)]

    page = _page(recent_posts=posts)

    assert len(page.recent_posts) == MAX_RECENT_POSTS
    assert page.recent_posts[0] == "first post"
    assert len(page.recent_posts[1]) == MAX_POST_CHARS
    assert _page().recent_posts == []


def test_brand_prompt_includes_recent_posts_and_website():
    page = _page(
        website="https://example-brand.co.th",
        categories=["Eyewear Store"],
        intro="Blue-light glasses for gamers and students.",
        recent_posts=[
            "New blue-light frames are in stock",
            "y" * 2_000,
            "Third post",
            "Fourth post",
        ],
    )

    messages = brand_messages("site text", page)
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "weak signal" in system.lower()
    assert "at most 3 words" in system
    assert "at most 4 words" in system
    assert "at most 8 items" in system
    assert "target_followers" in system
    assert "SME or small online shop" in system

    assert "website: https://example-brand.co.th" in user
    assert "categories: Eyewear Store" in user
    assert "intro: Blue-light glasses for gamers and students." in user
    assert (
        "weak_signals_recent_posts (weak signal only: use for tone and"
        " current campaigns; never as the sole source of a product):" in user
    )
    posts_block = user.split("weak_signals_recent_posts", 1)[1]
    assert posts_block.count("\n- ") == MAX_PROMPT_POSTS
    assert "New blue-light frames are in stock" in user
    assert "Third post" in user
    assert "Fourth post" not in user
    assert "y" * (MAX_PROMPT_POST_CHARS - 3) in user
    assert "y" * (MAX_PROMPT_POST_CHARS + 1) not in user

    assert user.index("Website text") < user.index("weak_signals_recent_posts")
    assert user.index("categories:") < user.index("intro:")
    assert user.index("intro:") < user.index("website:")
    assert user.index("website:") < user.index("weak_signals_recent_posts")


# Documented-shape fixtures: field names and value types follow the example
# dataset items on the Apify store pages for apify/facebook-pages-scraper and
# apify/facebook-posts-scraper. They are not live captures.
_URL = "https://www.facebook.com/examplebrand"
_TOKEN = "super-secret-token"
_PAGE_ITEM = {
    "facebookUrl": _URL,
    "categories": ["Page", "Eyewear Store"],
    "info": ["ExampleBrand. 48,200 likes"],
    "likes": 48200,
    "title": "ExampleBrand",
    "pageId": "100000000000001",
    "pageName": "examplebrand",
    "pageUrl": _URL,
    "intro": "Blue-light glasses for gamers and students in Bangkok.",
    "websites": ["https://example-brand.co.th/"],
    "website": "example-brand.co.th",
    "followers": 51000,
    "ad_status": "This Page is currently running ads.",
}

_NOT_AVAILABLE_ITEM = {
    "url": _URL,
    "error": "not_available",
    "errorDescription": (
        "This content isn't available because the owner only shared it with a "
        "small group of people or changed who can see it, or it's been deleted."
    ),
}

_LOGIN_WALL = (
    "Explore the things you love.\n\n"
    "Log into Facebook\n\n"
    "Email or mobile number\n\n"
    "Password\n\n"
    "Log in\n\n"
    "Forgot password?\n\n"
    "Create new account"
)


def _post_item(n: int) -> dict:
    return {
        "facebookUrl": _URL,
        "postId": str(1000 + n),
        "pageName": "examplebrand",
        "url": f"{_URL}/posts/pfbid{n}",
        "time": "2026-01-20T13:00:05.000Z",
        "timestamp": 1768914005 - n,
        "text": f"Post {n}: new blue-light frames\nfor long gaming nights",
        "likes": 10,
        "comments": 1,
        "shares": 0,
    }


class _Router:
    """MockTransport handler that answers per actor and records each call."""

    def __init__(self, pages=None, posts=None):
        self.pages = [_PAGE_ITEM] if pages is None else pages
        self.posts = [_post_item(n) for n in range(25)] if posts is None else posts
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.url.path, dict(request.url.params), body))
        if "facebook-pages-scraper" in request.url.path:
            answer = self.pages
        else:
            answer = self.posts
        if isinstance(answer, int):
            return httpx.Response(answer, json={"error": {"message": "actor failed"}})
        return httpx.Response(201, json=answer)


class _Reader:
    def __init__(self, text: str | Exception = "# ExampleBrand\n\nGaming glasses shop."):
        self.text = text
        self.urls: list[str] = []

    def read(self, url: str) -> str:
        self.urls.append(url)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _scraper(router, tmp_path, reader=None, **kwargs) -> ApifyFacebookScraper:
    return ApifyFacebookScraper(
        token=_TOKEN,
        fallback=reader or _Reader(),
        client=httpx.Client(transport=httpx.MockTransport(router)),
        backoff=0,
        max_retries=0,
        cache_dir=tmp_path / "facebook",
        **kwargs,
    )


def test_apify_page_and_posts_map_to_facebook_page(tmp_path):
    router = _Router()
    reader = _Reader()

    page = _scraper(router, tmp_path, reader).fetch_page(_URL)

    assert page.page_name == "examplebrand"
    assert page.title == "ExampleBrand"
    assert page.categories == ["Page", "Eyewear Store"]
    assert page.intro.startswith("Blue-light glasses")
    assert page.followers == 51000
    assert str(page.website) == "https://example-brand.co.th/"
    assert page.ad_running is True
    assert len(page.recent_posts) == MAX_RECENT_POSTS
    assert page.recent_posts[0] == "Post 0: new blue-light frames for long gaming nights"
    assert reader.urls == []

    (pages_path, pages_params, pages_body), (posts_path, posts_params, posts_body) = router.calls
    assert pages_path.endswith("/apify~facebook-pages-scraper/run-sync-get-dataset-items")
    assert posts_path.endswith("/apify~facebook-posts-scraper/run-sync-get-dataset-items")
    assert pages_body == {"startUrls": [{"url": _URL}]}
    assert posts_body["resultsLimit"] == MAX_RECENT_POSTS
    assert posts_params["maxItems"] == str(MAX_RECENT_POSTS)
    assert pages_params["maxTotalChargeUsd"] == posts_params["maxTotalChargeUsd"] == "0.5"


def test_bare_website_domain_and_follower_fallback(tmp_path):
    item = {**_PAGE_ITEM, "websites": [], "followers": None, "ad_status": None}
    page = _scraper(_Router(pages=[item], posts=[]), tmp_path).fetch_page(_URL)

    assert str(page.website) == "https://example-brand.co.th/"
    assert page.followers == 48200
    assert page.ad_running is False
    assert page.recent_posts == []


def test_apify_error_falls_back_to_web_reader(tmp_path):
    router = _Router(pages=500)
    reader = _Reader()

    page = _scraper(router, tmp_path, reader).fetch_page(_URL)

    assert reader.urls == [_URL]
    assert page.page_name == "examplebrand"
    assert page.title == "ExampleBrand"
    assert page.intro == "# ExampleBrand Gaming glasses shop."
    assert len(router.calls) == 1
    assert not (tmp_path / "facebook").exists()


def test_empty_or_error_dataset_falls_back_to_web_reader(tmp_path):
    for pages in ([], [{"url": _URL, "error": "Page not found"}]):
        reader = _Reader()
        page = _scraper(_Router(pages=pages), tmp_path, reader).fetch_page(_URL)
        assert reader.urls == [_URL]
        assert page.title == "ExampleBrand"


def test_item_error_is_named_and_clipped():
    with pytest.raises(ApifyError) as exc_info:
        page_from_items(_URL, [_NOT_AVAILABLE_ITEM])

    message = str(exc_info.value)
    assert message.startswith("Apify page item error not_available:")
    assert "This content isn't available" in message
    assert len(message) < 220


def test_page_from_text_rejects_the_login_wall():
    with pytest.raises(WebReadError) as exc_info:
        page_from_text(_URL, _LOGIN_WALL)

    assert "login wall" in str(exc_info.value)


def test_not_available_item_and_login_wall_report_both_reasons(tmp_path):
    reader = _Reader(_LOGIN_WALL)
    scraper = _scraper(_Router(pages=[_NOT_AVAILABLE_ITEM]), tmp_path, reader)

    with pytest.raises(WebReadError) as exc_info:
        scraper.fetch_page(_URL)

    message = str(exc_info.value)
    assert "not_available" in message
    assert "login wall" in message
    assert not (tmp_path / "facebook").exists()


def test_fallback_after_item_error_keeps_a_warning_not_the_llm_text(tmp_path):
    reader = _Reader("# ExampleBrand\n\nGaming glasses shop.")
    page = _scraper(_Router(pages=[_NOT_AVAILABLE_ITEM]), tmp_path, reader).fetch_page(_URL)

    assert page.title == "ExampleBrand"
    assert page.source_warning is not None
    assert "not_available" in page.source_warning
    assert not (tmp_path / "facebook").exists()
    prompt = brand_messages("site text", page)[1]["content"]
    assert "not_available" not in prompt
    assert "source_warning" not in prompt


def test_scrape_facebook_surfaces_the_fallback_warning(tmp_path):
    integrations = replace(
        mock_integrations(),
        facebook=_scraper(_Router(pages=[_NOT_AVAILABLE_ITEM]), tmp_path),
    )
    nodes = make_nodes(integrations, min_candidates=1, max_search_rounds=1)
    request = MatchRequest(website_url="https://example-brand.co.th", facebook_url=_URL)

    update = nodes["scrape_facebook"](MatchState(request=request))

    assert update["facebook_page"].source_warning
    assert update["errors"][0].startswith("scrape_facebook: warning: ")
    assert "not_available" in update["errors"][0]


def test_both_sources_failing_raises_with_redacted_reasons(tmp_path):
    reader = _Reader(WebReadError("Jina could not read it: HTTP 403"))
    scraper = _scraper(_Router(pages=500), tmp_path, reader)

    with pytest.raises(WebReadError) as exc:
        scraper.fetch_page(_URL)

    message = str(exc.value)
    assert "Apify failed" in message
    assert "web fallback failed" in message
    assert _TOKEN not in message


def test_posts_failure_keeps_page_and_skips_cache(tmp_path):
    router = _Router(posts=500)

    page = _scraper(router, tmp_path).fetch_page(_URL)

    assert page.title == "ExampleBrand"
    assert page.recent_posts == []
    assert not (tmp_path / "facebook").exists()


def test_cache_hit_skips_apify(tmp_path):
    first = _Router()
    page = _scraper(first, tmp_path).fetch_page(_URL)
    assert len(first.calls) == 2

    second = _Router(pages=500, posts=500)
    cached = _scraper(second, tmp_path).fetch_page(_URL + "/")

    assert second.calls == []
    assert cached == page


def test_expired_cache_refetches(tmp_path):
    now = [1_000.0]
    _scraper(_Router(), tmp_path, clock=lambda: now[0]).fetch_page(_URL)

    now[0] += 8 * 24 * 60 * 60
    router = _Router()
    _scraper(router, tmp_path, clock=lambda: now[0]).fetch_page(_URL)

    assert len(router.calls) == 2


def test_cache_write_failure_still_returns_page(tmp_path):
    blocker = tmp_path / "facebook"
    blocker.write_text("not a directory", encoding="utf-8")

    page = _scraper(_Router(), tmp_path).fetch_page(_URL)

    assert page.title == "ExampleBrand"
    assert len(page.recent_posts) == MAX_RECENT_POSTS


def test_default_cache_dir_respects_kol_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    assert default_facebook_cache_dir() == tmp_path / "facebook"


def test_build_integrations_live_fb_requires_token(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    with pytest.raises(ApifyConfigError, match="APIFY_TOKEN"):
        build_integrations(live_fb=True)
    with pytest.raises(ApifyConfigError, match="APIFY_TOKEN"):
        build_integrations(live_fb=True, apify_token="   ")
    assert isinstance(build_integrations().facebook, MockFacebookScraper)


def test_build_integrations_live_fb_falls_back_to_live_web_reader(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    bundle = build_integrations(live_fb=True, apify_token=_TOKEN)

    assert isinstance(bundle.facebook, ApifyFacebookScraper)
    assert isinstance(bundle.facebook._fallback, CachedWebReader)
    assert isinstance(bundle.web_reader, MockWebReader)


def test_cli_live_fb_without_token_exits_2_before_running(monkeypatch, capsys):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("integrations and graph should not be built")

    monkeypatch.setattr("app.cli.build_integrations", fail)
    monkeypatch.setattr("app.cli.build_graph", fail)
    exit_code = main(
        ["match", "--website", "https://example-brand.co.th", "--facebook", _URL, "--live-fb"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--live-fb requires APIFY_TOKEN" in captured.err
    assert captured.out == ""


def test_cli_live_fb_wires_apify_scraper_without_running_it(monkeypatch, capsys):
    monkeypatch.setenv("APIFY_TOKEN", "cli-test-token")
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["integrations"] = integrations

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter([("values", {"stage": "done", "results": [], "errors": []})])

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            "https://example-brand.co.th",
            "--facebook",
            _URL,
            "--json",
            "--live-fb",
        ]
    )

    captured_io = capsys.readouterr()
    assert exit_code == 0
    assert isinstance(captured["integrations"].facebook, ApifyFacebookScraper)
    assert isinstance(captured["integrations"].web_reader, MockWebReader)
    assert "cli-test-token" not in captured_io.out + captured_io.err


def test_scraped_posts_reach_brand_analysis(tmp_path):
    seen: list[FacebookPage | None] = []

    class _SpyAnalyzer(MockBrandAnalyzer):
        def analyze_brand(self, website_content, facebook_page):
            seen.append(facebook_page)
            return super().analyze_brand(website_content, facebook_page)

    integrations = replace(
        mock_integrations(),
        facebook=_scraper(_Router(), tmp_path),
        llm=_SpyAnalyzer(),
    )
    nodes = make_nodes(integrations, min_candidates=1, max_search_rounds=1)
    request = MatchRequest(website_url="https://example-brand.co.th", facebook_url=_URL)
    update = nodes["scrape_facebook"](MatchState(request=request))
    state = MatchState(request=request, website_content="site text", **update)

    nodes["analyze_brand"](state)

    (page,) = seen
    assert len(page.recent_posts) == MAX_RECENT_POSTS
    prompt = brand_messages("site text", page)[1]["content"]
    assert "Post 0: new blue-light frames" in prompt
    assert "Post 2: new blue-light frames" in prompt
    assert "Post 3: new blue-light frames" not in prompt
