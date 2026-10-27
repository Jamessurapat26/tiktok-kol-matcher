import json

import httpx
import pytest

from app.integrations.facebook_live import ApifyFacebookScraper, default_facebook_cache_dir
from app.integrations.web_reader import WebReadError
from app.models import MAX_POST_CHARS, MAX_RECENT_POSTS, FacebookPage
from app.prompts import brand_messages


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
        recent_posts=["New blue-light frames are in stock", "y" * 2_000],
    )

    user = brand_messages("site text", page)[1]["content"]

    assert "website: https://example-brand.co.th" in user
    assert "recent_posts:\n- New blue-light frames are in stock" in user
    assert "y" * 400 not in user


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
