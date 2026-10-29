import http.server
import json
import threading
from dataclasses import replace

import httpx
import pytest

from app.integrations import (
    build_integrations,
    mock_integrations,
    real_integrations,
)
from app.integrations.embeddings import MockEmbedder
from app.integrations.facebook import MockFacebookScraper
from app.integrations.kol_enricher import MockKolEnricher
from app.integrations.kol_search import MockKolSearch
from app.integrations.llm import MockBrandAnalyzer
from app.integrations.web_reader import MAX_CONTENT_CHARS, MockWebReader, WebPage, WebReadError
from app.integrations.web_reader_live import (
    CachedWebReader,
    FallbackWebReader,
    JinaWebReader,
    TrafilaturaWebReader,
    default_cache_dir,
    is_public_address,
)
from app.models import MatchRequest
from app.nodes import make_nodes
from app.state import MatchState

_URL = "https://example-brand.co.th/glasses"
_PUBLIC_IP = "93.184.216.34"
_API_KEY = "test-jina-key-not-a-secret-value"
_ARTICLE = (
    "ExampleBrand makes blue-light glasses in Bangkok for gamers and students who "
    "spend long hours on screens. The frames are lightweight and the lenses "
    "keep colours natural. "
) * 4


_CAPTCHA = (
    "Just a moment...\n\n"
    "example-brand.co.th\n\n"
    "Verifying you are human. This may take a few seconds.\n\n"
    "example-brand.co.th needs to review the security of your connection before proceeding.\n\n"
    "Ray ID: 8c1f2a3b4c5d6e7f\n\n"
    "Performance & security by Cloudflare"
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _jina_reply(body: str, *warnings: str) -> str:
    """Text reply shape from r.jina.ai (reader/snapshot-formatter.ts)."""
    mixins = "".join(f"{warning}\n\n" for warning in warnings)
    return f"Title: ExampleBrand\n\nURL Source: {_URL}\n\n{mixins}Markdown Content:\n{body}"


def _article_html() -> str:
    return (
        "<html><head><title>ExampleBrand</title></head>"
        "<body><article><h1>ExampleBrand</h1><p>"
        f"{_ARTICLE}"
        "</p></article></body></html>"
    )


def _jina(handler, **kwargs) -> JinaWebReader:
    kwargs.setdefault("api_key", "")
    kwargs.setdefault("backoff", 0)
    return JinaWebReader(client=_client(handler), **kwargs)


def _trafilatura(handler, **kwargs) -> TrafilaturaWebReader:
    kwargs.setdefault("backoff", 0)
    kwargs.setdefault("resolve", lambda host: [_PUBLIC_IP])
    return TrafilaturaWebReader(client=_client(handler), **kwargs)


class _Stub:
    def __init__(self, page: WebPage | None = None, error: WebReadError | None = None) -> None:
        self._page = page
        self._error = error
        self.calls = 0

    def fetch(self, url: str) -> WebPage:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._page is not None
        return self._page


def test_jina_success_truncates_and_keeps_the_key_out_of_the_page():
    body = "  ExampleBrand blue-light glasses for Thai gamers.  "
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["accept"] = request.headers["accept"]
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(200, text=_jina_reply(body))

    reader = _jina(handler, api_key=_API_KEY)
    page = reader.fetch(_URL)

    assert seen["url"] == f"https://r.jina.ai/{_URL}"
    assert seen["accept"] == "text/markdown"
    assert seen["authorization"] == f"Bearer {_API_KEY}"
    assert page.source == "jina"
    assert page.content == body.strip()
    assert page.content == reader.read(_URL)
    assert _API_KEY not in page.content

    long_body = "z" * (MAX_CONTENT_CHARS + 25)

    def long_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=long_body)

    long_page = _jina(long_handler).fetch(_URL)
    assert long_page.content == "z" * MAX_CONTENT_CHARS
    assert long_page.source == "jina"


def test_jina_strips_image_links_and_empty_bullets():
    body = (
        "# ExampleBrand\n\n"
        "* [![logo](https://x.test/a.png)](https://example-brand.co.th)\n"
        "*\n"
        "- \n"
        "![](https://x.test/b.jpg)\n"
        "[](https://x.test/c)\n\n\n\n"
        "* [Shop](https://example-brand.co.th/shop)\n"
        "Blue-light glasses for Thai gamers."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    content = _jina(handler).fetch(_URL).content

    assert "![" not in content
    assert "a.png" not in content
    assert "b.jpg" not in content
    assert "[Shop](https://example-brand.co.th/shop)" in content
    assert "Blue-light glasses for Thai gamers." in content
    assert "\n\n\n" not in content
    assert not any(line.strip() in {"*", "-", "+"} for line in content.splitlines())


def test_jina_retries_server_error_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ExampleBrand page recovered from Jina. " * 10)

    page = _jina(handler).fetch(_URL)

    assert calls["n"] == 2
    assert page.source == "jina"
    assert "recovered from Jina" in page.content


def test_jina_retries_transient_400_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, text="transient bad request")
        return httpx.Response(200, text="ExampleBrand page recovered after a transient 400. " * 5)

    page = _jina(handler).fetch(_URL)

    assert calls["n"] == 2
    assert page.source == "jina"
    assert "recovered after a transient 400" in page.content


def test_jina_retries_503_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ExampleBrand page recovered after a 503. " * 5)

    page = _jina(handler).fetch(_URL)

    assert calls["n"] == 2
    assert page.source == "jina"
    assert "recovered after a 503" in page.content


def test_jina_names_the_attempt_count_and_last_status_when_400_persists():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(WebReadError) as exc_info:
        _jina(handler).fetch(_URL)

    assert calls["n"] == 3
    message = str(exc_info.value)
    assert "after 3 attempts" in message
    assert "last HTTP status 400" in message


def test_jina_does_not_retry_a_non_retryable_4xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="missing")

    with pytest.raises(WebReadError) as exc_info:
        _jina(handler).fetch(_URL)

    assert calls["n"] == 1
    assert "HTTP 404" in str(exc_info.value)


def test_jina_persistent_400_falls_back_to_trafilatura():
    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="transient bad request")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_article_html())

    page = FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler)).fetch(_URL)

    assert page.source == "trafilatura"
    assert len(page.content) >= 200
    assert "blue-light glasses" in page.content


def test_long_jina_page_does_not_call_fallback():
    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ExampleBrand blue-light glasses. " * 20)

    def direct_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("fallback should not fetch when Jina returns enough text")

    page = FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler)).fetch(_URL)

    assert page.source == "jina"
    assert len(page.content) >= 200


def test_jina_failure_falls_back_to_trafilatura():
    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _URL
        assert "Mozilla" in request.headers["user-agent"]
        assert "text/html" in request.headers["accept"]
        assert "en-US" in request.headers["accept-language"]
        assert "gzip" in request.headers["accept-encoding"]
        return httpx.Response(200, text=_article_html())

    reader = FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler))
    page = reader.fetch(_URL)

    assert page.source == "trafilatura"
    assert len(page.content) >= 200
    assert "blue-light glasses" in page.content
    assert page.content == reader.read(_URL)


def test_short_jina_page_uses_trafilatura_fallback():
    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="short")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_article_html())

    page = FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler)).fetch(_URL)

    assert page.source == "trafilatura"
    assert page.content != "short"
    assert len(page.content) >= 200


def test_jina_target_error_warning_falls_back_and_is_not_cached(tmp_path):
    assert len(_CAPTCHA) >= 200, "the CAPTCHA page must beat min_chars on its own"

    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_jina_reply(_CAPTCHA, "Warning: Target URL returned error 403: Forbidden"),
        )

    def direct_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_article_html())

    reader = CachedWebReader(
        FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler)),
        tmp_path,
    )
    page = reader.fetch(_URL)

    assert page.source == "trafilatura"
    assert "Verifying you are human" not in page.content
    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["source"] == "trafilatura"
    assert "Verifying you are human" not in payload["content"]

    both_fail = FallbackWebReader(
        _jina(jina_handler),
        _Stub(error=WebReadError("extract failed")),
    )
    with pytest.raises(WebReadError) as exc_info:
        both_fail.fetch(_URL)
    assert "Target URL returned error 403" in str(exc_info.value)


def test_jina_header_does_not_count_toward_min_chars():
    body = "ExampleBrand blue-light glasses. " * 5
    reply = _jina_reply(body)
    assert len(body) < 200 < len(reply)

    def jina_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=reply)

    def direct_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_article_html())

    page = FallbackWebReader(_jina(jina_handler), _trafilatura(direct_handler)).fetch(_URL)

    assert page.source == "trafilatura"


def test_short_fallback_page_is_still_returned():
    fallback = _Stub(page=WebPage("tiny", "trafilatura"))
    reader = FallbackWebReader(
        _Stub(error=WebReadError("down")),
        fallback,
    )

    page = reader.fetch(_URL)

    assert fallback.calls == 1
    assert page == WebPage("tiny", "trafilatura")


def test_both_readers_fail_with_a_combined_error_and_no_secret():
    def jina_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        return httpx.Response(404, text="missing")

    def direct_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    reader = FallbackWebReader(
        _jina(jina_handler, api_key=_API_KEY),
        _trafilatura(direct_handler),
    )

    with pytest.raises(WebReadError) as exc_info:
        reader.fetch(_URL)

    message = str(exc_info.value)
    assert message.startswith(f"Could not read {_URL}: primary failed (")
    assert "Jina could not read" in message
    assert "HTTP 404" in message
    assert "fallback failed (" in message
    assert _API_KEY not in message
    assert "Bearer" not in message


def test_too_short_primary_names_the_length_when_fallback_fails():
    reader = FallbackWebReader(
        _Stub(page=WebPage("ab", "jina")),
        _Stub(error=WebReadError("extract failed")),
    )

    with pytest.raises(WebReadError) as exc_info:
        reader.fetch(_URL)

    message = str(exc_info.value)
    assert "primary failed (content too short (2 chars))" in message
    assert "fallback failed (extract failed)" in message


def test_cache_hit_skips_the_network(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text="ExampleBrand แว่นกันแสงสีฟ้า " + ("glasses " * 40))

    reader = CachedWebReader(_jina(handler), tmp_path, clock=lambda: 1_000.0)
    first = reader.fetch("https://Example.com/path/")
    second = reader.fetch("https://example.com/path")

    assert calls["n"] == 1
    assert second == first
    assert first.source == "jina"
    assert reader.read("https://example.com/path") == first.content

    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["url"] == "https://Example.com/path/"
    assert payload["fetched_at"] == 1_000.0
    assert payload["source"] == "jina"
    assert payload["content"] == first.content
    raw = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "แว่นกันแสงสีฟ้า" in raw


def test_cache_expiry_refetches(tmp_path):
    now = {"t": 1_000.0}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text=f"version-{calls['n']} " + ("word " * 40))

    reader = CachedWebReader(
        _jina(handler),
        tmp_path,
        ttl_seconds=10,
        clock=lambda: now["t"],
    )
    first = reader.fetch(_URL)
    now["t"] = 1_010.0
    assert reader.fetch(_URL) == first
    assert calls["n"] == 1

    now["t"] = 1_010.1
    refreshed = reader.fetch(_URL)
    assert calls["n"] == 2
    assert refreshed.content != first.content
    assert refreshed.source == "jina"


def test_corrupt_cache_is_refetched(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, text=f"page-{calls['n']} " + ("word " * 30))

    reader = CachedWebReader(_jina(handler), tmp_path, clock=lambda: 50.0)
    reader.fetch(_URL)
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")

    page = reader.fetch(_URL)

    assert calls["n"] == 2
    assert page.content.startswith("page-2")
    assert page.source == "jina"


def test_failed_fetch_is_not_cached(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(404, text="missing")
        return httpx.Response(200, text="ExampleBrand recovered. " * 20)

    reader = CachedWebReader(_jina(handler, max_retries=0), tmp_path)
    with pytest.raises(WebReadError):
        reader.fetch(_URL)
    assert list(tmp_path.glob("*.json")) == []

    page = reader.fetch(_URL)
    assert calls["n"] == 2
    assert page.source == "jina"
    assert list(tmp_path.glob("*.json"))


def test_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("KOL_CACHE_DIR", raising=False)
    cache = default_cache_dir()
    repo_root = cache.parents[2]
    assert cache == repo_root / "data" / "cache" / "web"
    assert (repo_root / "backend" / "pyproject.toml").is_file()

    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path / "web"


def test_website_content_is_truncated_before_state():
    class LongReader:
        def read(self, url: str) -> str:
            return "ก" * (MAX_CONTENT_CHARS + 80)

    integrations = replace(mock_integrations(), web_reader=LongReader())
    nodes = make_nodes(integrations, min_candidates=1, max_search_rounds=1)
    update = nodes["scrape_website"](
        MatchState(
            request=MatchRequest(
                website_url="https://example-brand.co.th",
                facebook_url="https://www.facebook.com/examplebrand",
            )
        )
    )

    assert update["website_content"] == "ก" * MAX_CONTENT_CHARS
    assert len(update["website_content"]) == MAX_CONTENT_CHARS


def test_factories_keep_non_web_services_mocked(monkeypatch, tmp_path):
    monkeypatch.setenv("KOL_CACHE_DIR", str(tmp_path))
    mocked = build_integrations()
    live = real_integrations()

    assert isinstance(mocked.web_reader, MockWebReader)
    assert isinstance(live.web_reader, CachedWebReader)
    assert live.web_reader is not mocked.web_reader
    for service in (
        mocked.web_reader,
        live.facebook,
        live.llm,
        live.kol_search,
        live.enricher,
        live.embedder,
    ):
        assert service.__class__.__doc__
        assert "MOCK" in service.__class__.__doc__
    assert isinstance(live.facebook, MockFacebookScraper)
    assert isinstance(live.llm, MockBrandAnalyzer)
    assert isinstance(live.kol_search, MockKolSearch)
    assert isinstance(live.enricher, MockKolEnricher)
    assert isinstance(live.embedder, MockEmbedder)


def test_trafilatura_retries_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="local_rate_limited")
        return httpx.Response(200, text=_article_html())

    page = _trafilatura(handler).fetch(_URL)

    assert calls["n"] == 3
    assert page.source == "trafilatura"
    assert "blue-light glasses" in page.content


def test_trafilatura_default_client_uses_http2():
    reader = TrafilaturaWebReader()
    try:
        pool = reader._client._transport._pool
        assert pool._http2 is True
    finally:
        reader._client.close()


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.1.1", "224.0.0.1", "0.0.0.0", "100.64.0.1",
     "::1", "fe80::1", "ff02::1", "::ffff:10.0.0.1", "::"],
)
def test_non_public_addresses_are_refused(address):
    assert not is_public_address(address)


def test_public_addresses_are_allowed():
    assert is_public_address(_PUBLIC_IP)
    assert is_public_address("2606:4700::1111")


def test_trafilatura_never_connects_to_a_loopback_server():
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(_article_html().encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    reader = TrafilaturaWebReader(max_retries=0, backoff=0)
    try:
        with pytest.raises(WebReadError) as exc_info:
            reader.fetch(f"http://127.0.0.1:{server.server_port}/admin")
    finally:
        server.shutdown()
        reader._client.close()

    assert hits == []
    assert "refused" in str(exc_info.value).lower()


def test_trafilatura_refuses_a_redirect_to_a_private_address():
    hits = []
    addresses = {"example-brand.co.th": [_PUBLIC_IP], "127.0.0.1": ["127.0.0.1"]}

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        if request.url.host == "example-brand.co.th":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:9/"})
        return httpx.Response(200, text=_article_html())

    reader = _trafilatura(handler, max_retries=0, resolve=addresses.__getitem__)

    with pytest.raises(WebReadError):
        reader.fetch(_URL)

    assert hits == ["example-brand.co.th"]


def test_trafilatura_follows_a_public_redirect():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/glasses":
            return httpx.Response(301, headers={"location": "/glasses/"})
        return httpx.Response(200, text=_article_html())

    page = _trafilatura(handler).fetch(_URL)

    assert page.source == "trafilatura"
    assert "blue-light glasses" in page.content


def test_trafilatura_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(WebReadError):
        _trafilatura(handler).fetch(_URL)

    assert calls["n"] == 1
