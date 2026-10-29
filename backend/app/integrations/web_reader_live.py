import hashlib
import ipaddress
import json
import os
import re
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura

from .web_reader import WebPage, WebReadError, blocked_page_reason, truncate

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Some Shopify storefronts answer 200 to curl and 429 to a bare httpx client.
# curl negotiates HTTP/2 and sends a browser Accept set; httpx defaults to
# HTTP/1.1 with ``accept: */*``. These headers plus ``http2=True`` close that gap.
_BROWSER_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINKED_IMAGE_RE = re.compile(r"\[\s*\]\([^)]*\)")
_EMPTY_BULLET_RE = re.compile(r"^\s*[-*+]\s*$")
# r.jina.ai text replies: "Title: ...", "URL Source: ...", optional "Warning: ..." lines,
# then "Markdown Content:" and the body (jina-ai/reader snapshot-formatter.ts).
_JINA_BODY_MARKER = "\nMarkdown Content:\n"
_JINA_TARGET_ERROR_RE = re.compile(r"^Warning: Target URL returned error.*$", re.MULTILINE)

# r.jina.ai occasionally answers a transient 400 for a URL it can read moments
# later, so treat that plus the usual timeout/rate-limit statuses as retryable.
# Every 5xx is retryable too; other 4xx (401, 403, 404, 410, 422, ...) are not.
_JINA_RETRYABLE_STATUSES = frozenset({400, 408, 425, 429})


def _is_jina_retryable_status(status: int) -> bool:
    return status in _JINA_RETRYABLE_STATUSES or status >= 500


def clean_markdown(text: str) -> str:
    """Drop image links, empty links, and empty bullets that Jina emits for nav and icons."""
    text = _IMAGE_RE.sub("", text)
    text = _LINKED_IMAGE_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines() if not _EMPTY_BULLET_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _reject_blocked_page(url: str, text: str, reader: str) -> None:
    reason = blocked_page_reason(text)
    if reason is not None:
        raise WebReadError(f"{reader} read a blocked page for {url}: {reason}")


class JinaWebReader:
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff: float = 1.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("JINA_API_KEY")
        self._client = client or httpx.Client(timeout=timeout)
        self._max_retries = max_retries
        self._backoff = backoff

    def fetch(self, url: str) -> WebPage:
        headers = {"Accept": "text/markdown"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        target = f"https://r.jina.ai/{url}"
        attempts = self._max_retries + 1
        reason = ""
        last_status: int | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.get(target, headers=headers)
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                last_status = None
            else:
                if resp.status_code < 300:
                    header, marker, body = resp.text.partition(_JINA_BODY_MARKER)
                    if not marker:
                        header, body = "", resp.text
                    # Jina answers 200 even when the target site failed; the error page
                    # (often a CAPTCHA) is only flagged by this header line.
                    if error := _JINA_TARGET_ERROR_RE.search(header):
                        raise WebReadError(f"Jina could not read {url}: {error.group()}")
                    content = truncate(clean_markdown(body))
                    _reject_blocked_page(url, content, "Jina")
                    return WebPage(content, "jina")
                last_status = resp.status_code
                reason = f"HTTP {resp.status_code}"
                if not _is_jina_retryable_status(resp.status_code):
                    raise WebReadError(f"Jina could not read {url}: {reason}")
            if attempt < attempts - 1:
                time.sleep(self._backoff * 2**attempt)

        detail = f"last HTTP status {last_status}" if last_status is not None else reason
        raise WebReadError(f"Jina failed to read {url} after {attempts} attempts: {detail}")

    def read(self, url: str) -> str:
        return self.fetch(url).content


Resolver = Callable[[str], list[str]]

_MAX_REDIRECTS = 10
# Fake-IP range some sandbox proxies return for every host. Only the direct
# fetch guard can optionally allow it, via WEB_FETCH_ALLOW_PROXY_RANGE.
_PROXY_RANGE = ipaddress.ip_network("198.18.0.0/15")


def resolve_host(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def is_public_address(address: str, *, allow_proxy_range: bool = False) -> bool:
    # is_global alone still says yes to multicast ranges.
    ip = ipaddress.ip_address(address)
    if allow_proxy_range and ip in _PROXY_RANGE:
        return True
    return ip.is_global and not ip.is_multicast


def _assert_public(
    host: str,
    resolve: Resolver,
    *,
    allow_proxy_range: bool = False,
) -> None:
    """Refuse hosts that resolve to loopback, private, link-local, or other non-public addresses.

    The job URL comes from the API caller, so a direct fetch of it must not reach
    this machine or its network. The check resolves separately from the connect,
    so a host whose DNS answer flips between the two (rebinding) is not covered.
    ``allow_proxy_range`` additionally permits exactly the fake-IP range some
    sandboxes use; it is explicit and off unless the live reader enables it.
    """
    try:
        addresses = resolve(host)
    except OSError as exc:
        raise WebReadError(f"Could not resolve {host}") from exc
    if not addresses or not all(
        is_public_address(address, allow_proxy_range=allow_proxy_range)
        for address in addresses
    ):
        raise WebReadError(f"Refused to fetch {host}: not a public address")


class TrafilaturaWebReader:
    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff: float = 1.0,
        resolve: Resolver = resolve_host,
        allow_proxy_range: bool = False,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout, http2=True)
        self._max_retries = max_retries
        self._backoff = backoff
        self._resolve = resolve
        self._allow_proxy_range = allow_proxy_range

    def _send(self, url: str) -> httpx.Response:
        """One GET, following redirects by hand so every hop's host is checked."""
        request = self._client.build_request("GET", url, headers=_BROWSER_HEADERS)
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_public(
                request.url.host,
                self._resolve,
                allow_proxy_range=self._allow_proxy_range,
            )
            resp = self._client.send(request, follow_redirects=False)
            if resp.next_request is None:
                return resp
            request = resp.next_request
        raise WebReadError(f"Failed to fetch {url}: too many redirects")

    def _get(self, url: str) -> httpx.Response:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                resp = self._send(url)
                if resp.status_code != 429 and resp.status_code < 500:
                    resp.raise_for_status()
                    return resp
                error: httpx.HTTPError = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            except httpx.TransportError as exc:
                error = exc
            except httpx.HTTPError as exc:
                raise WebReadError(f"Failed to fetch {url}: {type(exc).__name__}: {exc}") from exc
            if attempt == attempts - 1:
                raise WebReadError(
                    f"Failed to fetch {url}: {type(error).__name__}: {error}"
                ) from error
            time.sleep(self._backoff * 2**attempt)
        raise AssertionError("unreachable")

    def fetch(self, url: str) -> WebPage:
        resp = self._get(url)
        text = trafilatura.extract(resp.text, output_format="markdown", include_links=False)
        if not text:
            raise WebReadError(f"no readable content extracted from {url}")
        content = truncate(text)
        _reject_blocked_page(url, content, "trafilatura")
        return WebPage(content, "trafilatura")

    def read(self, url: str) -> str:
        return self.fetch(url).content


_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60


class PageFetcher(Protocol):
    def fetch(self, url: str) -> WebPage:
        """Fetch one page and return its text plus which reader produced it."""
        ...


def _normalize_url(url: str) -> str:
    """Lowercase the scheme and host, and drop one trailing slash."""
    parts = urlsplit(url.strip())
    normalized = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            parts.query,
            parts.fragment,
        )
    )
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def _cache_file(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(_normalize_url(url).encode()).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_cached_page(path: Path, *, now: float, ttl_seconds: float) -> WebPage | None:
    """Return a fresh cached page, or None on a miss, expiry, or bad file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = payload["fetched_at"]
        content = payload["content"]
        source = payload["source"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        isinstance(fetched_at, bool)
        or not isinstance(fetched_at, (int, float))
        or not isinstance(content, str)
        or not isinstance(source, str)
    ):
        return None
    if now - float(fetched_at) > ttl_seconds:
        return None
    if blocked_page_reason(content) is not None:
        return None
    return WebPage(content, source)


class FallbackWebReader:
    """Try the primary reader, then the fallback when it fails or is too short."""

    def __init__(
        self,
        primary: PageFetcher,
        fallback: PageFetcher,
        min_chars: int = 200,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._min_chars = min_chars

    def fetch(self, url: str) -> WebPage:
        try:
            page = self._primary.fetch(url)
            _reject_blocked_page(url, page.content, "primary reader")
        except WebReadError as exc:
            primary_reason = str(exc)
        else:
            if len(page.content) >= self._min_chars:
                return page
            primary_reason = f"content too short ({len(page.content)} chars)"

        try:
            page = self._fallback.fetch(url)
            _reject_blocked_page(url, page.content, "fallback reader")
            return page
        except WebReadError as exc:
            raise WebReadError(
                f"Could not read {url}: primary failed ({primary_reason}); "
                f"fallback failed ({exc})"
            ) from exc

    def read(self, url: str) -> str:
        return self.fetch(url).content


class CachedWebReader:
    """File cache in front of another page fetcher. Hits do not call the inner reader."""

    def __init__(
        self,
        inner: PageFetcher,
        cache_dir: Path,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def fetch(self, url: str) -> WebPage:
        path = _cache_file(self._cache_dir, url)
        cached = _load_cached_page(
            path,
            now=self._clock(),
            ttl_seconds=self._ttl_seconds,
        )
        if cached is not None:
            return cached

        page = self._inner.fetch(url)
        _reject_blocked_page(url, page.content, "web reader")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "url": url,
            "fetched_at": self._clock(),
            "source": page.source,
            "content": page.content,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return page

    def read(self, url: str) -> str:
        return self.fetch(url).content


def default_cache_dir() -> Path:
    """Web cache location: $KOL_CACHE_DIR/web if set, else <repo root>/data/cache/web."""
    override = os.environ.get("KOL_CACHE_DIR")
    if override:
        return Path(override) / "web"
    # This file is backend/app/integrations/web_reader_live.py.
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "web"
