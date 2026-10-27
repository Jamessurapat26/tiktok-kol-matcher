import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura

from .web_reader import WebPage, WebReadError, truncate

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINKED_IMAGE_RE = re.compile(r"\[\s*\]\([^)]*\)")
_EMPTY_BULLET_RE = re.compile(r"^\s*[-*+]\s*$")


def clean_markdown(text: str) -> str:
    """Drop image links, empty links, and empty bullets that Jina emits for nav and icons."""
    text = _IMAGE_RE.sub("", text)
    text = _LINKED_IMAGE_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines() if not _EMPTY_BULLET_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


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
        for attempt in range(attempts):
            try:
                resp = self._client.get(target, headers=headers)
            except httpx.TransportError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    return WebPage(truncate(clean_markdown(resp.text)), "jina")
                reason = f"HTTP {resp.status_code}"
                if resp.status_code != 429 and resp.status_code < 500:
                    raise WebReadError(f"Jina could not read {url}: {reason}")
            if attempt < attempts - 1:
                time.sleep(self._backoff * 2**attempt)

        raise WebReadError(f"Jina failed to read {url} after {attempts} attempts: {reason}")

    def read(self, url: str) -> str:
        return self.fetch(url).content


class TrafilaturaWebReader:
    def __init__(
        self,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff: float = 1.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self._max_retries = max_retries
        self._backoff = backoff

    def _get(self, url: str) -> httpx.Response:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                resp = self._client.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    follow_redirects=True,
                )
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
        return WebPage(truncate(text), "trafilatura")

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
        except WebReadError as exc:
            primary_reason = str(exc)
        else:
            if len(page.content) >= self._min_chars:
                return page
            primary_reason = f"content too short ({len(page.content)} chars)"

        try:
            return self._fallback.fetch(url)
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
