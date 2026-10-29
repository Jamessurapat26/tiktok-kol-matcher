"""Live Facebook page reader via Apify, with the web reader as a fallback.

Page profile: https://apify.com/apify/facebook-pages-scraper
Recent posts: https://apify.com/apify/facebook-posts-scraper

Each page costs two synchronous actor runs, both capped by ``max_charge_usd``.
A failed or empty page run falls back to reading the page URL with the web
reader. A failed posts run keeps the page with no posts. Only complete Apify
results are cached, under ``data/cache/facebook`` (or
``$KOL_CACHE_DIR/facebook``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import HttpUrl, TypeAdapter, ValidationError

from ..models import MAX_RECENT_POSTS, FacebookPage
from .apify import ApifyBudget, ApifyBudgetExceeded, ApifyClient, ApifyError, as_count
from .web_reader import WebReader, WebReadError, blocked_page_reason
from .web_reader_live import _DEFAULT_TTL_SECONDS, _normalize_url

PAGES_ACTOR = "apify/facebook-pages-scraper"
POSTS_ACTOR = "apify/facebook-posts-scraper"

_FALLBACK_INTRO_CHARS = 4_000
_ERROR_DESCRIPTION_CHARS = 160
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_HTTP_URL = TypeAdapter(HttpUrl)


def pages_input(url: str) -> dict[str, Any]:
    return {"startUrls": [{"url": url}]}


def posts_input(url: str) -> dict[str, Any]:
    return {
        "startUrls": [{"url": url}],
        "resultsLimit": MAX_RECENT_POSTS,
        "captionText": False,
    }


def page_from_items(url: str, items: list[Any]) -> FacebookPage:
    """Build a page from the first usable pages-scraper item."""
    named = (i for i in items if _usable(i) and (_text(i.get("title")) or _text(i.get("pageName"))))
    item = next(named, None)
    if item is None:
        detail = _item_error(items)
        if detail is not None:
            raise ApifyError(f"Apify page item error {detail}")
        raise ApifyError(f"Apify returned no usable page item for {url}")
    page_name = _text(item.get("pageName")) or _text(item.get("title"))
    followers = as_count(item.get("followers"))
    if followers is None:
        followers = as_count(item.get("likes"))
    ad_status = (_text(item.get("ad_status")) or "").casefold()
    return FacebookPage(
        page_name=page_name,
        title=_text(item.get("title")) or page_name,
        url=url,
        categories=_strings(item.get("categories")),
        intro=_text(item.get("intro")),
        followers=followers or 0,
        website=_website(item),
        ad_running="currently running ads" in ad_status and "not currently" not in ad_status,
    )


def post_texts(items: list[Any]) -> list[str]:
    return [i["text"] for i in items if _usable(i) and isinstance(i.get("text"), str)]


def page_from_text(url: str, text: str) -> FacebookPage:
    """Build a minimal page from web-reader text when Apify has nothing."""
    if not text.strip():
        raise WebReadError(f"web reader returned no text for {url}")
    if reason := blocked_page_reason(text):
        raise WebReadError(f"web reader returned a blocked page for {url}: {reason}")
    parts = urlsplit(url)
    segment = next((s for s in parts.path.split("/") if s), "")
    page_name = segment or parts.netloc
    heading = _HEADING_RE.search(text)
    intro = " ".join(text.split())[:_FALLBACK_INTRO_CHARS].rstrip()
    return FacebookPage(
        page_name=page_name,
        title=heading.group(1) if heading else page_name,
        url=url,
        intro=intro,
    )


def _usable(item: Any) -> bool:
    return isinstance(item, dict) and not item.get("error") and not item.get("errorCode")


def _item_error(items: list[Any]) -> str | None:
    """Name the first item-level error, e.g. ``not_available: ...``."""
    for item in items:
        if not isinstance(item, dict):
            continue
        code = _text(item.get("errorCode")) or _text(item.get("error"))
        if code is None:
            continue
        description = _text(item.get("errorDescription")) or _text(item.get("error")) or ""
        description = " ".join(description.split())
        if len(description) > _ERROR_DESCRIPTION_CHARS:
            description = description[: _ERROR_DESCRIPTION_CHARS - 3].rstrip() + "..."
        if description and description != code:
            return f"{code}: {description}"
        return code
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _website(item: dict[str, Any]) -> str | None:
    # ``websites`` holds full URLs; ``website`` is often a bare domain.
    candidates = _strings(item.get("websites"))
    if bare := _text(item.get("website")):
        candidates.append(bare if "://" in bare else f"https://{bare}")
    for candidate in candidates:
        try:
            return str(_HTTP_URL.validate_python(candidate))
        except ValidationError:
            continue
    return None


def default_facebook_cache_dir() -> Path:
    """Facebook cache: ``$KOL_CACHE_DIR/facebook`` or ``<repo>/data/cache/facebook``."""
    override = os.environ.get("KOL_CACHE_DIR")
    if override:
        return Path(override) / "facebook"
    # This file is backend/app/integrations/facebook_live.py.
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "facebook"


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(_normalize_url(url).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


class ApifyFacebookScraper:
    """Read a public Facebook page and its recent posts through Apify."""

    def __init__(
        self,
        token: str | None = None,
        *,
        fallback: WebReader,
        client: httpx.Client | None = None,
        timeout: float = 180.0,
        max_retries: int = 1,
        backoff: float = 1.0,
        max_charge_usd: float = 0.50,
        budget: ApifyBudget | None = None,
        cache_dir: Path | None = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        sleeper: Any = None,
    ) -> None:
        self._fallback = fallback
        self._cache_dir = cache_dir if cache_dir is not None else default_facebook_cache_dir()
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        client_kwargs: dict[str, Any] = {
            "client": client,
            "timeout": timeout,
            "max_retries": max_retries,
            "backoff": backoff,
            "max_charge_usd": max_charge_usd,
            "budget": budget,
        }
        if sleeper is not None:
            client_kwargs["sleeper"] = sleeper
        self._apify = ApifyClient(token, **client_kwargs)

    def fetch_page(self, url: str) -> FacebookPage:
        path = _cache_path(self._cache_dir, url)
        if (cached := self._load_cached(path)) is not None:
            return cached

        try:
            items = self._apify.run_sync_get_items(PAGES_ACTOR, pages_input(url), max_items=1)
            page = page_from_items(url, items)
        except (ApifyError, ValidationError) as exc:
            return self._from_fallback(url, exc)

        try:
            posts = post_texts(
                self._apify.run_sync_get_items(
                    POSTS_ACTOR, posts_input(url), max_items=MAX_RECENT_POSTS
                )
            )
        except ApifyError:
            # The profile alone still informs brand analysis; skip the cache so
            # the next run retries the posts.
            return page

        # Validate again so the model trims and caps the posts.
        page = FacebookPage.model_validate({**page.model_dump(), "recent_posts": posts})
        self._write_cache(path, url, page)
        return page

    def _from_fallback(self, url: str, apify_error: Exception) -> FacebookPage:
        if isinstance(apify_error, ApifyBudgetExceeded):
            apify_note = f"Apify budget reached ({apify_error})"
        else:
            apify_note = f"Apify failed ({apify_error})"
        try:
            page = page_from_text(url, self._fallback.read(url))
            return page.model_copy(update={"source_warning": str(apify_error)})
        except WebReadError as exc:
            raise WebReadError(
                f"Could not read Facebook page {url}: {apify_note}; "
                f"web fallback failed ({exc})"
            ) from exc

    def _load_cached(self, path: Path) -> FacebookPage | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = payload["fetched_at"]
            page = FacebookPage.model_validate(payload["page"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)):
            return None
        if self._clock() - float(fetched_at) > self._ttl_seconds:
            return None
        return page

    def _write_cache(self, path: Path, url: str, page: FacebookPage) -> None:
        payload = {
            "url": url,
            "fetched_at": self._clock(),
            "page": page.model_dump(mode="json"),
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # A read-only or full disk must not discard a paid result.
            pass
