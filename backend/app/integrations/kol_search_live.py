"""Live TikTok KOL search via the Apify actor ``clockworks/tiktok-scraper``.

Store page: https://apify.com/clockworks/tiktok-scraper

Each query is one synchronous run of
``POST /v2/actors/clockworks~tiktok-scraper/run-sync-get-dataset-items``.
A query that is only hashtags (an optional trailing "tiktok" is ignored) is
sent as ``hashtags``. Every other query is sent as ``searchQueries`` with
``searchSection`` ``/video``, so each dataset item is a video that carries
``authorMeta`` (handle, profile URL, ``fans``, ``signature``).

Videos are deduped by author. The higher follower count wins; empty bio,
nickname, language, and region are filled from the other video. Items with an
``errorCode``, and videos with no author handle, are returned as error records
so ``discover_kols`` skips and records them.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from ..models import normalize_handle
from .apify import ApifyAuthError, ApifyClient, ApifyError, as_count, dig

SEARCH_ACTOR = "clockworks/tiktok-scraper"

_HASHTAG_TOKEN = re.compile(r"#([\w\u0E00-\u0E7F]+)")
_TIKTOK_WORD = re.compile(r"tiktok", re.IGNORECASE)
_UNKNOWN_LANG = frozenset({"un", "und", "unknown", "none", "null"})
_PROFILE_FIELDS = ("nickname", "bio", "text_language", "location_created")

# Charged add-ons stay off. ``resultsPerPage`` is filled per call.
_QUIET_INPUT: dict[str, Any] = {
    "shouldDownloadVideos": False,
    "shouldDownloadCovers": False,
    "shouldDownloadSlideshowImages": False,
    "shouldDownloadSubtitles": False,
    "shouldDownloadAvatars": False,
    "shouldDownloadMusicCovers": False,
    "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
    "scrapeRelatedVideos": False,
    "commentsPerPost": 0,
    "proxyCountryCode": "None",
}


def classify_query(query: str) -> tuple[str, list[str] | str]:
    """Return ``("hashtags", tags)`` or ``("search", text)`` for one query."""
    text = query.strip()
    tags = _HASHTAG_TOKEN.findall(text)
    remainder = _TIKTOK_WORD.sub(" ", _HASHTAG_TOKEN.sub(" ", text)).strip()
    if tags and not remainder:
        unique: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(tag)
        return "hashtags", unique
    return "search", text


def search_input(query: str, *, max_results: int) -> dict[str, Any]:
    """Actor input for one query. Downloads and comment scrapes are off."""
    kind, value = classify_query(query)
    payload: dict[str, Any] = {
        "resultsPerPage": max_results,
        **_QUIET_INPUT,
    }
    if kind == "hashtags":
        payload["hashtags"] = value
    else:
        payload["searchQueries"] = [value]
        # ``/video`` is the actor's video-only search section. Top (``""``)
        # mixes in profile cards that do not carry a video ``authorMeta``.
        payload["searchSection"] = "/video"
    return payload


def map_search_item(item: Any) -> dict[str, Any]:
    """Map one dataset item to a discover-node record, or an error record."""
    if not isinstance(item, dict):
        return {
            "source": "apify",
            "error": "item is not an object",
            "errorCode": "INVALID_ITEM",
        }
    if item.get("errorCode"):
        code = str(item["errorCode"])
        return {
            "source": "apify",
            "error": str(item.get("error") or code),
            "errorCode": code,
        }

    raw_name = dig(item, "authorMeta", "name")
    if not isinstance(raw_name, str) or not raw_name.strip().lstrip("@").strip():
        return {
            "source": "apify",
            "error": "missing author handle",
            "errorCode": "MISSING_AUTHOR",
        }

    handle = raw_name.strip().lstrip("@").strip()
    profile_url = dig(item, "authorMeta", "profileUrl")
    if isinstance(profile_url, str) and profile_url.startswith("http"):
        url = profile_url.split("?", 1)[0]
    else:
        url = f"https://www.tiktok.com/@{handle}"

    fans = as_count(dig(item, "authorMeta", "fans"))
    row: dict[str, Any] = {
        "handle": handle,
        "url": url,
        "followers": fans if fans is not None else 0,
        "source": "apify",
    }
    nickname = dig(item, "authorMeta", "nickName")
    if isinstance(nickname, str) and nickname.strip():
        row["nickname"] = nickname.strip()
    signature = dig(item, "authorMeta", "signature")
    if isinstance(signature, str):
        bio = re.sub(r"\s+", " ", signature).strip()
        if bio:
            row["bio"] = bio
    language = _language(item.get("textLanguage"))
    if language:
        row["text_language"] = language
    region = _region(item)
    if region:
        row["location_created"] = region
    return row


def map_search_items(items: list[Any]) -> list[dict[str, Any]]:
    """Dedupe authors and keep error rows so the discover node can record them."""
    order: list[str] = []
    by_handle: dict[str, dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    for item in items:
        mapped = map_search_item(item)
        if mapped.get("errorCode"):
            bad.append(mapped)
            continue
        key = normalize_handle(str(mapped["handle"]))
        current = by_handle.get(key)
        if current is None:
            by_handle[key] = mapped
            order.append(key)
        else:
            by_handle[key] = _prefer_profile(current, mapped)
    return [by_handle[key] for key in order] + bad


def _prefer_profile(current: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Keep the richer follower count and fill blank profile fields."""
    if int(new.get("followers") or 0) > int(current.get("followers") or 0):
        primary, secondary = new, current
    else:
        primary, secondary = current, new
    merged = dict(primary)
    for field in _PROFILE_FIELDS:
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    return merged


def _language(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().lower()
    if not code or code in _UNKNOWN_LANG:
        return None
    return code


def _region(item: dict[str, Any]) -> str | None:
    for candidate in (
        dig(item, "authorMeta", "region"),
        item.get("locationCreated"),
        dig(item, "authorMeta", "locationCreated"),
    ):
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        value = candidate.strip()
        if len(value) == 2 and value.isalpha():
            return value.upper()
        return value
    return None


class ApifyKolSearch:
    """Search TikTok for creators with ``clockworks/tiktok-scraper``."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        max_results: int = 10,
        timeout: float = 180.0,
        max_retries: int = 1,
        backoff: float = 1.0,
        max_charge_usd: float = 0.50,
        sleeper: Any = None,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.max_results = max_results
        client_kwargs: dict[str, Any] = {
            "client": client,
            "timeout": timeout,
            "max_retries": max_retries,
            "backoff": backoff,
            "max_charge_usd": max_charge_usd,
        }
        if sleeper is not None:
            client_kwargs["sleeper"] = sleeper
        self._apify = ApifyClient(token, **client_kwargs)

    def search(self, queries: list[str], round: int) -> list[dict[str, Any]]:
        """Run one actor call per query and return discover-node records.

        ``round`` is part of the ``KolSearch`` protocol. Apify search does not
        branch on it; the caller already passes only this round's queries.
        """
        del round
        raw_items: list[Any] = []
        errors: list[dict[str, Any]] = []
        for query in queries:
            if not query or not str(query).strip():
                continue
            try:
                items = self._apify.run_sync_get_items(
                    SEARCH_ACTOR,
                    search_input(query, max_results=self.max_results),
                    max_items=self.max_results,
                )
            except ApifyAuthError:
                raise
            except ApifyError as exc:
                # One rejected query must not drop creators the other queries found.
                errors.append(
                    {
                        "source": "apify",
                        "error": str(exc),
                        "errorCode": "APIFY_REQUEST_FAILED",
                    }
                )
                continue
            raw_items.extend(items)
        # Dedupe authors across every query, then keep error rows for discover_kols.
        return map_search_items(raw_items) + errors
