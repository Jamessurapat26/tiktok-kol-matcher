"""Post-validation cleanup for a live brand brief.

The model occasionally returns list items borrowed from a post or
sentence-like keywords. This module keeps the brief useful without changing
the public model: every rule is applied to a copy, never to the object passed
in.
"""

from __future__ import annotations

import string
from collections.abc import Callable

from .models import BrandProfile

MAX_KEY_PRODUCTS = 8
MAX_PRODUCT_WORDS = 4
MAX_PRODUCT_CHARS = 60
MAX_BRAND_NAMES = 8
MAX_BRAND_NAME_CHARS = 40
MAX_KEYWORDS = 12
MAX_KEYWORD_WORDS = 3
MAX_KEYWORD_CHARS = 40
MAX_INTERESTS = 8
MAX_INTEREST_WORDS = 3
MAX_COMPETITORS = 8
MAX_COMPETITOR_WORDS = 4
MAX_RETAILERS = 8
MAX_BRAND_HANDLES = 8
MAX_BRAND_HANDLE_CHARS = 40

# Strip quotes and punctuation only at the edges. Inner hyphens, slashes, and
# other punctuation stay, so "blue-light glasses" survives unchanged.
_EDGE_CHARS = string.punctuation + "“”‘’«»–—…"


def tidy_brand_profile(profile: BrandProfile) -> BrandProfile:
    """Return a cleaned copy of ``profile``. The input is not mutated."""
    audience = profile.target_audience
    tidy_audience = audience.model_copy(
        update={
            "interests": _tidy_list(
                audience.interests,
                limit=MAX_INTERESTS,
                max_words=MAX_INTEREST_WORDS,
            ),
        }
    )
    retailers = _tidy_list(
        profile.retailers,
        limit=MAX_RETAILERS,
        max_words=MAX_COMPETITOR_WORDS,
        max_chars=MAX_BRAND_NAME_CHARS,
    )
    return profile.model_copy(
        update={
            "key_products": _tidy_list(
                profile.key_products,
                limit=MAX_KEY_PRODUCTS,
                max_words=MAX_PRODUCT_WORDS,
                max_chars=MAX_PRODUCT_CHARS,
            ),
            "brand_names": _filter_items(
                profile.brand_names,
                limit=MAX_BRAND_NAMES,
                max_words=None,
                max_chars=MAX_BRAND_NAME_CHARS,
                transform=None,
            ),
            "target_audience": tidy_audience,
            "key_products_th": _plain_items(
                profile.key_products_th,
                limit=MAX_KEY_PRODUCTS,
                max_chars=MAX_PRODUCT_CHARS,
            ),
            "interests_th": _plain_items(
                profile.interests_th,
                limit=MAX_INTERESTS,
                max_chars=MAX_PRODUCT_CHARS,
            ),
            "topic_keywords_th": _tidy_list(
                profile.topic_keywords_th,
                limit=MAX_KEYWORDS,
                max_words=MAX_KEYWORD_WORDS,
                max_chars=MAX_KEYWORD_CHARS,
            ),
            "topic_keywords_en": _tidy_list(
                profile.topic_keywords_en,
                limit=MAX_KEYWORDS,
                max_words=MAX_KEYWORD_WORDS,
                max_chars=MAX_KEYWORD_CHARS,
            ),
            "competitors": _drop_listed(
                _tidy_list(
                    profile.competitors,
                    limit=MAX_COMPETITORS,
                    max_words=MAX_COMPETITOR_WORDS,
                    max_chars=MAX_BRAND_NAME_CHARS,
                ),
                retailers,
            ),
            "retailers": retailers,
            "brand_handles": _plain_items(
                profile.brand_handles,
                limit=MAX_BRAND_HANDLES,
                max_chars=MAX_BRAND_HANDLE_CHARS,
            ),
        }
    )


def _drop_listed(items: list[str], listed: list[str]) -> list[str]:
    """Remove items that also appear in ``listed``, ignoring case.

    A seller named as both a retailer and a competitor stays a retailer, so it
    never triggers the competitor filter or flags.
    """
    keys = {item.casefold() for item in listed}
    return [item for item in items if item.casefold() not in keys]


def _clean_text(value: str) -> str:
    """Collapse whitespace and strip quotes or punctuation from the edges."""
    text = " ".join(str(value).split())
    while True:
        cleaned = " ".join(text.strip(_EDGE_CHARS).split())
        if cleaned == text:
            return cleaned
        text = cleaned


def _tidy_list(
    items: list[str],
    *,
    limit: int,
    max_words: int | None = None,
    max_chars: int | None = None,
    transform: Callable[[str], str] | None = None,
) -> list[str]:
    cleaned = _filter_items(
        items,
        limit=limit,
        max_words=max_words,
        max_chars=max_chars,
        transform=transform,
    )
    if cleaned or not items:
        return cleaned
    # Everything was dropped. Keep a small deterministic fallback instead of
    # returning an empty list, using the first few words of the first few
    # original entries. The fallback intentionally skips the length filters:
    # its purpose is to keep at least one usable item.
    return _fallback_items(items, limit=min(limit, 3), transform=transform)


def _plain_items(
    items: list[str],
    *,
    limit: int,
    max_chars: int,
) -> list[str]:
    """Collapse whitespace without stripping edge punctuation such as ``@``."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = " ".join(str(raw).split())
        if not text or len(text) > max_chars:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _filter_items(
    items: list[str],
    *,
    limit: int,
    max_words: int | None,
    max_chars: int | None,
    transform: Callable[[str], str] | None,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _clean_text(raw)
        if transform is not None:
            text = transform(text)
        if not text:
            continue
        if max_words is not None and len(text.split()) > max_words:
            continue
        if max_chars is not None and len(text) > max_chars:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _fallback_items(
    items: list[str],
    *,
    limit: int,
    transform: Callable[[str], str] | None,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in items[:3]:
        words = _clean_text(raw).split()
        if not words:
            continue
        text = " ".join(words[:3])
        if transform is not None:
            text = transform(text)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result
