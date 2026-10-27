"""Post-validation cleanup for a live brand profile.

The model occasionally returns list items borrowed from a post, sentence-like
keywords, or inconsistent age bounds. This module keeps the profile useful
without changing the public model: every rule is applied to a copy, never to
the object passed in.
"""

from __future__ import annotations

import string
from collections.abc import Callable

from .models import BrandProfile

MAX_PRODUCTS = 8
MAX_PRODUCT_WORDS = 4
MAX_PRODUCT_CHARS = 60
MAX_KEYWORDS = 12
MAX_KEYWORD_WORDS = 3
MAX_KEYWORD_CHARS = 40
MAX_INTERESTS = 8
MAX_INTEREST_WORDS = 3
MAX_HASHTAGS = 10
MAX_HASHTAG_CHARS = 30
MIN_AGE = 13
MAX_AGE = 80

# Strip quotes and punctuation only at the edges. Inner hyphens, slashes, and
# other punctuation stay, so "blue-light glasses" survives unchanged.
_EDGE_CHARS = string.punctuation + "“”‘’«»–—…"


def tidy_brand_profile(profile: BrandProfile) -> BrandProfile:
    """Return a cleaned copy of ``profile``. The input is not mutated."""
    audience = profile.target_audience
    age_min, age_max = _tidy_ages(audience.age_min, audience.age_max)
    tidy_audience = audience.model_copy(
        update={
            "age_min": age_min,
            "age_max": age_max,
            "interests": _tidy_list(
                audience.interests,
                limit=MAX_INTERESTS,
                max_words=MAX_INTEREST_WORDS,
            ),
        }
    )
    return profile.model_copy(
        update={
            "products": _tidy_list(
                profile.products,
                limit=MAX_PRODUCTS,
                max_words=MAX_PRODUCT_WORDS,
                max_chars=MAX_PRODUCT_CHARS,
            ),
            "target_audience": tidy_audience,
            "keywords_th": _tidy_list(
                profile.keywords_th,
                limit=MAX_KEYWORDS,
                max_words=MAX_KEYWORD_WORDS,
                max_chars=MAX_KEYWORD_CHARS,
            ),
            "keywords_en": _tidy_list(
                profile.keywords_en,
                limit=MAX_KEYWORDS,
                max_words=MAX_KEYWORD_WORDS,
                max_chars=MAX_KEYWORD_CHARS,
            ),
            "hashtags": _tidy_list(
                profile.hashtags,
                limit=MAX_HASHTAGS,
                max_chars=MAX_HASHTAG_CHARS,
                transform=_hashtag,
            ),
            "preferred_tiers": _dedupe_plain(profile.preferred_tiers),
        }
    )


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


def _dedupe_plain(items: list[str]) -> list[str]:
    """Clean and dedupe a list that has no length rules of its own."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _clean_text(raw)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
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


def _hashtag(text: str) -> str:
    """Remove inner spaces and hashes, then add exactly one leading '#'."""
    body = "".join(text.split()).replace("#", "")
    if not body:
        return ""
    return "#" + body


def _tidy_ages(
    age_min: int | None,
    age_max: int | None,
) -> tuple[int | None, int | None]:
    if age_min is not None and age_max is not None and age_min > age_max:
        age_min, age_max = age_max, age_min
    if age_min is not None:
        age_min = max(MIN_AGE, min(MAX_AGE, age_min))
    if age_max is not None:
        age_max = max(MIN_AGE, min(MAX_AGE, age_max))
    return age_min, age_max
