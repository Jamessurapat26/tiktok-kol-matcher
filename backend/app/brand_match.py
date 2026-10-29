"""Match a creator's own content against a brand's names and aliases."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit

from .models import BrandProfile, KolProfile

BRAND_UNMENTIONED_RELEVANCE_CAP = 0.6

_FACEBOOK_RESERVED_SLUGS = {"profile.php", "pages", "people", "groups"}
_FACEBOOK_SLUG_SUFFIXES = ("thailand", "official", "thai", "th")
_GENERIC_HOST_LABELS = {
    "www",
    "m",
    "shop",
    "store",
    "online",
    "web",
    "en",
    "th",
    "app",
}
_PUBLIC_SUFFIX_LABELS = {
    "co",
    "com",
    "net",
    "org",
    "or",
    "ac",
    "go",
    "in",
    "th",
}


def normalize_for_match(text: str) -> str:
    """Casefold and keep only letters, digits, and Thai characters."""
    return "".join(
        char
        for char in text.casefold()
        if char.isalnum() or "\u0e00" <= char <= "\u0e7f"
    )


def aliases_from_urls(website_url: str, facebook_url: str) -> list[str]:
    """Return brand aliases taken from the website host and Facebook slug."""
    aliases: list[str] = []

    website_label = _website_label(website_url)
    if website_label:
        aliases.append(website_label)

    slug = _facebook_slug(facebook_url)
    if slug:
        aliases.append(slug)
        for suffix in _FACEBOOK_SLUG_SUFFIXES:
            if not slug.casefold().endswith(suffix):
                continue
            trimmed = slug[: -len(suffix)]
            if len(normalize_for_match(trimmed)) >= 3:
                aliases.append(trimmed)
    return aliases


def brand_aliases(
    brand_profile: BrandProfile,
    website_url: str,
    facebook_url: str,
) -> list[str]:
    """Return normalized, deduped aliases that are long enough to match."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in (
        *brand_profile.brand_names,
        *aliases_from_urls(website_url, facebook_url),
    ):
        alias = normalize_for_match(raw)
        if len(alias) < 3 or alias in seen:
            continue
        seen.add(alias)
        result.append(alias)
    return result


def mentions_brand(kol: KolProfile, aliases: list[str]) -> bool:
    """True when the creator's nickname, bio, or content names the brand."""
    text = normalize_for_match(
        " ".join(
            part
            for part in (kol.nickname, kol.bio, kol.content_summary)
            if part
        )
    )
    return any(
        alias and alias in text
        for alias in (normalize_for_match(item) for item in aliases)
    )


def _website_label(website_url: str) -> str:
    host = _hostname(website_url)
    for label in host.split("."):
        if not label:
            continue
        if label in _GENERIC_HOST_LABELS or label in _PUBLIC_SUFFIX_LABELS:
            continue
        return label
    return ""


def _facebook_slug(facebook_url: str) -> str:
    path = _urlsplit(facebook_url).path.strip("/")
    if not path:
        return ""
    slug = path.split("/")[0]
    if slug.casefold() in _FACEBOOK_RESERVED_SLUGS:
        return ""
    return slug


def _hostname(url: str) -> str:
    return _urlsplit(url).hostname or ""


def _urlsplit(url: str) -> SplitResult:
    text = str(url).strip()
    if not text:
        return urlsplit("")
    if "://" not in text:
        text = "//" + text
    return urlsplit(text)
