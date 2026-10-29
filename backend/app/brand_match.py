"""Match a creator's own content against a brand's names and aliases."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import SplitResult, urlsplit

from .models import BrandProfile, KolProfile

# Suffixes that may be removed from an alias, but only when separated by "-"
# or "." and only when the remaining alias is long enough to stand alone.
_ALIAS_SUFFIXES = frozenset({"th", "thai", "thailand", "official", "shop", "store"})
_ALIAS_MIN_CHARS = 3
# A stripped or mentioned alias must be long enough to stand on its own.
MIN_ALIAS_CHARS = 5
# A suffix-stripped alias must stand on its own. Generic words are not a
# brand, even when they happen to be five characters long.
_GENERIC_ALIAS_BASES = frozenset(
    {"brand", "shop", "store", "official", "thai", "thailand"}
)

_FACEBOOK_RESERVED_SLUGS = {"profile.php", "pages", "people", "groups"}
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


def _strip_one_suffix(raw: str) -> str | None:
    """Remove one known suffix only when it follows a "-" or "." separator."""
    text = raw.strip()
    cut = max(text.rfind("-"), text.rfind("."))
    if cut <= 0:
        return None
    if text[cut + 1 :].casefold() not in _ALIAS_SUFFIXES:
        return None
    return text[:cut]


def _alias_variants(raw: str) -> list[str]:
    """Return the alias plus safe suffix-stripped forms.

    A suffix is only stripped when it is separated by "-" or "." (so "booth"
    and "HealthSmith" keep their full form) and the stripped alias is long
    enough to be a useful alias on its own.
    """
    variants = [raw]
    current = raw.strip()
    while True:
        base = _strip_one_suffix(current)
        if base is None:
            break
        normalized_base = normalize_for_match(base)
        if (
            len(normalized_base) < MIN_ALIAS_CHARS
            or normalized_base in _GENERIC_ALIAS_BASES
        ):
            break
        variants.append(base)
        current = base
    return variants


def normalized_aliases(values: Iterable[str]) -> list[str]:
    """Normalize raw aliases, adding safe suffix-stripped forms and deduping."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        for variant in _alias_variants(text):
            alias = normalize_for_match(variant)
            if len(alias) < _ALIAS_MIN_CHARS or alias in seen:
                continue
            seen.add(alias)
            result.append(alias)
    return result


def aliases_from_urls(website_url: str, facebook_url: str) -> list[str]:
    """Return brand aliases taken from the website host and Facebook slug."""
    aliases: list[str] = []

    website_label = _website_label(website_url)
    if website_label:
        aliases.extend(_alias_variants(website_label))

    slug = _facebook_slug(facebook_url)
    if slug:
        aliases.extend(_alias_variants(slug))
    return aliases


def brand_aliases(
    brand_profile: BrandProfile,
    website_url: str,
    facebook_url: str,
) -> list[str]:
    """Return normalized, deduped aliases from every brand source."""
    return normalized_aliases(
        (
            *brand_profile.brand_names,
            *brand_profile.brand_handles,
            *aliases_from_urls(website_url, facebook_url),
        )
    )


def account_matches(handle: str, aliases: Iterable[str]) -> bool:
    """True when a handle is an alias, or an alias plus a known suffix.

    This is equality-based after normalization. It never matches on a
    substring, so "book" and "facebook" cannot match the alias "booth".
    """
    normalized_handle = normalize_for_match(handle)
    if not normalized_handle:
        return False
    for raw in aliases:
        alias = normalize_for_match(str(raw))
        if not alias:
            continue
        if normalized_handle == alias:
            return True
        if any(
            normalized_handle == alias + suffix for suffix in _ALIAS_SUFFIXES
        ):
            return True
    return False


def mentions_brand(
    kol: KolProfile,
    aliases: Iterable[str],
    *,
    brand_names: Iterable[str] = (),
) -> bool:
    """True when the creator's profile text names the brand.

    Mentions are substring matches on normalized text, but a short alias is
    only trusted when it is one of the brief's full ``brand_names`` entries.
    """
    text = normalize_for_match(
        " ".join(
            part
            for part in (kol.nickname, kol.bio, kol.content_summary)
            if part
        )
    )
    full_names = {normalize_for_match(name) for name in brand_names}
    for raw in aliases:
        alias = normalize_for_match(str(raw))
        if not alias:
            continue
        if len(alias) < MIN_ALIAS_CHARS and alias not in full_names:
            continue
        if alias in text:
            return True
    return False


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
