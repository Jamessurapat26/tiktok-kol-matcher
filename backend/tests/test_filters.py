"""Behavior tests for brand aliases and the hard filters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.brand_match import (
    account_matches,
    brand_aliases,
    mentions_brand,
    normalized_aliases,
)
from app.filters import (
    COMPETITOR_ACCOUNT,
    FOLLOWERS_OUT_OF_RANGE,
    INACTIVE,
    NOT_THAI,
    OWN_BRAND_ACCOUNT,
    SHOP_MEDIA_ACCOUNT,
    hard_filter,
)
from app.matching_config import DEFAULT_CONFIG
from app.models import BrandProfile, KolProfile, TargetAudience, VideoStats

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


def _brief(**overrides) -> BrandProfile:
    data = {
        "category": "gaming eyewear",
        "key_products": ["blue-light glasses"],
        "price_tier": "mid",
        "target_audience": TargetAudience(
            gender="any",
            age_range="16-34",
            interests=["gaming", "tech"],
        ),
        "tone": "friendly",
        "topic_keywords_th": ["แว่นเกมมิ่ง"],
        "topic_keywords_en": ["gaming glasses"],
        "competitors": ["OtherLens"],
        "brand_handles": ["@ExampleBrand"],
        "brand_names": ["ExampleBrand"],
        "target_followers": 30_000,
        "tier_label": "micro",
        "tier_confidence": 0.7,
        "tier_reason": "A small online shop.",
    }
    data.update(overrides)
    return BrandProfile.model_validate(data)


def _video(days_ago: int | None, caption: str = "") -> VideoStats:
    posted_at = None if days_ago is None else NOW - timedelta(days=days_ago)
    return VideoStats(
        url=f"https://www.tiktok.com/@creator/video/{days_ago}",
        posted_at=posted_at,
        caption=caption,
    )


def _kol(handle: str, **overrides) -> KolProfile:
    data = {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle.lstrip('@')}",
        "nickname": handle,
        "followers": 50_000,
        "language": "th",
        "region": "TH",
        "bio": "รีวิวเกมและแว่นกันแสง",
        "videos": [_video(10, "รีวิวแว่นเกมมิ่ง")],
        "source": "apify",
    }
    data.update(overrides)
    return KolProfile.model_validate(data)


def _aliases(brief: BrandProfile) -> list[str]:
    return brand_aliases(
        brief,
        "https://example-brand.co.th",
        "https://www.facebook.com/examplebrand",
    )


def _filter(kol: KolProfile, brief: BrandProfile, aliases=None) -> str | None:
    return hard_filter(
        kol,
        brief,
        aliases if aliases is not None else _aliases(brief),
        normalized_aliases(brief.competitors),
        DEFAULT_CONFIG,
        NOW,
    )


def test_alias_extraction_does_not_strip_unseparated_suffixes():
    brief = _brief(brand_names=["booth", "HealthSmith"])
    aliases = brand_aliases(brief, "", "")

    assert "booth" in aliases
    assert "boo" not in aliases
    assert "healthsmith" in aliases
    assert "healthsmi" not in aliases
    assert not account_matches("book", aliases)
    assert not account_matches("facebook", aliases)
    assert not mentions_brand(
        _kol("reader", nickname="Reader", bio="facebook"),
        aliases,
    )


def test_alias_extraction_requires_a_long_stripped_alias():
    brief = _brief(
        brand_names=["brand-th", "abcd-th", "abcde-th", "examplebrand-th"]
    )
    aliases = brand_aliases(brief, "", "")

    assert "brandth" in aliases
    assert "brand" not in aliases
    assert "abcd" not in aliases
    assert "abcde" in aliases
    assert "examplebrandth" in aliases
    assert "examplebrand" in aliases


def test_own_brand_accounts_are_rejected():
    brief = _brief()

    assert _filter(_kol("examplebrand.th"), brief) == OWN_BRAND_ACCOUNT
    assert _filter(_kol("ExampleBrand"), brief) == OWN_BRAND_ACCOUNT


def test_official_competitor_account_is_rejected():
    brief = _brief()

    assert _filter(_kol("otherlens_official"), brief) == COMPETITOR_ACCOUNT


def test_shop_and_news_accounts_are_rejected():
    brief = _brief()

    assert _filter(_kol("gadgetshop_bkk"), brief) == SHOP_MEDIA_ACCOUNT
    assert _filter(_kol("gamernews.th"), brief) == SHOP_MEDIA_ACCOUNT
    assert (
        _filter(
            _kol("dailyclip", nickname="ข่าวเกมเมอร์", bio="สำนักข่าวเกม"),
            brief,
        )
        == SHOP_MEDIA_ACCOUNT
    )


def test_inactive_creator_is_rejected_but_undated_creator_passes():
    brief = _brief()

    assert (
        _filter(_kol("inactivegamer.tan", videos=[_video(95)]), brief)
        == INACTIVE
    )
    assert _filter(_kol("undatedcreator", videos=[_video(None)]), brief) is None


def test_followers_outside_the_configured_range_are_rejected():
    brief = _brief()

    assert (
        _filter(_kol("tinycreator.num", followers=600), brief)
        == FOLLOWERS_OUT_OF_RANGE
    )
    assert (
        _filter(_kol("megastar.gamer", followers=6_200_000), brief)
        == FOLLOWERS_OUT_OF_RANGE
    )


def test_english_us_creator_is_rejected_as_not_thai():
    brief = _brief()
    kol = _kol(
        "screenreview.joe",
        nickname="Joe Reviews Screens",
        bio="Monitor and gear reviews from Austin",
        language="en",
        region="US",
        videos=[_video(9, "blue light glasses after a ranked grind")],
    )

    assert _filter(kol, brief) == NOT_THAI


def test_thai_caption_keeps_a_creator_with_unknown_origin():
    brief = _brief()
    kol = _kol(
        "unknownorigin",
        nickname="Creator",
        bio="English bio",
        language=None,
        region=None,
        videos=[_video(5, "รีวิวแว่นกันแสงสำหรับเกมเมอร์")],
    )

    assert _filter(kol, brief) is None


def test_normal_creator_passes():
    brief = _brief()

    assert _filter(_kol("normalkol", nickname="Normal KOL"), brief) is None
