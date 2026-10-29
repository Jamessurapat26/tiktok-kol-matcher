"""Offline tests for brand-name matching and rerank prompt hints."""

from __future__ import annotations

import pytest

from app.brand_match import (
    aliases_from_urls,
    mentions_brand,
    normalize_for_match,
)
from app.integrations.llm import MockBrandAnalyzer
from app.models import KolProfile
from app.prompts import RERANK_SYSTEM, rerank_messages

_OTHERMART_CAPTION = (
    "พามาล่าแสตมป์ 🎁 แลกของที่ OtherMart กันครับทุกคน!! "
    "ช่วงนี้เค้ามีกิจกรรมสะสม E-Stamp แลกของได้เยอะมากกก "
    "#OtherMart #ขุมทรัพย์นักช้อป2569"
)


def _kol(handle: str, caption: str, **overrides) -> KolProfile:
    data = {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle}",
        "nickname": handle,
        "followers": 50_000,
        "language": "th",
        "region": "TH",
        "median_engagement_rate": 0.05,
        "content_summary": caption,
        "source": "seed",
    }
    data.update(overrides)
    return KolProfile.model_validate(data)


def _brand():
    return MockBrandAnalyzer().analyze_brand("website", None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Example-Brand", "examplebrand"),
        ("#ExampleBrandThailand", "examplebrandthailand"),
        ("EB-24", "eb24"),
        ("  Example Brand!! ", "examplebrand"),
    ],
)
def test_normalize_for_match_keeps_only_letters_and_digits(text, expected):
    assert normalize_for_match(text) == expected


def test_normalize_for_match_keeps_thai_script():
    assert normalize_for_match("เอ็กแซมเปิล!") == "เอ็กแซมเปิล"


def test_aliases_from_urls_uses_the_website_label_and_facebook_slug():
    aliases = aliases_from_urls(
        "https://www.example-brand.co.th",
        "https://www.facebook.com/ExampleBrandThailand",
    )
    normalized = {normalize_for_match(alias) for alias in aliases}

    assert "examplebrand" in normalized
    assert "examplebrandthailand" in normalized


@pytest.mark.parametrize(
    ("website_url", "expected"),
    [
        ("https://shop.example-brand.co.th", "example-brand"),
        ("https://m.example-brand.com", "example-brand"),
        ("https://www.example-brand.co.th", "example-brand"),
    ],
)
def test_aliases_from_urls_skips_generic_website_subdomains(
    website_url, expected
):
    assert aliases_from_urls(website_url, "") == [expected]


def test_aliases_from_urls_has_no_website_alias_for_generic_host():
    assert aliases_from_urls("https://shop.co.th", "") == []


def test_aliases_from_urls_ignores_reserved_facebook_paths():
    assert aliases_from_urls("", "https://www.facebook.com/profile.php?id=1") == []


@pytest.mark.parametrize(
    ("caption", "alias"),
    [
        ("#ExampleBrandThailand #รีวิวของกิน", "ExampleBrandThailand"),
        ("EB-24 combo", "EB-24"),
        ("ไปเอ็กแซมเปิลกัน", "เอ็กแซมเปิล"),
    ],
)
def test_mentions_brand_matches_sampled_content(caption, alias):
    assert mentions_brand(_kol("Creator", caption), [normalize_for_match(alias)])


def test_mentions_brand_is_false_for_another_store_campaign():
    aliases = [
        normalize_for_match(name)
        for name in ("ExampleBrand", "EB-24", "เอ็กแซมเปิล")
    ]

    assert not mentions_brand(_kol("OtherMartCreator", _OTHERMART_CAPTION), aliases)


def test_mentions_brand_does_not_count_the_handle():
    kol = _kol("ExampleBrand", "อาหารและท่องเที่ยว", nickname=None)

    assert not mentions_brand(kol, [normalize_for_match("ExampleBrand")])


def test_rerank_messages_mark_whether_each_creator_names_the_brand():
    brand = _brand().model_copy(
        update={"brand_names": ["ExampleBrand", "EB-24", "เอ็กแซมเปิล"]}
    )
    other = _kol("OtherMartCreator", _OTHERMART_CAPTION)
    example = _kol("ExampleCreator", "โปร ExampleBrand ข้าว+น้ำ 69.- #รีวิวของกิน")

    user = rerank_messages(brand, [other, example])[1]["content"]

    assert "brand_names: ExampleBrand, EB-24, เอ็กแซมเปิล" in user
    other_card = user.split("handle: OtherMartCreator", 1)[1].split("\n\n", 1)[0]
    example_card = user.split("handle: ExampleCreator", 1)[1].split("\n\n", 1)[0]
    assert "mentions_brand: no" in other_card
    assert "mentions_brand: yes" in example_card
    assert "from another store or brand is not about this brand" in RERANK_SYSTEM


def test_rerank_messages_omit_the_hint_without_brand_names():
    brand = _brand().model_copy(update={"brand_names": []})
    kol = _kol("OtherMartCreator", _OTHERMART_CAPTION)

    user = rerank_messages(brand, [kol])[1]["content"]

    assert "brand_names:" in user
    assert "mentions_brand:" not in user
