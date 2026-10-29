"""Offline tests for brand-name matching and judge prompt content."""

from __future__ import annotations

import pytest

from app.brand_match import (
    aliases_from_urls,
    mentions_brand,
    normalize_for_match,
)
from app.integrations.llm import MockBrandAnalyzer
from app.models import KolProfile
from app.prompts import JUDGE_SYSTEM, judge_messages

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
    assert mentions_brand(
        _kol("Creator", caption),
        [normalize_for_match(alias)],
        brand_names=[alias],
    )


def test_mentions_brand_is_false_for_another_store_campaign():
    aliases = [
        normalize_for_match(name)
        for name in ("ExampleBrand", "EB-24", "เอ็กแซมเปิล")
    ]

    assert not mentions_brand(_kol("OtherMartCreator", _OTHERMART_CAPTION), aliases)


def test_mentions_brand_does_not_count_the_handle():
    kol = _kol("ExampleBrand", "อาหารและท่องเที่ยว", nickname=None)

    assert not mentions_brand(kol, [normalize_for_match("ExampleBrand")])


def test_judge_messages_include_the_brand_and_one_card_per_creator():
    brand = _brand().model_copy(
        update={"brand_names": ["ExampleBrand", "EB-24", "เอ็กแซมเปิล"]}
    )
    other = _kol("OtherMartCreator", _OTHERMART_CAPTION)
    example = _kol("ExampleCreator", "โปร ExampleBrand ข้าว+น้ำ 69.- #รีวิวของกิน")

    messages = judge_messages(brand, [other, example])
    user = messages[1]["content"]

    assert messages[0]["content"] == JUDGE_SYSTEM
    assert "brand_names: ExampleBrand, EB-24, เอ็กแซมเปิล" in user
    assert user.count("handle: OtherMartCreator") == 1
    assert user.count("handle: ExampleCreator") == 1
    assert "content: โปร ExampleBrand ข้าว+น้ำ 69.- #รีวิวของกิน" in user


def test_judge_system_requires_text_only_audience_evidence():
    assert "Use text/content/bio ONLY" in JUDGE_SYSTEM
    assert "Never infer" in JUDGE_SYSTEM
    assert "brand's own account" in JUDGE_SYSTEM
    assert "risky" in JUDGE_SYSTEM


def test_judge_messages_are_empty_for_no_creators():
    messages = judge_messages(_brand(), [])

    assert "Creators:\n" in messages[1]["content"]
    assert "handle:" not in messages[1]["content"]
