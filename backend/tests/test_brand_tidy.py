"""Offline tests for the brand profile post-validation."""

from __future__ import annotations

import pytest

from app.brand_tidy import tidy_brand_profile
from app.models import BrandProfile


def _audience(**overrides) -> dict:
    data = {
        "age_min": 18,
        "age_max": 30,
        "gender": "any",
        "interests": ["gaming"],
        "location": "Thailand",
    }
    data.update(overrides)
    return data


def _brand(**overrides) -> BrandProfile:
    data = {
        "category": "eyewear",
        "products": ["blue-light glasses"],
        "target_audience": _audience(),
        "tone": "friendly",
        "content_style": "short reviews",
        "keywords_th": ["แว่นตา"],
        "keywords_en": ["blue light glasses"],
        "hashtags": ["#ExampleBrand"],
        "preferred_tiers": ["micro"],
        "reasoning": "From the website.",
    }
    data.update(overrides)
    return BrandProfile.model_validate(data)


def test_sentence_like_thai_and_english_keywords_are_dropped():
    long_thai = (
        "แว่นตากันแสงสีฟ้าสำหรับเกมเมอร์ที่ต้องนั่งหน้าจอคอมพิวเตอร์"
        "เป็นเวลานานทุกวันโดยไม่พักสายตา"
    )
    long_english = "what to look for when buying blue light glasses"
    brand = _brand(
        keywords_th=["แว่นตา", long_thai],
        keywords_en=["blue light glasses", long_english],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.keywords_th == ["แว่นตา"]
    assert tidied.keywords_en == ["blue light glasses"]
    assert long_thai not in tidied.keywords_th
    assert long_english not in tidied.keywords_en


def test_lists_are_deduped_case_insensitively_keeping_first_order():
    brand = _brand(
        products=["Blue Light Glasses", "blue light glasses", "BLUE LIGHT GLASSES"],
        keywords_th=["แว่นตา", "แว่นตา"],
        keywords_en=["Gaming Glasses", "gaming glasses"],
        hashtags=["#ExampleBrand", "#examplebrand"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.products == ["Blue Light Glasses"]
    assert tidied.keywords_th == ["แว่นตา"]
    assert tidied.keywords_en == ["Gaming Glasses"]
    assert tidied.hashtags == ["#ExampleBrand"]


def test_lists_are_capped_at_the_profile_limits():
    brand = _brand(
        products=[f"product {index}" for index in range(12)],
        keywords_th=[f"คำ{index}" for index in range(15)],
        keywords_en=[f"keyword {index}" for index in range(15)],
        hashtags=[f"#tag{index}" for index in range(15)],
        target_audience=_audience(
            interests=[f"interest {index}" for index in range(12)]
        ),
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.products == [f"product {index}" for index in range(8)]
    assert tidied.keywords_th == [f"คำ{index}" for index in range(12)]
    assert tidied.keywords_en == [f"keyword {index}" for index in range(12)]
    assert tidied.hashtags == [f"#tag{index}" for index in range(10)]
    assert tidied.target_audience.interests == [
        f"interest {index}" for index in range(8)
    ]


def test_hashtags_get_one_leading_hash_no_spaces_and_a_length_limit():
    brand = _brand(
        hashtags=[
            "ExampleBrand",
            "##Sale",
            "# blue light",
            "#no spaces here",
            "#ExampleBrand!",
            "x" * 40,
        ]
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.hashtags == [
        "#ExampleBrand",
        "#Sale",
        "#bluelight",
        "#nospaceshere",
    ]
    assert all(item.startswith("#") and not item.startswith("##") for item in tidied.hashtags)
    assert all(" " not in item for item in tidied.hashtags)


def test_fallback_uses_first_words_when_every_item_is_too_long():
    brand = _brand(
        products=[
            "this product name has far too many words to keep",
            "another very long product description that is also too long",
            "a third very long product description that should not be used",
            "a fourth very long product description that should not be used",
        ],
        keywords_en=[
            "what is the best blue light glasses for gaming",
            "how to pick a pair of gaming glasses",
        ],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.products == ["this product name", "another very long", "a third very"]
    assert tidied.keywords_en == ["what is the", "how to pick"]


def test_items_too_long_are_dropped_when_a_valid_item_remains():
    brand = _brand(
        products=["blue-light glasses", "this product name has too many words"],
        keywords_en=["blue light glasses", "a keyword phrase that is too long"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.products == ["blue-light glasses"]
    assert tidied.keywords_en == ["blue light glasses"]


def test_surrounding_quotes_and_punctuation_are_stripped():
    brand = _brand(
        products=['"Gaming glasses"', "(Blue light)?"],
        keywords_en=["blue light glasses."],
        hashtags=["#ExampleBrand!"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.products == ["Gaming glasses", "Blue light"]
    assert tidied.keywords_en == ["blue light glasses"]
    assert tidied.hashtags == ["#ExampleBrand"]


@pytest.mark.parametrize(
    ("age_min", "age_max", "expected"),
    [
        (70, 20, (20, 70)),
        (90, 100, (80, 80)),
        (5, 10, (13, 13)),
        (None, 100, (None, 80)),
        (10, None, (13, None)),
        (None, None, (None, None)),
    ],
)
def test_ages_are_swapped_and_clamped(age_min, age_max, expected):
    brand = _brand(target_audience=_audience(age_min=age_min, age_max=age_max))

    tidied = tidy_brand_profile(brand)

    assert (tidied.target_audience.age_min, tidied.target_audience.age_max) == expected


def test_input_profile_is_not_mutated():
    brand = _brand(
        products=["  Blue   Light Glasses  ", "blue light glasses"],
        keywords_en=["blue light glasses.", "gaming glasses"],
        hashtags=["ExampleBrand", "##Sale"],
        target_audience=_audience(
            age_min=70,
            age_max=20,
            interests=[" Gaming ", "gaming", "too many words here"],
            location=" Thailand ",
        ),
    )
    before = brand.model_dump()

    tidied = tidy_brand_profile(brand)

    assert brand.model_dump() == before
    assert tidied is not brand
    assert tidied.target_audience is not brand.target_audience
    assert tidied.products == ["Blue Light Glasses"]
    assert tidied.keywords_en == ["blue light glasses", "gaming glasses"]
    assert tidied.hashtags == ["#ExampleBrand", "#Sale"]
    assert tidied.target_audience.interests == ["Gaming"]
    assert (tidied.target_audience.age_min, tidied.target_audience.age_max) == (20, 70)
    assert tidied.target_audience.location == "Thailand"
