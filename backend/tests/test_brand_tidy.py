"""Offline tests for the brand brief post-validation."""

from __future__ import annotations

from app.brand_tidy import MAX_INTERESTS, MAX_KEY_PRODUCTS, tidy_brand_profile
from app.models import BrandProfile


def _audience(**overrides) -> dict:
    data = {
        "gender": "any",
        "age_range": "18-30",
        "interests": ["gaming"],
    }
    data.update(overrides)
    return data


def _brand(**overrides) -> BrandProfile:
    data = {
        "category": "eyewear",
        "key_products": ["blue-light glasses"],
        "price_tier": "mid",
        "target_audience": _audience(),
        "tone": "friendly",
        "topic_keywords_th": ["แว่นตา"],
        "topic_keywords_en": ["blue light glasses"],
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


def test_sentence_like_thai_and_english_keywords_are_dropped():
    long_thai = (
        "แว่นตากันแสงสีฟ้าสำหรับเกมเมอร์ที่ต้องนั่งหน้าจอคอมพิวเตอร์"
        "เป็นเวลานานทุกวันโดยไม่พักสายตา"
    )
    long_english = "what to look for when buying blue light glasses"
    brand = _brand(
        topic_keywords_th=["แว่นตา", long_thai],
        topic_keywords_en=["blue light glasses", long_english],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.topic_keywords_th == ["แว่นตา"]
    assert tidied.topic_keywords_en == ["blue light glasses"]
    assert long_thai not in tidied.topic_keywords_th
    assert long_english not in tidied.topic_keywords_en


def test_lists_are_deduped_case_insensitively_keeping_first_order():
    brand = _brand(
        key_products=["Blue Light Glasses", "blue light glasses", "BLUE LIGHT GLASSES"],
        topic_keywords_th=["แว่นตา", "แว่นตา"],
        topic_keywords_en=["Gaming Glasses", "gaming glasses"],
        competitors=["OtherLens", "otherlens"],
        brand_handles=["@ExampleBrand", "@examplebrand"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.key_products == ["Blue Light Glasses"]
    assert tidied.topic_keywords_th == ["แว่นตา"]
    assert tidied.topic_keywords_en == ["Gaming Glasses"]
    assert tidied.competitors == ["OtherLens"]
    assert tidied.brand_handles == ["@ExampleBrand"]


def test_lists_are_capped_at_the_profile_limits():
    brand = _brand(
        key_products=[f"product {index}" for index in range(12)],
        topic_keywords_th=[f"คำ{index}" for index in range(15)],
        topic_keywords_en=[f"keyword {index}" for index in range(15)],
        competitors=[f"Competitor{index}" for index in range(15)],
        brand_handles=[f"@handle{index}" for index in range(15)],
        target_audience=_audience(
            interests=[f"interest {index}" for index in range(12)]
        ),
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.key_products == [f"product {index}" for index in range(8)]
    assert tidied.topic_keywords_th == [f"คำ{index}" for index in range(12)]
    assert tidied.topic_keywords_en == [f"keyword {index}" for index in range(12)]
    assert tidied.competitors == [f"Competitor{index}" for index in range(8)]
    assert tidied.brand_handles == [f"@handle{index}" for index in range(8)]
    assert tidied.target_audience.interests == [
        f"interest {index}" for index in range(8)
    ]


def test_brand_names_are_cleaned_deduped_and_capped():
    brand = _brand(
        brand_names=[
            " ExampleBrand ",
            "",
            "examplebrand",
            "EB-24",
            "เอ็กแซมเปิล",
            "x" * 41,
            *[f"Name{index}" for index in range(10)],
        ]
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.brand_names == [
        "ExampleBrand",
        "EB-24",
        "เอ็กแซมเปิล",
        "Name0",
        "Name1",
        "Name2",
        "Name3",
        "Name4",
    ]


def test_fallback_uses_first_words_when_every_item_is_too_long():
    brand = _brand(
        key_products=[
            "this product name has far too many words to keep",
            "another very long product description that is also too long",
            "a third very long product description that should not be used",
            "a fourth very long product description that should not be used",
        ],
        topic_keywords_en=[
            "what is the best blue light glasses for gaming",
            "how to pick a pair of gaming glasses",
        ],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.key_products == [
        "this product name",
        "another very long",
        "a third very",
    ]
    assert tidied.topic_keywords_en == ["what is the", "how to pick"]


def test_items_too_long_are_dropped_when_a_valid_item_remains():
    brand = _brand(
        key_products=["blue-light glasses", "this product name has too many words"],
        topic_keywords_en=["blue light glasses", "a keyword phrase that is too long"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.key_products == ["blue-light glasses"]
    assert tidied.topic_keywords_en == ["blue light glasses"]


def test_surrounding_quotes_and_punctuation_are_stripped():
    brand = _brand(
        key_products=['"Gaming glasses"', "(Blue light)?"],
        topic_keywords_en=["blue light glasses."],
        competitors=["OtherLens!"],
    )

    tidied = tidy_brand_profile(brand)

    assert tidied.key_products == ["Gaming glasses", "Blue light"]
    assert tidied.topic_keywords_en == ["blue light glasses"]
    assert tidied.competitors == ["OtherLens"]


def test_input_profile_is_not_mutated():
    brand = _brand(
        key_products=["  Blue   Light Glasses  ", "blue light glasses"],
        topic_keywords_en=["blue light glasses.", "gaming glasses"],
        target_audience=_audience(
            interests=[" Gaming ", "gaming", "too many words here"],
        ),
    )
    before = brand.model_dump()

    tidied = tidy_brand_profile(brand)

    assert brand.model_dump() == before
    assert tidied is not brand
    assert tidied.target_audience is not brand.target_audience
    assert tidied.key_products == ["Blue Light Glasses"]
    assert tidied.topic_keywords_en == ["blue light glasses", "gaming glasses"]
    assert tidied.target_audience.interests == ["Gaming"]


def test_thai_display_lists_are_capped():
    brand = _brand(
        key_products_th=[f"สินค้า {i}" for i in range(20)],
        interests_th=[f"ความสนใจ {i}" for i in range(20)],
    )

    tidy = tidy_brand_profile(brand)

    assert len(tidy.key_products_th) == MAX_KEY_PRODUCTS
    assert len(tidy.interests_th) == MAX_INTERESTS
