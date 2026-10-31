"""The Thai display fields must never reach the judge or query prompts."""

from __future__ import annotations

from app.models import BrandProfile, TargetAudience
from app.prompts import JUDGE_SYSTEM, judge_messages, query_messages


def _brand() -> BrandProfile:
    return BrandProfile(
        category="eyewear",
        key_products=["glasses"],
        price_tier="mid",
        target_audience=TargetAudience(gender="any", age_range="18-30", interests=["gaming"]),
        tone="friendly",
        topic_keywords_th=["แว่นตา"],
        topic_keywords_en=["glasses"],
        competitors=[],
        brand_handles=[],
        tier_label="micro",
        tier_confidence=0.7,
        tier_reason="A small shop.",
        category_th="DISPLAY_CATEGORY",
        key_products_th=["DISPLAY_PRODUCT"],
        tone_th="DISPLAY_TONE",
        interests_th=["DISPLAY_INTEREST"],
        tier_reason_th="DISPLAY_REASON",
    )


def test_thai_display_fields_stay_out_of_later_prompts():
    for messages in (judge_messages(_brand(), []), query_messages(_brand(), [], 1)):
        assert "DISPLAY_" not in str(messages)


def test_judge_writes_its_reason_in_thai():
    assert "reason is one short sentence in Thai" in JUDGE_SYSTEM
