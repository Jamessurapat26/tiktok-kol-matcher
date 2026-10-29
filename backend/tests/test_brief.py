"""Offline tests for the brand brief schema and tier parsing."""

from __future__ import annotations

import json

import pytest

from app.integrations.llm_live import LiveBrandAnalyzer, LlmSettings
from app.matching_config import MatchingConfig
from app.models import BrandProfile


class _FakeChat:
    def __init__(self, payload: dict) -> None:
        self._reply = json.dumps(payload)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self._reply


def _brief(**overrides) -> dict:
    data = {
        "category": "gaming eyewear",
        "key_products": ["blue-light glasses"],
        "price_tier": "mid",
        "target_audience": {
            "gender": "any",
            "age_range": "18-34",
            "interests": ["gaming", "tech"],
        },
        "tone": "friendly",
        "topic_keywords_th": ["แว่นเกมมิ่ง"],
        "topic_keywords_en": ["gaming glasses"],
        "competitors": ["OtherLens"],
        "brand_handles": ["@ExampleBrand"],
        "brand_names": ["ExampleBrand"],
        "target_followers": 30_000,
        "tier_label": "micro",
        "tier_confidence": 0.7,
        "tier_reason": "A small online shop, so a micro creator around 30k.",
    }
    data.update(overrides)
    return data


def _analyze(payload: dict) -> BrandProfile:
    chat = _FakeChat(payload)
    analyzer = LiveBrandAnalyzer(
        settings=LlmSettings(api_key="test-key"),
        chat=chat,
    )
    return analyzer.analyze_brand("A small shop sells gaming glasses.", None)


def test_sme_signal_parses_to_micro_around_30k():
    profile = _analyze(_brief())

    assert profile.tier_label == "micro"
    assert profile.target_followers == 30_000
    assert "small online shop" in profile.tier_reason


def test_multinational_signal_parses_to_macro():
    profile = _analyze(
        _brief(
            price_tier="premium",
            target_followers=800_000,
            tier_label="macro",
            tier_confidence=0.9,
            tier_reason="A multinational brand sold through nationwide stores.",
        )
    )

    assert profile.tier_label == "macro"
    assert profile.target_followers == 800_000


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 10_000), (9_000_000, 2_000_000)],
)
def test_target_followers_is_clamped_to_the_config_bounds(value, expected):
    profile = BrandProfile.model_validate(_brief(target_followers=value))

    assert profile.target_followers == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"fit_w_sim": 0.5},
        {"fit_w_on_topic": 0.4},
        {"quality_w_e": 0.5},
    ],
)
def test_matching_config_rejects_weights_that_do_not_sum_to_one(overrides):
    with pytest.raises(ValueError, match="must sum to 1"):
        MatchingConfig(**overrides)


def test_matching_config_defaults_are_valid():
    config = MatchingConfig()

    assert config.fit_w_sim + config.fit_w_on_topic + config.fit_w_judge == pytest.approx(1.0)
    assert config.quality_w_e + config.quality_w_t + config.quality_w_a == pytest.approx(1.0)


def test_live_brief_ignores_a_model_follower_count_and_maps_the_tier():
    # The model copied the brand page's own 806k followers as the target.
    profile = _analyze(_brief(target_followers=806_000, tier_label="mid"))

    assert profile.tier_label == "mid"
    assert profile.target_followers == MatchingConfig().tier_target_mid


@pytest.mark.parametrize(
    ("tier_label", "expected"),
    [("nano", 10_000), ("micro", 30_000), ("mid", 150_000), ("macro", 800_000)],
)
def test_missing_target_followers_comes_from_the_tier_label(tier_label, expected):
    data = _brief(tier_label=tier_label)
    del data["target_followers"]

    assert BrandProfile.model_validate(data).target_followers == expected


def test_matching_config_rejects_tier_targets_out_of_order():
    with pytest.raises(ValueError, match="tier targets"):
        MatchingConfig(tier_target_mid=20_000)


def test_retailers_are_kept_apart_from_competitors():
    profile = _analyze(
        _brief(
            competitors=["OtherLens", "ExampleMart"],
            retailers="ExampleMart, ExamplePharmacy",
        )
    )

    assert profile.competitors == ["OtherLens"]
    assert profile.retailers == ["ExampleMart", "ExamplePharmacy"]


def test_retailers_default_to_empty_for_older_briefs():
    assert BrandProfile.model_validate(_brief()).retailers == []
