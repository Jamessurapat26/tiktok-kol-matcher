"""Behavior tests for the matching-v2 scoring functions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.brand_match import normalized_aliases
from app.matching_config import DEFAULT_CONFIG, MatchingConfig
from app.models import (
    BrandProfile,
    Judgement,
    KolProfile,
    TargetAudience,
    VideoStats,
)
from app.scoring import (
    audience_eff,
    engagement,
    fit,
    flags_for,
    mmr_select,
    on_topic_share,
    percentile_ranks,
    quality,
    recent_videos,
    score_pool,
    tier_fit,
    tier_of,
)

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
        "target_followers": 50_000,
        "tier_label": "micro",
        "tier_confidence": 0.7,
        "tier_reason": "A small online shop.",
    }
    data.update(overrides)
    return BrandProfile.model_validate(data)


def _video(
    *,
    views: int = 1_000,
    likes: int = 50,
    comments: int = 0,
    shares: int = 0,
    days_ago: int | None = 5,
    caption: str = "",
) -> VideoStats:
    return VideoStats(
        url=f"https://www.tiktok.com/@creator/video/{views}-{likes}-{days_ago}",
        views=views,
        likes=likes,
        comments=comments,
        shares=shares,
        posted_at=None if days_ago is None else NOW - timedelta(days=days_ago),
        caption=caption,
    )


def _kol(
    handle: str,
    *,
    followers: int = 50_000,
    videos: list[VideoStats] | None = None,
    stats_source: str = "search",
    nickname: str | None = None,
    bio: str | None = "รีวิวเกมและแว่นกันแสง",
) -> KolProfile:
    return KolProfile(
        handle=handle,
        url=f"https://www.tiktok.com/@{handle}",
        nickname=nickname or handle,
        followers=followers,
        language="th",
        region="TH",
        bio=bio,
        videos=videos if videos is not None else [],
        stats_source=stats_source,
        source="apify",
    )


def _brand_aliases(brief: BrandProfile) -> list[str]:
    return normalized_aliases([*brief.brand_names, *brief.brand_handles])


def _competitor_aliases(brief: BrandProfile) -> list[str]:
    return normalized_aliases(brief.competitors)


def test_tier_of_uses_the_configured_boundaries():
    cfg = DEFAULT_CONFIG

    assert tier_of(9_999, cfg) == "nano"
    assert tier_of(10_000, cfg) == "micro"
    assert tier_of(99_999, cfg) == "micro"
    assert tier_of(100_000, cfg) == "mid"
    assert tier_of(499_999, cfg) == "mid"
    assert tier_of(500_000, cfg) == "macro"


def test_recent_videos_keeps_newest_first_and_undated_last():
    kol = _kol(
        "recent.videos",
        videos=[
            _video(days_ago=30),
            _video(days_ago=None),
            _video(days_ago=1),
            _video(days_ago=10),
        ],
    )

    result = recent_videos(kol, MatchingConfig(recent_videos=4))

    assert [video.posted_at for video in result[:3]] == [
        NOW - timedelta(days=1),
        NOW - timedelta(days=10),
        NOW - timedelta(days=30),
    ]
    assert result[3].posted_at is None
    assert [
        video.posted_at
        for video in recent_videos(kol, MatchingConfig(recent_videos=2))
    ] == [NOW - timedelta(days=1), NOW - timedelta(days=10)]


def test_on_topic_share_uses_normalized_caption_terms():
    brief = _brief()
    kol = _kol(
        "mixed.niche",
        videos=[
            _video(caption="รีวิว Gaming-Glasses สำหรับเกมเมอร์"),
            _video(caption="cooking recipe for dinner"),
        ],
    )

    assert on_topic_share(kol, brief, DEFAULT_CONFIG) == pytest.approx(0.5)
    assert on_topic_share(_kol("empty"), brief, DEFAULT_CONFIG) == 0.0


def test_percentile_ranks_use_mid_ranks():
    assert percentile_ranks([]) == []
    assert percentile_ranks([0.7]) == [0.5]
    assert percentile_ranks([10, 20, 20, 30]) == pytest.approx(
        [0.125, 0.5, 0.5, 0.875]
    )


def test_higher_engagement_earns_higher_e():
    low = _kol(
        "low.er",
        videos=[_video(views=1_000, likes=20) for _ in range(5)],
    )
    high = _kol(
        "high.er",
        videos=[_video(views=1_000, likes=60) for _ in range(5)],
    )

    results = engagement([low, high], DEFAULT_CONFIG)

    assert results[0].E != results[1].E
    assert results[1].E > results[0].E


def test_shrinkage_pulls_a_single_video_closer_to_the_tier_prior():
    pool = [
        _kol(f"pool.{index}", videos=[_video(likes=10)])
        for index in range(10)
    ]
    one_video = _kol("one.video", videos=[_video(likes=100)])
    ten_videos = _kol(
        "ten.videos",
        videos=[_video(likes=100) for _ in range(10)],
    )

    results = engagement([*pool, one_video, ten_videos], DEFAULT_CONFIG)
    one_result = results[-2]
    ten_result = results[-1]
    prior = 0.01  # the ten pool creators all have a median ER of 1%.

    assert abs(one_result.er_star - prior) < abs(ten_result.er_star - prior)


def test_er_percentile_is_within_tier_not_global():
    macro_best = _kol(
        "macro.best",
        followers=1_000_000,
        videos=[_video(views=100_000, likes=3_000)],
    )
    macro_others = [
        _kol(
            f"macro.{index}",
            followers=1_000_000,
            videos=[_video(views=100_000, likes=likes)],
        )
        for index, likes in enumerate((1_000, 2_000, 2_500))
    ]
    nano_median = _kol(
        "nano.median",
        followers=5_000,
        videos=[_video(views=500, likes=30)],
    )
    nano_others = [
        _kol(
            f"nano.{index}",
            followers=5_000,
            videos=[_video(views=500, likes=likes)],
        )
        for index, likes in enumerate((20, 40))
    ]

    kols = [macro_best, *macro_others, nano_median, *nano_others]
    results = engagement(kols, DEFAULT_CONFIG)
    by_handle = {kol.handle: result for kol, result in zip(kols, results)}

    assert by_handle["macro.best"].E >= by_handle["nano.median"].E


def test_tier_fit_peaks_at_target_and_low_confidence_is_flatter():
    target = 100_000

    assert tier_fit(target, target, 0.9) == pytest.approx(1.0)
    assert tier_fit(10_000, target, 0.9) < 1.0
    assert tier_fit(1_000_000, target, 0.9) < 1.0
    assert tier_fit(target * 10, target, 0.2) > tier_fit(
        target * 10, target, 0.9
    )


@pytest.mark.parametrize("audience", [0.0, 0.3, 1.0])
def test_audience_eff_is_neutral_without_confidence(audience):
    assert audience_eff(audience, 0.0) == pytest.approx(0.5)


def test_off_niche_creator_cannot_win_on_engagement_alone():
    brief = _brief(target_followers=50_000)
    on_niche = _kol(
        "on.niche",
        videos=[
            _video(views=1_000, likes=50, caption="gaming glasses review")
            for _ in range(5)
        ],
    )
    off_niche = _kol(
        "off.niche",
        videos=[
            _video(views=1_000, likes=200, caption="cooking recipe")
            for _ in range(5)
        ],
    )

    results = score_pool(
        [on_niche, off_niche],
        brief,
        [0.9, 0.1],
        None,
        _brand_aliases(brief),
        _competitor_aliases(brief),
        DEFAULT_CONFIG,
        NOW,
    )
    by_handle = {result.kol.handle: result for result in results}

    assert by_handle["off.niche"].scores.E > by_handle["on.niche"].scores.E
    assert by_handle["off.niche"].final < by_handle["on.niche"].final


def test_mention_bonus_adds_without_penalizing_unmentioned_creators():
    cfg = DEFAULT_CONFIG
    unmentioned = fit(0.5, 0.5, 0.5, False, cfg)
    mentioned = fit(0.5, 0.5, 0.5, True, cfg)

    assert unmentioned == pytest.approx(0.5)
    assert mentioned == pytest.approx(unmentioned + cfg.mention_bonus)
    assert fit(1.0, 1.0, 1.0, True, cfg) == pytest.approx(1.0)


def test_mmr_prefers_a_distinct_creator_over_a_near_duplicate():
    finals = [1.0, 0.99, 0.98]

    def sim(left: int, right: int) -> float:
        return 0.98 if {left, right} == {0, 1} else 0.0

    assert mmr_select(finals, sim, 2, 0.7) == [0, 2]


def test_competitor_sponsorship_is_flagged_but_not_penalized():
    brief = _brief()
    brand_aliases = _brand_aliases(brief)
    competitor_aliases = _competitor_aliases(brief)
    plain = _kol(
        "plain.mention",
        videos=[_video(days_ago=3, caption="OtherLens new drop")],
    )
    sponsored = _kol(
        "sponsored.mention",
        videos=[_video(days_ago=3, caption="#ad OtherLens new drop")],
    )

    plain_flags = flags_for(
        plain,
        brief,
        brand_aliases,
        competitor_aliases,
        0.1,
        DEFAULT_CONFIG,
        NOW,
    )
    sponsored_flags = flags_for(
        sponsored,
        brief,
        brand_aliases,
        competitor_aliases,
        0.1,
        DEFAULT_CONFIG,
        NOW,
    )

    assert "mentions_competitor" in sponsored_flags
    assert "competitor_sponsored" in sponsored_flags
    assert "competitor_sponsored" not in plain_flags

    plain_scored = score_pool(
        [plain],
        brief,
        [0.5],
        None,
        brand_aliases,
        competitor_aliases,
        DEFAULT_CONFIG,
        NOW,
    )[0]
    sponsored_scored = score_pool(
        [sponsored],
        brief,
        [0.5],
        None,
        brand_aliases,
        competitor_aliases,
        DEFAULT_CONFIG,
        NOW,
    )[0]

    assert sponsored_scored.scores.F == pytest.approx(plain_scored.scores.F)


def test_score_pool_uses_judgements_and_marks_risky_content():
    brief = _brief()
    kol = _kol("judged.creator", videos=[_video(caption="gaming glasses")])
    judgement = Judgement(
        handle="@Judged.Creator",
        judge=0.9,
        audience=0.8,
        audience_confidence=0.5,
        reason="On topic and a good tone.",
        risky=True,
    )

    result = score_pool(
        [kol],
        brief,
        [0.5],
        {"judged.creator": judgement},
        _brand_aliases(brief),
        _competitor_aliases(brief),
        DEFAULT_CONFIG,
        NOW,
    )[0]

    assert result.scores.judged is True
    assert result.scores.judge == pytest.approx(0.9)
    assert result.scores.A_eff == pytest.approx(0.65)
    assert result.scores.A_conf == pytest.approx(0.5)
    assert result.scores.F_conf == pytest.approx(
        DEFAULT_CONFIG.f_conf_judged
    )
    assert "risky_content" in result.flags
    assert result.final == pytest.approx(result.scores.F * result.quality)


def test_quality_combines_the_three_configured_weights():
    cfg = DEFAULT_CONFIG

    assert quality(0.8, 0.6, 0.4, cfg) == pytest.approx(
        cfg.quality_w_e * 0.8
        + cfg.quality_w_t * 0.6
        + cfg.quality_w_a * 0.4
    )
