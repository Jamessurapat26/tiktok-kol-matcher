"""Every tuning number for the matching pipeline in one frozen config.

The scoring, filtering, and recall stages read these values instead of
scattering magic numbers through the code. ``DEFAULT_CONFIG`` is the single
module-level instance the rest of the app uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchingConfig:
    """Tuning constants for matching-v2.

    Weights are validated at construction so a bad config fails loudly instead
    of silently changing the ranking.
    """

    # Fit: F = sim + on-topic + judge (+ mention bonus).
    fit_w_sim: float = 0.4
    fit_w_on_topic: float = 0.3
    fit_w_judge: float = 0.3
    mention_bonus: float = 0.05

    # Partial on-topic credit when the bio matches but recent videos do not.
    on_topic_bio_floor: float = 0.3

    # Quality = E + T + A_eff.
    quality_w_e: float = 0.45
    quality_w_t: float = 0.30
    quality_w_a: float = 0.25

    # Engagement shrinkage and within-tier blend.
    er_shrink_k: int = 3
    e_w_er: float = 0.8
    e_w_reach: float = 0.2
    e_conf_search: float = 0.6
    e_conf_profile: float = 1.0
    low_reach_ratio: float = 0.02

    # Follower tier boundaries: nano <10k <= micro <100k <= mid <500k <= macro.
    tier_bounds: tuple[int, int, int] = (10_000, 100_000, 500_000)
    target_followers_min: int = 10_000
    target_followers_max: int = 2_000_000
    # Creator follower centre for each tier the brand analysis may pick. The
    # LLM chooses only the label; code maps it to a follower count.
    tier_target_nano: int = 10_000
    tier_target_micro: int = 30_000
    tier_target_mid: int = 150_000
    tier_target_macro: int = 800_000

    # Hard filters.
    followers_min: int = 1_000
    followers_max: int = 5_000_000
    active_days: int = 60

    # Stage sizes and selection.
    recent_videos: int = 10
    max_search_queries: int = 10
    enrich_top_n: int = 20
    judge_top_n: int = 20
    # Extra judge calls that replace accounts the judge removed.
    judge_refill_calls: int = 2
    mmr_lambda: float = 0.7

    # Confidence before and after the LLM judge.
    f_conf_unjudged: float = 0.6
    f_conf_judged: float = 0.9

    def __post_init__(self) -> None:
        self._require_unit_sum(
            "fit weights",
            (self.fit_w_sim, self.fit_w_on_topic, self.fit_w_judge),
        )
        self._require_unit_sum(
            "quality weights",
            (self.quality_w_e, self.quality_w_t, self.quality_w_a),
        )
        self._require_unit_sum(
            "engagement weights",
            (self.e_w_er, self.e_w_reach),
        )

        weights = (
            self.fit_w_sim,
            self.fit_w_on_topic,
            self.fit_w_judge,
            self.mention_bonus,
            self.quality_w_e,
            self.quality_w_t,
            self.quality_w_a,
            self.e_w_er,
            self.e_w_reach,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("config weights must be finite and non-negative")

        for name in (
            "e_conf_search",
            "e_conf_profile",
            "low_reach_ratio",
            "mmr_lambda",
            "f_conf_unjudged",
            "f_conf_judged",
            "on_topic_bio_floor",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        for name in (
            "er_shrink_k",
            "target_followers_min",
            "target_followers_max",
            "tier_target_nano",
            "tier_target_micro",
            "tier_target_mid",
            "tier_target_macro",
            "followers_min",
            "followers_max",
            "active_days",
            "recent_videos",
            "max_search_queries",
            "enrich_top_n",
            "judge_top_n",
            "judge_refill_calls",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.target_followers_min >= self.target_followers_max:
            raise ValueError("target_followers_min must be below target_followers_max")
        targets = self._tier_targets()
        if not (
            self.target_followers_min
            <= targets[0]
            < targets[1]
            < targets[2]
            < targets[3]
            <= self.target_followers_max
        ):
            raise ValueError(
                "tier targets must ascend within the target follower bounds"
            )
        if self.followers_min >= self.followers_max:
            raise ValueError("followers_min must be below followers_max")

        if (
            len(self.tier_bounds) != 3
            or any(
                not isinstance(bound, int) or isinstance(bound, bool)
                for bound in self.tier_bounds
            )
            or not (
                self.target_followers_min
                <= self.tier_bounds[0]
                < self.tier_bounds[1]
                < self.tier_bounds[2]
                <= self.followers_max
            )
        ):
            raise ValueError("tier_bounds must be three ascending follower counts")

    def _tier_targets(self) -> tuple[int, int, int, int]:
        return (
            self.tier_target_nano,
            self.tier_target_micro,
            self.tier_target_mid,
            self.tier_target_macro,
        )

    def target_followers_for(self, tier_label: str) -> int:
        """Follower centre for a tier label: nano, micro, mid, or macro."""
        targets = dict(zip(("nano", "micro", "mid", "macro"), self._tier_targets()))
        return targets[tier_label]

    @staticmethod
    def _require_unit_sum(name: str, values: tuple[float, ...]) -> None:
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{name} must be finite")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} must sum to 1")


DEFAULT_CONFIG = MatchingConfig()
