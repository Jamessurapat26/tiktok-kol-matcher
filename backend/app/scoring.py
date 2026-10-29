"""Pure matching-v2 scoring functions.

No I/O and no hidden constants: every tuning number comes from the supplied
``MatchingConfig``. The functions are deliberately small so the pipeline can
compose them without the old weighted-sum scorer.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from .brand_match import MIN_ALIAS_CHARS, normalize_for_match
from .matching_config import MatchingConfig
from .models import (
    BrandProfile,
    Judgement,
    KolProfile,
    Scores,
    VideoStats,
    normalize_handle,
)
from .topic_lexicon import expand_terms

# English hashtags must end at a tag boundary, so "#adidas" and "#adventure"
# are not "#ad". The Thai markers are substring matches exactly as the design
# lists them, because Thai tags run straight into the next word.
_SPONSOR_TAG_RE = re.compile(r"#(?:ad|sponsored)(?![a-z0-9_])")
_HASHTAG_RE = re.compile(r"#[^\s#]+")
_THAI_SPONSOR_MARKERS = (
    "#โฆษณา",
    "#รีวิว",
    "ได้รับการสนับสนุน",
)

# Neutral judge prior for creators left out of the judge stage.
_UNJUDGED_JUDGE_CAP = 0.5


@dataclass(frozen=True)
class EngagementResult:
    """Engagement score and the evidence used to build it."""

    E: float
    E_conf: float
    er_star: float
    reach: float | None
    n: int


@dataclass
class ScoredCreator:
    """One creator after every score has been combined."""

    kol: KolProfile
    scores: Scores
    quality: float
    final: float
    flags: list[str]


def tier_of(followers: int, cfg: MatchingConfig) -> str:
    """Return the follower tier: nano, micro, mid, or macro."""
    nano, micro, mid = cfg.tier_bounds
    if followers < nano:
        return "nano"
    if followers < micro:
        return "micro"
    if followers < mid:
        return "mid"
    return "macro"


def recent_videos(kol: KolProfile, cfg: MatchingConfig) -> list[VideoStats]:
    """Return the newest ``cfg.recent_videos`` videos, undated ones last."""
    dated = sorted(
        (video for video in kol.videos if video.posted_at is not None),
        key=lambda video: video.posted_at,
        reverse=True,
    )
    undated = [video for video in kol.videos if video.posted_at is None]
    return (dated + undated)[: cfg.recent_videos]


def _topic_terms(brief: BrandProfile) -> list[str]:
    """Return normalized brief terms plus lexicon terms for its categories.

    The brief's own terms come first. A category is only widened when the
    brief's category, key products, or topic keywords mention a lexicon
    trigger, so an unrelated brief is not expanded.
    """
    category_parts = brief.category.replace("/", ",").split(",")
    explicit = [
        *brief.topic_keywords_th,
        *brief.topic_keywords_en,
        *brief.key_products,
        *category_parts,
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for value in (*explicit, *expand_terms(explicit)):
        term = normalize_for_match(str(value))
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _video_text(video: VideoStats) -> str:
    """Return the normalized caption plus its hashtags.

    TikTok captions already contain their hashtags. Extracting the tags again
    keeps the match explicit and future-proof if a source ever returns tags
    separately from the caption body.
    """
    hashtags = " ".join(_HASHTAG_RE.findall(video.caption))
    return normalize_for_match(f"{video.caption} {hashtags}")


def video_matches_brief(video: VideoStats, brief: BrandProfile) -> bool:
    """True when one video's caption or hashtags names a brief topic term."""
    text = _video_text(video)
    return any(term in text for term in _topic_terms(brief))


def _bio_matches(kol: KolProfile, terms: Iterable[str]) -> bool:
    """True when the creator bio names any expanded brief topic term."""
    bio = normalize_for_match(kol.bio or "")
    return bool(bio) and any(term in bio for term in terms)


def on_topic_share(
    kol: KolProfile,
    brief: BrandProfile,
    cfg: MatchingConfig,
) -> float:
    """Share of recent videos that name an expanded brief topic term.

    Brief terms are widened with the built-in category lexicon. A bio mention
    is partial evidence when the recent videos are thin or use unrelated
    wording, so the result is ``max(video share, cfg.on_topic_bio_floor)``.
    """
    terms = _topic_terms(brief)
    if not terms:
        return 0.0

    videos = recent_videos(kol, cfg)
    video_share = 0.0
    if videos:
        texts = [_video_text(video) for video in videos]
        hits = sum(1 for text in texts if any(term in text for term in terms))
        video_share = hits / len(videos)

    bio_floor = cfg.on_topic_bio_floor if _bio_matches(kol, terms) else 0.0
    return max(video_share, bio_floor)


def percentile_ranks(values: list[float]) -> list[float]:
    """Mid-rank percentiles: (count_less + 0.5 * count_equal) / n."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    ranks: list[float] = []
    for value in values:
        less = sum(1 for other in values if other < value)
        equal = sum(1 for other in values if other == value)
        ranks.append((less + 0.5 * equal) / n)
    return ranks


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def engagement(
    kols: list[KolProfile],
    cfg: MatchingConfig,
) -> list[EngagementResult]:
    """Score engagement quality with shrinkage and within-tier percentiles."""
    medians: list[float | None] = []
    reaches: list[float | None] = []
    tiers: list[str] = []
    counts: list[int] = []

    for kol in kols:
        videos = recent_videos(kol, cfg)
        er_values = [
            (video.likes + video.comments + video.shares) / video.views
            for video in videos
            if video.views > 0
        ]
        medians.append(_median(er_values))
        counts.append(len(er_values))
        tiers.append(tier_of(kol.followers, cfg))
        if not videos or kol.followers <= 0:
            reaches.append(None)
        else:
            reaches.append(
                statistics.median(video.views for video in videos) / kol.followers
            )

    all_medians = [value for value in medians if value is not None]
    pool_median = _median(all_medians)
    if pool_median is None:
        pool_median = 0.0

    tier_priors: dict[str, float] = {}
    for tier in set(tiers):
        tier_medians = [
            value
            for value, item_tier in zip(medians, tiers)
            if item_tier == tier and value is not None
        ]
        tier_median = _median(tier_medians)
        tier_priors[tier] = pool_median if tier_median is None else tier_median

    er_stars: list[float] = []
    for median_er, tier, count in zip(medians, tiers, counts):
        prior = tier_priors[tier]
        if count == 0 or median_er is None:
            er_stars.append(prior)
        else:
            er_stars.append(
                (count * median_er + cfg.er_shrink_k * prior)
                / (count + cfg.er_shrink_k)
            )

    er_percentiles = [0.0] * len(kols)
    reach_percentiles = [0.5] * len(kols)
    tier_indices: dict[str, list[int]] = {}
    for index, tier in enumerate(tiers):
        tier_indices.setdefault(tier, []).append(index)

    for indices in tier_indices.values():
        ranks = percentile_ranks([er_stars[index] for index in indices])
        for index, rank in zip(indices, ranks):
            er_percentiles[index] = rank

        reach_indices = [index for index in indices if reaches[index] is not None]
        if reach_indices:
            ranks = percentile_ranks([reaches[index] for index in reach_indices])
            for index, rank in zip(reach_indices, ranks):
                reach_percentiles[index] = rank

    results: list[EngagementResult] = []
    for index, kol in enumerate(kols):
        confidence = (
            cfg.e_conf_profile if kol.stats_source == "profile" else cfg.e_conf_search
        )
        n = counts[index]
        results.append(
            EngagementResult(
                E=(
                    cfg.e_w_er * er_percentiles[index]
                    + cfg.e_w_reach * reach_percentiles[index]
                ),
                E_conf=(n / (n + cfg.er_shrink_k)) * confidence,
                er_star=er_stars[index],
                reach=reaches[index],
                n=n,
            )
        )
    return results


def tier_fit(
    followers: int,
    target_followers: int,
    tier_confidence: float,
) -> float:
    """How close a creator's follower count is to the brief's target tier."""
    sigma = 0.5 + 0.5 * (1 - tier_confidence)
    distance = abs(
        math.log10(max(followers, 1)) - math.log10(max(target_followers, 1))
    )
    return math.exp(-distance / sigma)


def audience_eff(audience: float, confidence: float) -> float:
    """Shrink an audience estimate toward neutral when the judge is unsure."""
    return confidence * audience + (1 - confidence) * 0.5


def fit(
    sim_pct: float,
    on_topic: float,
    judge: float,
    mentioned: bool,
    cfg: MatchingConfig,
) -> float:
    """Fit score: a weighted blend plus an optional, capped mention bonus."""
    value = (
        cfg.fit_w_sim * sim_pct
        + cfg.fit_w_on_topic * on_topic
        + cfg.fit_w_judge * judge
    )
    if mentioned:
        value += cfg.mention_bonus
    return min(1.0, value)


def quality(E: float, T: float, A_eff: float, cfg: MatchingConfig) -> float:
    """Combine engagement, tier, and audience into the quality multiplier."""
    return cfg.quality_w_e * E + cfg.quality_w_t * T + cfg.quality_w_a * A_eff


def _mention_terms(
    aliases: Iterable[str],
    full_names: Iterable[str],
) -> list[str]:
    """Normalized mention terms, keeping short aliases only as full names."""
    trusted_short = {normalize_for_match(str(name)) for name in full_names}
    terms: list[str] = []
    seen: set[str] = set()
    for raw in aliases:
        alias = normalize_for_match(str(raw))
        if not alias or alias in seen:
            continue
        if len(alias) < MIN_ALIAS_CHARS and alias not in trusted_short:
            continue
        seen.add(alias)
        terms.append(alias)
    return terms


def _kol_text(kol: KolProfile) -> str:
    parts = [kol.nickname, kol.bio, kol.content_summary]
    parts.extend(video.caption for video in kol.videos)
    return normalize_for_match(" ".join(part for part in parts if part))


def _mentions(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _is_sponsored(caption: str) -> bool:
    caption = caption.casefold()
    return bool(_SPONSOR_TAG_RE.search(caption)) or any(
        marker in caption for marker in _THAI_SPONSOR_MARKERS
    )


def _competitor_sponsored(
    kol: KolProfile,
    competitor_terms: Iterable[str],
    cfg: MatchingConfig,
    now: datetime,
) -> bool:
    cutoff = now - timedelta(days=cfg.active_days)
    for video in kol.videos:
        if video.posted_at is None or video.posted_at < cutoff:
            continue
        if not _is_sponsored(video.caption):
            continue
        if _mentions(normalize_for_match(video.caption), competitor_terms):
            return True
    return False


def flags_for(
    kol: KolProfile,
    brief: BrandProfile,
    brand_aliases: Iterable[str],
    competitor_aliases: Iterable[str],
    reach: float | None,
    cfg: MatchingConfig,
    now: datetime,
) -> list[str]:
    """Return the explainable flags for one creator."""
    flags: list[str] = []
    text = _kol_text(kol)
    brand_terms = _mention_terms(brand_aliases, brief.brand_names)
    competitor_terms = _mention_terms(competitor_aliases, brief.competitors)

    if _mentions(text, brand_terms):
        flags.append("mentions_brand")
    if _mentions(text, competitor_terms):
        flags.append("mentions_competitor")
    if _competitor_sponsored(kol, competitor_terms, cfg, now):
        flags.append("competitor_sponsored")
    if reach is not None and reach < cfg.low_reach_ratio:
        flags.append("low_reach")
    if not any(video.posted_at is not None for video in kol.videos):
        flags.append("activity_unknown")
    if kol.stats_source == "search":
        flags.append("search_stats_only")
    return flags


def score_pool(
    kols: list[KolProfile],
    brief: BrandProfile,
    sims: list[float],
    judgements: dict[str, Judgement] | None,
    brand_aliases: Iterable[str],
    competitor_aliases: Iterable[str],
    cfg: MatchingConfig,
    now: datetime,
) -> list[ScoredCreator]:
    """Score every creator in ``kols`` and return results in input order.

    Raw cosine similarities are converted to within-pool percentiles. Before
    stage 6 there is no LLM judge, so ``judge`` falls back to that same sim
    percentile and ``A_eff`` stays neutral. Once some creators are judged, an
    unjudged creator's ``judge`` is capped at a neutral 0.5, so a creator the
    judge never saw cannot outrank the judged leaders on sim alone.
    """
    sim_percentiles = percentile_ranks(sims)
    engagements = engagement(kols, cfg)

    judged_by_handle: dict[str, Judgement] = {}
    unjudged_cap = _UNJUDGED_JUDGE_CAP if judgements else 1.0
    if judgements:
        for key, judgement in judgements.items():
            judged_by_handle[normalize_handle(key)] = judgement
            judged_by_handle[normalize_handle(judgement.handle)] = judgement

    results: list[ScoredCreator] = []
    for kol, sim_pct, engagement_result in zip(
        kols, sim_percentiles, engagements
    ):
        judgement = judged_by_handle.get(normalize_handle(kol.handle))
        judged = judgement is not None

        if judged and judgement is not None:
            judge = judgement.judge
            a_eff = audience_eff(
                judgement.audience, judgement.audience_confidence
            )
            a_conf = judgement.audience_confidence
        else:
            judge = min(sim_pct, unjudged_cap)
            a_eff = 0.5
            a_conf = 0.0

        flags = flags_for(
            kol,
            brief,
            brand_aliases,
            competitor_aliases,
            engagement_result.reach,
            cfg,
            now,
        )
        if judged and judgement is not None and judgement.risky:
            flags.append("risky_content")

        on_topic = on_topic_share(kol, brief, cfg)
        mentioned = "mentions_brand" in flags or "mentions_competitor" in flags
        f = fit(sim_pct, on_topic, judge, mentioned, cfg)
        t = tier_fit(kol.followers, brief.target_followers, brief.tier_confidence)
        q = quality(engagement_result.E, t, a_eff, cfg)

        scores = Scores(
            F=f,
            E=engagement_result.E,
            T=t,
            A_eff=a_eff,
            F_conf=cfg.f_conf_judged if judged else cfg.f_conf_unjudged,
            E_conf=engagement_result.E_conf,
            T_conf=brief.tier_confidence,
            A_conf=a_conf,
            sim=sim_pct,
            on_topic=on_topic,
            judge=judge,
            er_star=engagement_result.er_star,
            reach=engagement_result.reach,
            judged=judged,
        )
        results.append(
            ScoredCreator(
                kol=kol,
                scores=scores,
                quality=q,
                final=f * q,
                flags=flags,
            )
        )
    return results


def mmr_select(
    finals: list[float],
    sim: Callable[[int, int], float],
    k: int,
    lam: float,
) -> list[int]:
    """Greedy MMR selection: relevance minus similarity to chosen items."""
    if k <= 0 or not finals:
        return []

    remaining = list(range(len(finals)))
    first = max(remaining, key=lambda index: (finals[index], -index))
    selected = [first]
    remaining.remove(first)

    while remaining and len(selected) < k:
        best_index = remaining[0]
        best_score = None
        for index in remaining:
            # Chosen item first: an embedder-backed ``sim`` then builds one
            # row per selected item instead of one per remaining candidate.
            max_sim = max(sim(chosen, index) for chosen in selected)
            score = lam * finals[index] - (1 - lam) * max_sim
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)

    return selected
