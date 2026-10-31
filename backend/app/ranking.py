"""Candidate ranking and diversity selection for matching-v2.

``rank_candidates`` is the reusable core of the old ``judge_and_rank`` node:
embed, score once, judge the first-pass leaders in one call, score again, and
build explainable ``RankedKol`` rows. ``diversify`` is the old MMR block from
``rank_and_output``. Both are pure apart from the injected embedder and judge.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from .integrations.embeddings import (
    Embedder,
    brand_query_text,
    kol_document_text,
    kol_title,
)
from .filters import MEDIA_ACCOUNT, SELLER_ACCOUNT
from .matching_config import MatchingConfig
from .models import (
    BrandProfile,
    Judgement,
    KolProfile,
    RankedKol,
    Scores,
    normalize_handle,
)
from .scoring import (
    mmr_select,
    recent_videos,
    score_pool,
    tier_of,
    video_matches_brief,
)

_MAX_EVIDENCE_URLS = 3
_REJECTED_ACCOUNT_TYPES = {"seller": SELLER_ACCOUNT, "media": MEDIA_ACCOUNT}
# The judge was asked about this creator but gave no answer.
UNREVIEWED_FLAG = "unreviewed"

JudgeFn = Callable[[BrandProfile, list[KolProfile]], list[Judgement]]


@dataclass
class RankOutcome:
    """The ranked creators plus what happened during the judge stage."""

    ranked: list[RankedKol]
    judged_count: int
    errors: list[str]
    rejected: dict[str, int]


def similarities(
    embedder: Embedder,
    brief: BrandProfile,
    kols: list[KolProfile],
) -> list[float]:
    """Return one brief-to-creator similarity per creator, in input order."""
    if not kols:
        return []
    return embedder.similarity(
        brand_query_text(brief),
        [kol_document_text(kol) for kol in kols],
        titles=[kol_title(kol) for kol in kols],
    )


def _evidence_urls(
    kol: KolProfile,
    brief: BrandProfile,
    cfg: MatchingConfig,
) -> list[str]:
    """Up to three high-view URLs, preferring on-topic recent videos."""
    videos = [
        video
        for video in recent_videos(kol, cfg)
        if video.url is not None
    ]
    on_topic = [
        video for video in videos if video_matches_brief(video, brief)
    ]
    chosen = sorted(on_topic or videos, key=lambda video: -video.views)
    return [str(video.url) for video in chosen[:_MAX_EVIDENCE_URLS]]


def _unjudged_reason(
    scores: Scores,
    kol: KolProfile,
    cfg: MatchingConfig,
) -> str:
    return (
        f"{scores.on_topic:.0%} ของวิดีโอล่าสุดตรงกับหัวข้อของแบรนด์ "
        f"การมีส่วนร่วม {scores.E * 100:.0f}/100 "
        f"เมื่อเทียบกับ KOL ขนาด {tier_of(kol.followers, cfg).capitalize()} "
        "(ยังไม่ได้ผ่านการประเมินโดย AI)"
    )


def _rank_key(item: RankedKol) -> tuple[float, str]:
    """Final score descending, then normalized handle ascending."""
    return (-item.final, normalize_handle(item.kol.handle))


def rank_candidates(
    brief: BrandProfile,
    candidates: list[KolProfile],
    *,
    embedder: Embedder,
    judge_fn: JudgeFn,
    brand_aliases: Iterable[str],
    competitor_aliases: Iterable[str],
    cfg: MatchingConfig,
    now: datetime,
    top_k: int = 0,
) -> RankOutcome:
    """Score, judge the first-pass leaders once, and rank every candidate.

    The judge sees at least ``top_k`` creators, because only reviewed creators
    can be recommended. A judge exception is recorded in ``errors`` and leaves every creator
    unjudged instead of aborting the run. ``judged_count`` is the number of
    creators sent to the judge, which is also what the cost log reports.
    Creators the judge calls a seller or media account leave the ranking and
    are counted in ``rejected``. Creators it was sent but did not answer for
    carry ``UNREVIEWED_FLAG``.
    """
    brand_alias_list = list(brand_aliases)
    competitor_alias_list = list(competitor_aliases)
    sims = similarities(embedder, brief, candidates)

    first_pass = score_pool(
        candidates,
        brief,
        sims,
        None,
        brand_alias_list,
        competitor_alias_list,
        cfg,
        now,
    )
    first_pass.sort(
        key=lambda item: (-item.final, normalize_handle(item.kol.handle))
    )
    to_judge = [item.kol for item in first_pass[: max(cfg.judge_top_n, top_k)]]

    errors: list[str] = []
    judgements: dict[str, Judgement] = {}
    if to_judge:
        try:
            for judgement in judge_fn(brief, to_judge):
                judgements[normalize_handle(judgement.handle)] = judgement
        except Exception as exc:  # noqa: BLE001 - judging is an optional stage
            errors.append(f"judge: {exc}")
        else:
            missing = [
                kol.handle
                for kol in to_judge
                if normalize_handle(kol.handle) not in judgements
            ]
            if missing:
                errors.append(
                    f"judge: no score for {len(missing)} of {len(to_judge)} "
                    f"creators ({', '.join(missing)}); ranked them unjudged"
                )

    rejected: dict[str, int] = {}
    kept: list[KolProfile] = []
    kept_sims: list[float] = []
    for kol, sim in zip(candidates, sims):
        judgement = judgements.get(normalize_handle(kol.handle))
        reason = (
            _REJECTED_ACCOUNT_TYPES.get(judgement.account_type)
            if judgement is not None
            else None
        )
        if reason is None:
            kept.append(kol)
            kept_sims.append(sim)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    sent_to_judge = {normalize_handle(kol.handle) for kol in to_judge}
    scored = score_pool(
        kept,
        brief,
        kept_sims,
        judgements,
        brand_alias_list,
        competitor_alias_list,
        cfg,
        now,
    )
    ranked: list[RankedKol] = []
    for item in scored:
        handle = normalize_handle(item.kol.handle)
        judgement = judgements.get(handle)
        unreviewed = judgement is None and handle in sent_to_judge
        reason = (
            judgement.reason
            if judgement is not None
            else _unjudged_reason(item.scores, item.kol, cfg)
        )
        ranked.append(
            RankedKol(
                kol=item.kol,
                scores=item.scores,
                quality=item.quality,
                final=item.final,
                flags=[*item.flags, UNREVIEWED_FLAG] if unreviewed else item.flags,
                reason=reason,
                evidence_urls=_evidence_urls(item.kol, brief, cfg),
                profile_url=(
                    str(item.kol.url)
                    or f"https://www.tiktok.com/@{normalize_handle(item.kol.handle)}"
                ),
            )
        )
    ranked.sort(key=_rank_key)
    return RankOutcome(
        ranked=ranked,
        judged_count=len(to_judge),
        errors=errors,
        rejected=rejected,
    )


def diversify(
    ranked: list[RankedKol],
    embedder: Embedder,
    top_k: int,
    cfg: MatchingConfig,
) -> list[RankedKol]:
    """Select a diverse top-k with greedy MMR over the ranked pool."""
    # Bound the MMR similarity work to a small pool around the requested k.
    pool = ranked[: max(3 * top_k, 30)]
    docs = [kol_document_text(item.kol) for item in pool]
    rows: dict[int, list[float]] = {}

    def row(index: int) -> list[float]:
        if index not in rows:
            rows[index] = embedder.similarity(
                docs[index],
                docs,
            )
        return rows[index]

    def sim(left: int, right: int) -> float:
        return row(left)[right]

    selected = mmr_select(
        [item.final for item in pool],
        sim,
        top_k,
        cfg.mmr_lambda,
    )
    return [pool[index] for index in selected]
