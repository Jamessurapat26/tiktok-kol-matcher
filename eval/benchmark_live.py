#!/usr/bin/env python3
"""Benchmark content analysis on real brands, spending Apify money only once.

``capture`` runs the live graph for one brand and saves everything. ``compare``
re-judges those saved creators with the live LLM and never calls Apify:

* ``before`` shows the judge 3 captions with no hashtags or ad labels.
* ``after`` shows up to 10 captions with both, as the graph does now.

Each runs twice, so a change can be read against the model's own run-to-run
noise. ``sheet`` writes a blind labeling CSV, and ``score`` reports NDCG@10 and
style agreement once a person has filled it in. Without human labels this tool
can show that a ranking changed, not that it improved.

Output and caches stay under ``data/cache/benchmark``, which git ignores, because
they hold scraped creator data.

    set -a; . ./.env; set +a
    cd backend && uv run python ../eval/benchmark_live.py capture akhaama \\
        --website https://www.akhaamacoffee.com/ \\
        --facebook https://www.facebook.com/akhaamacoffee
    uv run python ../eval/benchmark_live.py compare akhaama
    uv run python ../eval/benchmark_live.py sheet akhaama
    uv run python ../eval/benchmark_live.py score akhaama
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app import prompts  # noqa: E402
from app.brand_match import brand_aliases, normalized_aliases  # noqa: E402
from app.graph import build_graph, initial_state  # noqa: E402
from app.integrations import build_integrations  # noqa: E402
from app.integrations.embeddings_live import SentenceTransformerEmbedder  # noqa: E402
from app.integrations.llm_live import LiveBrandAnalyzer  # noqa: E402
from app.matching_config import DEFAULT_CONFIG  # noqa: E402
from app.models import (  # noqa: E402
    BrandProfile,
    KolProfile,
    MatchRequest,
    RankedKol,
    normalize_handle,
)
from app.ranking import rank_candidates  # noqa: E402

_ROOT = _REPO_ROOT / "data" / "cache" / "benchmark"
_RUNS = _ROOT / "runs"
_REUSED_CACHES = ("web", "facebook", "embeddings")
_BEFORE_VIDEOS = 3
_TOP_N = 10
_ORDERS = ("before_1", "before_2", "after_1", "after_2")

# A capture starts only if the account's real spend plus this estimate fits under
# the cap. Apify's own $5 monthly limit is the backstop.
_SPEND_CAP_USD = 4.75
_CAPTURE_ESTIMATE_USD = 1.55


def _apify_usage_usd() -> float:
    response = httpx.get(
        "https://api.apify.com/v2/users/me/limits",
        headers={"Authorization": f"Bearer {os.environ['APIFY_TOKEN']}"},
        timeout=30,
    )
    response.raise_for_status()
    return float(response.json()["data"]["current"]["monthlyUsageUsd"])


def _use_benchmark_caches() -> None:
    """Keep page and embedding caches, but fetch profiles again for the new fields."""
    os.environ["KOL_CACHE_DIR"] = str(_ROOT)
    for name in _REUSED_CACHES:
        source = _REPO_ROOT / "data" / "cache" / name
        target = _ROOT / name
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)


def _run_path(slug: str, kind: str = "run") -> Path:
    return _RUNS / f"{slug}.{kind}.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def capture(args: argparse.Namespace) -> int:
    used = _apify_usage_usd()
    if used + _CAPTURE_ESTIMATE_USD > _SPEND_CAP_USD:
        print(
            f"refused: ${used:.2f} used + ${_CAPTURE_ESTIMATE_USD:.2f} estimate is "
            f"over the ${_SPEND_CAP_USD:.2f} cap",
            file=sys.stderr,
        )
        return 2
    _use_benchmark_caches()
    integrations = build_integrations(
        live_web=True,
        live_fb=True,
        live_kols=True,
        live_llm=True,
        live_embed=True,
        live_search=True,
    )
    request = MatchRequest(
        website_url=args.website,
        facebook_url=args.facebook,
        top_k=_TOP_N,
    )
    started = time.time()
    final = build_graph(integrations).invoke(initial_state(request))
    seconds = round(time.time() - started)
    # Apify's usage figure lags behind the last run.
    time.sleep(30)
    after = _apify_usage_usd()
    _write_json(
        _run_path(args.slug),
        {
            "slug": args.slug,
            "website": args.website,
            "facebook": args.facebook,
            "now": datetime.now(UTC).isoformat(),
            "seconds": seconds,
            "apify_usd": round(after - used, 4),
            "stage": final["stage"],
            "errors": final["errors"],
            "cost_log": final["cost_log"],
            "rejected": final["rejected"],
            "brand_research": final.get("brand_research"),
            "brand_profile": _dump(final.get("brand_profile")),
            "candidates": [_dump(kol) for kol in final["candidates"]],
            "ranked": [_dump(row) for row in final["ranked"]],
        },
    )
    print(
        f"{args.slug}: stage={final['stage']} {seconds}s "
        f"apify=${after - used:.3f} total=${after:.3f} "
        f"candidates={len(final['candidates'])} ranked={len(final['ranked'])}"
    )
    for error in final["errors"]:
        print(f"  warning: {error[:160]}")
    return 0 if final["stage"] == "done" else 1


def _before_view(kol: KolProfile) -> KolProfile:
    videos = [
        video.model_copy(update={"hashtags": [], "is_ad": False})
        for video in kol.videos
    ]
    return kol.model_copy(update={"videos": videos})


def _rank(
    run: dict[str, Any],
    analyzer: LiveBrandAnalyzer,
    embedder: SentenceTransformerEmbedder,
    *,
    before: bool,
) -> list[RankedKol]:
    brief = BrandProfile.model_validate(run["brand_profile"])
    candidates = [KolProfile.model_validate(kol) for kol in run["candidates"]]

    def judge(brief_: BrandProfile, kols: list[KolProfile]):
        if not before:
            return analyzer.judge(brief_, kols)
        limit = prompts.MAX_JUDGE_VIDEOS
        prompts.MAX_JUDGE_VIDEOS = _BEFORE_VIDEOS
        try:
            return analyzer.judge(brief_, [_before_view(kol) for kol in kols])
        finally:
            prompts.MAX_JUDGE_VIDEOS = limit

    outcome = rank_candidates(
        brief,
        candidates,
        embedder=embedder,
        judge_fn=judge,
        brand_aliases=brand_aliases(brief, run["website"], run["facebook"]),
        competitor_aliases=normalized_aliases(brief.competitors),
        cfg=DEFAULT_CONFIG,
        now=datetime.fromisoformat(run["now"]),
    )
    if outcome.errors:
        print(f"  judge warning: {outcome.errors}", file=sys.stderr)
    return outcome.ranked


def _handles(rows: list[dict[str, Any]], n: int = _TOP_N) -> list[str]:
    return [row["kol"]["handle"] for row in rows[:n]]


def _overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    return len(set(_handles(left)) & set(_handles(right)))


def _paired(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    read: Any,
) -> tuple[list[float], list[float]]:
    right_by_handle = {row["kol"]["handle"]: row for row in right}
    xs: list[float] = []
    ys: list[float] = []
    for row in left:
        other = right_by_handle.get(row["kol"]["handle"])
        if other is None:
            continue
        x, y = read(row), read(other)
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return round(statistics.correlation(xs, ys, method="ranked"), 3)


def _judge_score(row: dict[str, Any]) -> float | None:
    return row["scores"]["judge"] if row["scores"]["judged"] else None


def _style_fit(row: dict[str, Any]) -> float | None:
    return (row.get("content") or {}).get("style_fit")


def _agreement(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    judge_x, judge_y = _paired(left, right, _judge_score)
    return {
        "top10_overlap": _overlap(left, right),
        "final_spearman": _spearman(
            *_paired(left, right, lambda row: row["final"])
        ),
        "judge_spearman": _spearman(judge_x, judge_y),
        "judge_mean_abs_diff": (
            round(statistics.fmean(abs(x - y) for x, y in zip(judge_x, judge_y)), 3)
            if judge_x
            else None
        ),
    }


def _data_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the judge actually had to read for the creators it judged."""
    judged = [row for row in rows if row["scores"]["judged"]]
    videos = [video for row in judged for video in row["kol"]["videos"]]
    words: list[int] = []
    for video in videos:
        text = " ".join(
            word for word in video["caption"].split() if not word.startswith("#")
        )
        words.append(len(text))
    ad_shares = [
        sum(video.get("is_ad", False) for video in row["kol"]["videos"])
        / len(row["kol"]["videos"])
        for row in judged
        if row["kol"]["videos"]
    ]
    return {
        "judged": len(judged),
        "videos_per_creator": round(len(videos) / max(len(judged), 1), 1),
        "captions_empty_share": round(
            sum(not video["caption"].strip() for video in videos) / max(len(videos), 1), 3
        ),
        "captions_under_20_chars_without_hashtags": round(
            sum(length < 20 for length in words) / max(len(words), 1), 3
        ),
        "median_chars_without_hashtags": statistics.median(words) if words else 0,
        "hashtags_per_video": round(
            statistics.fmean(len(video.get("hashtags", [])) for video in videos), 1
        )
        if videos
        else 0,
        "ad_labelled_video_share": round(
            sum(video.get("is_ad", False) for video in videos) / max(len(videos), 1), 3
        ),
        "creators_with_any_ad_label": sum(share > 0 for share in ad_shares),
        "creators_over_half_ad_labelled": sum(share > 0.5 for share in ad_shares),
    }


def compare(args: argparse.Namespace) -> int:
    _use_benchmark_caches()
    # Nothing here builds an Apify client.
    run = json.loads(_run_path(args.slug).read_text(encoding="utf-8"))
    analyzer = LiveBrandAnalyzer()
    embedder = SentenceTransformerEmbedder.from_env()
    results: dict[str, list[dict[str, Any]]] = {}
    for name in _ORDERS:
        started = time.time()
        rows = _rank(run, analyzer, embedder, before=name.startswith("before"))
        results[name] = [_dump(row) for row in rows]
        print(f"  {name}: {time.time() - started:.0f}s", file=sys.stderr)

    after = results["after_1"]
    fits = [fit for fit in map(_style_fit, after) if fit is not None]
    judged_after = [row for row in after if row["scores"]["judged"]]
    report = {
        "slug": args.slug,
        "data": _data_quality(after),
        "noise_before": _agreement(results["before_1"], results["before_2"]),
        "noise_after": _agreement(results["after_1"], results["after_2"]),
        "before_vs_after": _agreement(results["before_1"], results["after_1"]),
        "style_fit": {
            "with_score": len(fits),
            "judged": len(judged_after),
            "min": min(fits, default=None),
            "max": max(fits, default=None),
            "stdev": round(statistics.pstdev(fits), 3) if fits else None,
            "repeat_spearman": _spearman(
                *_paired(results["after_1"], results["after_2"], _style_fit)
            ),
            "vs_judge_spearman": _spearman(
                [float(_style_fit(row)) for row in judged_after if _style_fit(row) is not None],
                [float(row["scores"]["judge"]) for row in judged_after if _style_fit(row) is not None],
            ),
        },
    }
    _write_json(_run_path(args.slug, "compare"), {"report": report, "results": results})
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def _sheet_path(slug: str) -> Path:
    return _ROOT / "labels" / f"{slug}.csv"


def sheet(args: argparse.Namespace) -> int:
    """Write every creator either version ranked in its top 10, without scores."""
    data = json.loads(_run_path(args.slug, "compare").read_text(encoding="utf-8"))
    results = data["results"]
    rows: dict[str, dict[str, Any]] = {}
    for name in ("before_1", "after_1"):
        for row in results[name][:_TOP_N]:
            rows.setdefault(row["kol"]["handle"], row)
    for row in results["after_1"]:
        if row["scores"]["judged"]:
            rows.setdefault(row["kol"]["handle"], row)
    path = _sheet_path(args.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.force:
        print(f"refused: {path} exists and may hold labels; use --force", file=sys.stderr)
        return 2
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["handle", "profile_url", "fit_0_1_2", "style_0_1_2", "note"])
        for name in sorted(rows, key=str.casefold):
            writer.writerow([name, rows[name]["profile_url"], "", "", ""])
    print(f"{path}: {len(rows)} creators to label")
    return 0


def _ndcg(rows: list[dict[str, Any]], labels: dict[str, int]) -> float:
    def dcg(gains: list[int]) -> float:
        return sum(
            gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1)
        )

    ranked = [
        2 ** labels.get(normalize_handle(row["kol"]["handle"]), 0) - 1
        for row in rows[:_TOP_N]
    ]
    ideal = sorted((2**label - 1 for label in labels.values()), reverse=True)[:_TOP_N]
    return dcg(ranked) / dcg(ideal) if dcg(ideal) else 0.0


def _read_labels(slug: str, column: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    with _sheet_path(slug).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row[column].strip()
            if value:
                labels[normalize_handle(row["handle"])] = int(value)
    return labels


def score(args: argparse.Namespace) -> int:
    results = json.loads(_run_path(args.slug, "compare").read_text(encoding="utf-8"))[
        "results"
    ]
    fit = _read_labels(args.slug, "fit_0_1_2")
    style = _read_labels(args.slug, "style_0_1_2")
    if not fit:
        print(f"error: no labels in {_sheet_path(args.slug)}", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "slug": args.slug,
        "labeled_fit": len(fit),
        "labeled_style": len(style),
        "ndcg_at_10": {name: round(_ndcg(results[name], fit), 3) for name in _ORDERS},
    }
    pairs = [
        (float(fit_value), float(style[normalize_handle(row["kol"]["handle"])]))
        for row in results["after_1"]
        if (fit_value := _style_fit(row)) is not None
        and normalize_handle(row["kol"]["handle"]) in style
    ]
    report["style_fit_vs_human_spearman"] = _spearman(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture", help="paid: run the live graph once")
    run.add_argument("slug")
    run.add_argument("--website", required=True)
    run.add_argument("--facebook", required=True)
    run.set_defaults(handler=capture)
    for name, handler, text in (
        ("compare", compare, "LLM only: re-judge the saved creators"),
        ("score", score, "no network: score a labeled sheet"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("slug")
        command.set_defaults(handler=handler)
    write = commands.add_parser("sheet", help="no network: write a labeling CSV")
    write.add_argument("slug")
    write.add_argument("--force", action="store_true")
    write.set_defaults(handler=sheet)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
