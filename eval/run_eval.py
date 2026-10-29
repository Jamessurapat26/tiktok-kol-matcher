#!/usr/bin/env python3
"""Offline ranking evaluation against draft golden labels.

Ranking is measured by the Final score only: MMR diversification is NOT
applied here, because this harness evaluates scoring rather than the final
diverse presentation. No network, paid API, or model download is used.

Run it as either::

    cd backend && uv run python ../eval/run_eval.py
    python eval/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_BACKEND_ERROR: ModuleNotFoundError | None = None
try:
    from app.integrations.embeddings import MockEmbedder
    from app.integrations.llm import MockBrandAnalyzer
    from app.matching_config import DEFAULT_CONFIG
    from app.models import BrandProfile, KolProfile, normalize_handle
    from app.ranking import rank_candidates
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the caller
    _BACKEND_ERROR = exc

_CRITERIA = ("F", "E", "T", "A")
_ABLATION_DESCRIPTIONS = {
    "F": "Fit set to 0.5",
    "E": "Engagement set to 0.5",
    "T": "Tier fit set to 0.5",
    "A": "Audience fit set to 0.5",
}
_TOP_N = 10


def _require_backend() -> bool:
    """Return True when backend imports are available, else explain once.

    Called from a plain ``python`` without the backend's dependencies, the
    script re-runs itself inside the backend's uv environment.
    """
    if _BACKEND_ERROR is None:
        return True
    uv = shutil.which("uv")
    if uv is not None and not os.environ.get("RUN_EVAL_REEXEC"):
        os.environ["RUN_EVAL_REEXEC"] = "1"
        os.execv(
            uv,
            [uv, "run", "--project", str(_BACKEND), "python", __file__, *sys.argv[1:]],
        )
    print(
        f"error: backend dependency {_BACKEND_ERROR.name!r} is missing; run "
        "`cd backend && uv run python ../eval/run_eval.py`",
        file=sys.stderr,
    )
    return False


def _dcg(gains: list[int]) -> float:
    """Discounted cumulative gain with a log2(rank + 1) discount."""
    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )


def ndcg_at_10(ranked_handles: list[str], labels: dict[str, int]) -> float:
    """NDCG@10 with gain 2**label - 1; labels are already normalized handles.

    Labels not present in ``labels`` count as gain 0. The ideal DCG uses the
    golden labels of handles in the pool, sorted best first.
    """
    gains = [
        2 ** labels.get(handle, 0) - 1
        for handle in ranked_handles[:_TOP_N]
    ]
    ideal_labels = sorted(labels.values(), reverse=True)[:_TOP_N]
    ideal_gains = [2**label - 1 for label in ideal_labels]
    ideal = _dcg(ideal_gains)
    if ideal <= 0.0:
        return 0.0
    return _dcg(gains) / ideal


def _load_golden(path: Path) -> dict[str, int]:
    """Read one JSONL golden file into normalized-handle -> label."""
    labels: dict[str, int] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        label = int(row["label"])
        if label not in (0, 1, 2):
            raise ValueError(f"{path}:{line_no}: label must be 0, 1, or 2")
        labels[normalize_handle(str(row["handle"]))] = label
    return labels


def _load_pool(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate(pool: dict[str, Any], golden: dict[str, int]) -> dict[str, Any]:
    """Rank one cached pool and compute its baseline NDCG@10."""
    brief = BrandProfile.model_validate(pool["brief"])
    candidates = [
        KolProfile.model_validate(raw) for raw in pool["candidates"]
    ]
    now = datetime.fromisoformat(str(pool["now"]).replace("Z", "+00:00"))
    embedder = MockEmbedder()
    judge = MockBrandAnalyzer()

    outcome = rank_candidates(
        brief,
        candidates,
        embedder=embedder,
        judge_fn=judge.judge,
        brand_aliases=pool["brand_aliases"],
        competitor_aliases=pool["competitor_aliases"],
        cfg=DEFAULT_CONFIG,
        now=now,
    )
    ranked = sorted(
        outcome.ranked,
        key=lambda item: (-item.final, normalize_handle(item.kol.handle)),
    )
    pool_handles = {normalize_handle(kol.handle) for kol in candidates}
    labels = {
        handle: label
        for handle, label in golden.items()
        if handle in pool_handles
    }
    ranked_handles = [
        normalize_handle(item.kol.handle) for item in ranked
    ]
    return {
        "brand": str(pool["brand"]),
        "n_pool": len(candidates),
        "n_labeled": len(labels),
        "score": ndcg_at_10(ranked_handles, labels),
        "ranked": ranked,
        "labels": labels,
        "handle_order": ranked_handles,
    }


def _ablation_order(
    ranked: list[Any],
    criteria: set[str],
) -> list[str]:
    """Re-sort by Final after replacing the selected scores with 0.5."""
    rows: list[tuple[float, str]] = []
    for item in ranked:
        scores = item.scores
        fit = 0.5 if "F" in criteria else scores.F
        engagement = 0.5 if "E" in criteria else scores.E
        tier = 0.5 if "T" in criteria else scores.T
        audience = 0.5 if "A" in criteria else scores.A_eff
        quality = (
            DEFAULT_CONFIG.quality_w_e * engagement
            + DEFAULT_CONFIG.quality_w_t * tier
            + DEFAULT_CONFIG.quality_w_a * audience
        )
        rows.append(
            (fit * quality, normalize_handle(item.kol.handle))
        )
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [handle for _, handle in rows]


def _print_baseline(results: list[dict[str, Any]]) -> float:
    print(f"{'brand':<24} {'n_pool':>7} {'n_labeled':>9} {'NDCG@10':>8}")
    for result in results:
        print(
            f"{result['brand']:<24} {result['n_pool']:>7} "
            f"{result['n_labeled']:>9} {result['score']:>8.4f}"
        )
    mean = statistics.fmean(result["score"] for result in results)
    print(f"{'mean':<24} {'':>7} {'':>9} {mean:>8.4f}")
    return mean


def _print_ablation(
    results: list[dict[str, Any]],
    baseline_mean: float,
    criteria: list[str],
) -> None:
    print()
    print(
        "Ablation: set score to 0.5, recompute "
        "Final = F x (qe*E + qt*T + qa*A_eff)"
    )
    means: dict[str, float] = {}
    for criterion in criteria:
        scores: list[float] = []
        print()
        print(f"[{criterion}] {_ABLATION_DESCRIPTIONS[criterion]}")
        print(
            f"{'brand':<24} {'baseline':>9} {'ablated':>8} {'delta':>8}"
        )
        for result in results:
            order = _ablation_order(result["ranked"], {criterion})
            score = ndcg_at_10(order, result["labels"])
            scores.append(score)
            print(
                f"{result['brand']:<24} {result['score']:>9.4f} "
                f"{score:>8.4f} {score - result['score']:>+8.4f}"
            )
        mean = statistics.fmean(scores)
        means[criterion] = mean
        print(
            f"{'mean':<24} {baseline_mean:>9.4f} "
            f"{mean:>8.4f} {mean - baseline_mean:>+8.4f}"
        )

    no_lower = [
        criterion
        for criterion in criteria
        if means[criterion] >= baseline_mean - 1e-12
    ]
    print()
    print(
        "candidates for deletion: "
        + (", ".join(no_lower) if no_lower else "none")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            "Score cached candidate pools with NDCG@10 against draft golden "
            "labels. Final score only: MMR diversification is NOT applied."
        ),
        epilog=(
            "Ablation sets one score to 0.5, recomputes "
            "Final = F x (qe*E + qt*T + qa*A_eff), and reports the NDCG "
            "change. A criterion that does not lower mean NDCG is a candidate "
            "for deletion."
        ),
    )
    parser.add_argument(
        "--ablate",
        action="append",
        choices=(*_CRITERIA, "all"),
        default=[],
        metavar="{F,E,T,A}",
        help="ablate a score (repeatable) or use 'all'",
    )
    return parser


def _selected_criteria(values: list[str]) -> list[str]:
    if not values:
        return []
    if "all" in values:
        return list(_CRITERIA)
    selected: list[str] = []
    for value in values:
        if value not in selected:
            selected.append(value)
    return selected


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not _require_backend():
        return 2

    golden_dir = _EVAL_DIR / "golden"
    pool_dir = _EVAL_DIR / "pools"
    golden_paths = sorted(golden_dir.glob("*.jsonl"))
    if not golden_paths:
        print(f"error: no golden files in {golden_dir}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for golden_path in golden_paths:
        pool_path = pool_dir / f"{golden_path.stem}.json"
        if not pool_path.exists():
            print(
                f"skip {golden_path.name}: missing {pool_path}",
                file=sys.stderr,
            )
            continue
        results.append(_evaluate(_load_pool(pool_path), _load_golden(golden_path)))

    if not results:
        print("error: no golden file had a matching cached pool", file=sys.stderr)
        return 1

    baseline_mean = _print_baseline(results)
    criteria = _selected_criteria(args.ablate)
    if criteria:
        _print_ablation(results, baseline_mean, criteria)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
