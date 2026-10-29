# Ranking evaluation harness

Offline, deterministic checks for the ranking scores. No network, paid API, or
model download is used: the pools are cached and the code runs the mock
embedder and mock brand analyzer.

## Files

- `golden/<brand-slug>.jsonl` — one creator per line:
  `{"brand": str, "handle": str, "label": 2|1|0, "reason": str, "labeled_by": "draft"|"human"}`
- `pools/<brand-slug>.json` — cached candidate pool:
  `brand`, `brief` (BrandProfile JSON), `brand_aliases`, `competitor_aliases`,
  `now` (fixed ISO timestamp), and `candidates` (KolProfile JSON after the
  mock hard filter and enrichment of the first-pass top `enrich_top_n`).
- `cache_pool.py` — rebuilds a pool from the fully mocked graph.
- `run_eval.py` — ranks each pool, reports NDCG@10, and runs ablations.

The evaluator ranks by the `Final` score only. MMR diversification is **not**
applied: this measures scoring quality, not the final diverse top-k.

## Label rubric

- `2` — good fit: on-niche gaming/eyewear/tech-review creator, active, and a
  sensible follower tier.
- `1` — acceptable: adjacent niche (office, student, esports, general tech), or
  the tier/tone is somewhat off.
- `0` — not a fit: off-niche (food, beauty, travel), brand/competitor/shop/media
  account, inactive, risky, or out of follower range.

Labels marked `"labeled_by": "draft"` are **not human-checked**; treat them as
starter labels to review.

## Run

```sh
cd backend && uv run python ../eval/run_eval.py
cd backend && uv run python ../eval/run_eval.py --ablate all
cd backend && uv run python ../eval/run_eval.py --ablate F --ablate E
```

From the repo root, `python eval/run_eval.py` also works when the backend
dependencies are importable. Otherwise it prints one line telling you to use
the `cd backend && uv run ...` form.

`--ablate {F,E,T,A}` sets that score to `0.5` for every creator, recomputes
`Final = F × (qe·E + qt·T + qa·A_eff)` with the config weights, re-sorts, and
reports the NDCG@10 delta per brand and on the mean. The last line lists
criteria whose ablation does not lower mean NDCG as deletion candidates.

## NDCG

Gain is `2^label - 1`; discount is `log2(rank + 1)` with 1-based ranks. The
ideal DCG uses the golden labels of handles present in the pool. A pool handle
with no golden label counts as gain `0`.

## Add a brand

1. Add `golden/<slug>.jsonl` using handles from the fixture/demo data.
2. Either hand-write `pools/<slug>.json` in the format above, or adapt
   `cache_pool.py` to run the mock graph for that brand and write the pool.
3. Keep `now` fixed in the pool so repeated runs are deterministic; video
   timestamps are absolute.
4. Run `cd backend && uv run python ../eval/run_eval.py`.
