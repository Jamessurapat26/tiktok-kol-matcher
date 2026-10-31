# TikTok KOL Matcher

Give it a brand's website and Facebook page, and it returns a ranked, explained
shortlist of Thai TikTok creators who fit the brand.

![Ranked results in demo mode](docs/screenshots/demo-overview.png)

KOL Matcher is a LangGraph pipeline behind a FastAPI service and a React web
app. It reads the brand's own pages, researches the brand on the web, has an LLM
build a structured brand brief, searches TikTok for creators, removes accounts
that cannot fit, and scores each survivor on four signals:

| Score | Meaning | How it is computed |
| --- | --- | --- |
| F | Fit | `0.4·sim + 0.3·on_topic + 0.3·judge`, plus a `0.05` mention bonus (F capped at 1). `sim` is the brief-to-caption embedding cosine as a pool percentile, `on_topic` the share of recent videos that name an expanded brief topic term (with a small bio fallback), and `judge` the LLM rubric score. |
| E | Engagement quality | Median per-video `(likes + comments + shares) / views`, shrunk toward the same-tier median (`k = 3`), then ranked within the follower tier. `E = 0.8·ER percentile + 0.2·reach percentile`, where reach is median views divided by followers. |
| T | Tier fit | `exp(−\|log10(followers) − log10(target_followers)\| / σ)` with `σ = 0.5 + 0.5·(1 − tier_confidence)`. |
| A_eff | Audience fit (estimate) | The judge's text-based audience estimate `A`, shrunk toward neutral by its confidence `c`: `A_eff = c·A + (1 − c)·0.5`. Before judging, `A_eff = 0.5`. |

`Final = F × Quality` and `Quality = 0.45·E + 0.30·T + 0.25·A_eff`.

Scoring runs only after hard filters remove the brand's own accounts,
competitor accounts, seller and media accounts, creators who are not Thai,
creators outside 1k–5M followers, and creators whose latest post is over 60
days old.

The API and CLI return F, E, T, A_eff, their confidences, flags, quality, final,
a one-line reason, up to three evidence video links, and the profile URL. Judged
creators also get an unscored content analysis: formats, style, and a style fit
that must cite a video. The web app is in Thai and written for marketers: it shows the scores out of 100, marks
low-confidence scores as limited data, and names flags in plain words (see
[GLOSSARY.md](GLOSSARY.md)). The whole pipeline also runs offline in demo mode,
with no keys.

## How this meets the brief

This prototype answers the Convert Cake AI Engineer case study.

| Brief | What the tool does | Limit |
| --- | --- | --- |
| Analyze a client's Facebook page and website | Reads both, researches the brand on the web, and has the LLM build a brief: category, products, audience, tone, content formats and style, keywords, competitors, and creator tier | Content style comes from text, not from watching videos |
| Identify and evaluate Thai TikTok KOLs | Searches TikTok, removes accounts that cannot fit, fetches recent videos, and has the LLM judge the leaders | Sellers are caught by rules and the judge, but some detection can still misfire; see [Known limitations](#known-limitations) |
| Rank by relevance, engagement, and audience demographics | Scores F, E, T, and A_eff, then selects with MMR | Audience is estimated from text. TikTok does not expose demographics to scrapers |
| Present a curated list with profile information | Returns profile links, followers, scores, a reason, flags, and evidence videos in the Thai web app, API, CLI, and CSV | |

## Contents

- [How this meets the brief](#how-this-meets-the-brief)
- [Screenshots](#screenshots)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Scoring](#scoring)
- [Live results](#live-results)
- [Business value](#business-value)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Ethics and privacy](#ethics-and-privacy)
- [Known limitations](#known-limitations)
- [Testing](#testing)
- [Next steps](#next-steps)
- [Repo layout](#repo-layout)
- [References](#references)
- [Reference: HTTP API and live integrations](docs/reference.md)

## Screenshots

These come from demo mode, so the brand and every result are sample data.

| Brand form | Progress |
| --- | --- |
| ![Brand form](docs/screenshots/demo-form.png) | ![Stepper mid-run, with the brand profile already shown](docs/screenshots/demo-progress.png) |

The brand profile appears as soon as `analyze_brand` finishes, while discovery
is still running. The web app folds the pipeline stages into five steps; the
API's event stream and the CLI keep the per-node detail.

| Score breakdown | Mobile |
| --- | --- |
| ![A result row with its component scores and content style](docs/screenshots/demo-score-breakdown.png) | <img src="docs/screenshots/demo-mobile.png" alt="Results on a phone-width screen" width="220"> |

The history sheet reopens earlier jobs,
and the job id stays in the URL as `?job=`, so a reload or a shared link opens
the same job.

| Demo results | History |
| --- | --- |
| ![Demo-mode results on fixture data](docs/screenshots/demo-results.png) | ![History sheet with recent jobs](docs/screenshots/demo-history.png) |

## Quick start

### Docker Compose

```bash
cp .env.example .env   # optional: add keys for live mode
docker compose up --build
```

Open http://localhost:5173. With no `.env`, or no `APIFY_TOKEN` in it, the
service starts in demo mode. Press **ลองดูตัวอย่าง** to fill the form with the
example brand. That brand exists only in the fixtures, so in live mode enter a
real website and Facebook page instead.

Both services bind to `127.0.0.1` only (frontend on 5173, API on 8000), because
the API has no authentication. nginx serves the built app, proxies `/api` to
the backend, and turns off buffering on the event stream. Jobs and caches live
in the `backend-data` volume. The backend runs as a non-root user with a health
check on `/api/health`.

Compose installs the local embedding packages, including the CPU build of
torch, so the backend image is about 2.16 GB. The model itself downloads into
the `backend-data` volume the first time a live job embeds. Demo mode does not
use it. A plain `docker build` of `backend/` leaves the packages out unless you
pass `--build-arg INSTALL_EMBED=true`.

### Local development

You need Python 3.13 with [uv](https://docs.astral.sh/uv/), and Node 24 with
pnpm 11.

`./start.sh` loads `.env` and runs both servers; Ctrl-C stops them. Add `--demo`
to force demo mode. To run them separately:

```bash
# terminal 1: API in demo mode on :8000
cd backend
uv sync                  # add --extra embed for the local embedding model
DEMO_MODE=1 uv run uvicorn app.api.main:app --port 8000

# terminal 2: Vite dev server on :5173, proxies /api to :8000
cd frontend
pnpm install
pnpm dev
```

The API reads its settings from the environment at startup. It does not load
`.env` itself, so export the variables or use Compose. Run it from `backend/`
so `data/jobs.db` lands there, or set `JOBS_DB`.

### CLI

The same graph runs in the foreground from the CLI. With no `--live-*` flag,
every service is mocked and nothing touches the network:

```bash
cd backend
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 10
```

Progress goes to stderr and the ranked table to stdout. The table shows F, E,
T, A_eff, their confidences, flags, and the profile URL. It also prints the
rejected counts and the per-stage cost log. `--json` prints one JSON object
instead. Add `--live-web`, `--live-fb`, `--live-kols`, `--live-llm`, `--live-embed`, or
`--live-search` to run that service for real. Cost flags:
`--max-results-per-query`, `--max-enrich` (overrides the config's
`enrich_top_n`, default 20), `--min-candidates`, and `--max-rounds`.

## Configuration

### Demo and live mode

`GET /api/health` reports the mode.

- **Demo mode** turns every live flag off, so jobs run on mocks and fixtures
  with no keys. It applies when `DEMO_MODE` is true, or when `DEMO_MODE` is
  unset and `APIFY_TOKEN` is missing.
- **Live mode** applies when `DEMO_MODE` is false, or when it is unset and
  `APIFY_TOKEN` is set. A flag the request omits turns on when it can run. A
  flag set to `true` that cannot run returns 400 and names the missing
  variable, for example `live_kols is unavailable: APIFY_TOKEN is not set`.

Each job stores the flags it actually ran with.

### LLM model

Unless `.env` says otherwise, the code uses DeepSeek on OpenRouter. Only the key
is required:

```bash
LLM_API_KEY=your-openrouter-key
```

`GET /api/health` shows the model and host in use. Jobs do not store the model,
so older results cannot be traced to one.

### Environment variables

[`.env.example`](.env.example) lists every variable the code reads. Blank
values take the defaults. Keys and tokens are never logged or put in error
messages, and `/api/health` reports each one only as set or missing.

| Variable | Purpose | Default |
| --- | --- | --- |
| `DEMO_MODE` | `1` forces demo, `0` forces live | unset: live only when `APIFY_TOKEN` is set |
| `APIFY_TOKEN` | Apify, for Facebook, TikTok search, and enrichment | unset |
| `APIFY_JOB_BUDGET_USD` | Estimated Apify budget for one job or CLI run | `1.55` |
| `APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD` | Per-run cap for each TikTok search | `0.50` |
| `APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD` | Per-run cap for each profile enrichment | `0.50` |
| `APIFY_FACEBOOK_MAX_CHARGE_USD` | Per-run cap for each Facebook page or posts run | `0.50` |
| `APIFY_ENRICH_RESERVE_USD` | Budget TikTok search must leave for enrichment, below the job budget | `0.70` |
| `TIKTOK_RESULTS_PER_QUERY` | Videos per TikTok search query, 1-50 | `15` |
| `TIKTOK_PROFILE_VIDEOS` | Recent videos fetched per enriched profile, 1-30 | `10` |
| `DISCOVERY_MIN_THAI_CANDIDATES` | Thai creators discovery aims for before it stops | `30` |
| `DISCOVERY_MAX_ROUNDS` | Most query and search rounds, 1-5 | `3` |
| `JINA_API_KEY` | Jina key: optional for the web reader (higher rate limit), required for web research on the brand | unset |
| `WEB_FETCH_ALLOW_PROXY_RANGE` | Allow the direct web fetch guard to reach exactly 198.18.0.0/15 (fake-IP proxy range for some sandboxes; weakens the private-address guard) | off |
| `LLM_BASE_URL` | OpenAI-compatible API root | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Chat model id | `deepseek/deepseek-v4.1-flash-1017` |
| `LLM_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY` | LLM bearer token, first non-empty wins | unset |
| `LLM_TIMEOUT`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE` | Request limits (timeout is per attempt) | `60`, `16000`, `0.2` |
| `LLM_MAX_RETRIES`, `LLM_RETRY_BACKOFF` | Retries for timeouts, connection errors, 408, 429, and 5xx, with doubling backoff | `2`, `2` |
| `LLM_DISABLE_REASONING`, `LLM_NO_THINK_MESSAGE` | How to turn model thinking off | `extra_body` for Nemotron on NVIDIA only, otherwise `off` |
| `EMBED_MODEL` | Local sentence-transformers model | `google/embeddinggemma-2` |
| `EMBED_DEVICE`, `EMBED_DIM`, `EMBED_DTYPE`, `EMBED_BATCH_SIZE` | Embedder settings | `cpu`, `768`, `float32`, `32` |
| `HF_TOKEN` | Hugging Face token, only for a gated model | unset |
| `KOL_CACHE_DIR` | Cache root for web, Facebook, profile enrichment, and embeddings; TikTok search is not cached | `data/cache` at the repo root |
| `JOBS_DB` | SQLite job store | `data/jobs.db` |
| `CORS_ORIGINS` | Allowed browser origins, comma-separated | `http://localhost:5173` |
| `JOB_WORKERS` | Jobs that run at once | `2` |
| `MAX_QUEUED_JOBS` | Queued or running jobs before `POST /api/jobs` returns 429 | `20` |

The last five must be unset, not blank, to take their defaults. An invalid
Apify or discovery value, such as a TikTok cap of 0 or a non-numeric
budget, makes `POST /api/jobs` return 400 naming the variable, and makes the
CLI exit with code 2, before any Apify run starts. `--max-results-per-query`,
`--min-candidates`, and `--max-rounds` override the matching variables for one
CLI run. `--max-enrich` is optional and overrides
`MatchingConfig.enrich_top_n` (default 20).

## Architecture

### Pipeline

The graph lives in `backend/app/graph.py` and the nodes in
`backend/app/nodes.py`.

```mermaid
flowchart TD
    REQ([Brand website + Facebook URL]) --> W
    REQ --> F

    subgraph ingest [Brand ingest, in parallel]
        W["scrape_website<br/>Jina Reader, trafilatura fallback"]
        F["scrape_facebook<br/>Apify page + posts actors"]
    end

    W --> R["research_brand<br/>LLM with a Jina web search tool, at most 3 searches"]
    F --> R
    R --> A["analyze_brand<br/>one LLM call, structured brief"]
    A --> Q["generate_queries<br/>topic keywords plus competitors"]
    Q --> D["discover_kols<br/>Apify TikTok search, per-video stats"]
    D -->|"under target Thai creators and rounds left"| Q
    D -->|"enough Thai creators or round cap"| HF["hard_filter<br/>rules only, counts rejections"]
    HF --> FP["first_pass_rank<br/>score with data already in hand"]
    FP --> E["enrich_kols<br/>first-pass top 20, not the largest accounts"]
    E --> J["judge_and_rank<br/>batched LLM judge on the top 20 or top k,<br/>with up to two refill calls"]
    J --> O(["rank_and_output<br/>MMR lambda 0.7, top k"])
```

1. **Brand ingest.** The website and Facebook nodes run in the same
   superstep. Either source is enough: brand analysis runs when at least one of
   the website text or the Facebook page is available, and fails only when both
   are missing. A missing source is a warning that names the failed scrape.
   An LLM agent then searches the web for what those pages leave out, such as
   competitors, stores, price position, and brand size. `bind_tools` and
   `ToolNode` run the loop, code caps it at 3 searches, and a separate step writes
   notes with source URLs. Each search costs about 10,000 Jina tokens. Notes exist
   only when a search returned results, and a research failure is only a warning.
2. **Structured brief.** One LLM call returns category, key products, price
   tier, target audience, tone, content formats, content style, Thai and English
   topic keywords, competitors, `brand_handles`, and a tier guess: `target_followers`, `tier_label`,
   `tier_confidence`, and `tier_reason`. The tier guess maps an SME or small
   online shop to micro (~30k), a nationwide brand to mid (~150k), and a
   multinational to macro (~800k), and clamps `target_followers` to the
   10k–2M range. `tidy_brand_profile` then cleans the lists, and the output
   shows `tier_label` and `tier_reason`.
3. **Discovery loop.** Query generation and TikTok search repeat until 30
   unique creators that pass the Thai filter are found, or three rounds have
   run. Each round asks for 6-10 new queries built from topic keywords and
   competitors, never the brand name. Each query is one paid Apify run and
   returns up to 15 videos. Creators are deduped, and every video's views,
   likes, comments, shares, date, caption, and URL are kept.
4. **Hard filters.** Free rules drop the brand's own accounts, competitor
   accounts, seller and media accounts, creators who are not Thai, creators
   outside 1k–5M followers, and creators whose latest post is over 60 days
   old. The activity rule waits until after enrichment, because a search hit
   is not a creator's newest video; every rule then runs again on the
   enriched profiles. Each rejection is counted by reason for the CLI and API
   output, and the web app shows the seller count. Plain sellers do not count
   toward the discovery target, so a pool of dealers keeps the search going up
   to its query cap. Finding only sellers ends the Search with an empty list and
   an explanation, not a failure.
5. **First-pass rank.** Every survivor is scored with data already in hand:
   embedding similarity, on-topic share, and search-derived engagement. The
   judge has not run yet, so `judge = sim percentile` and `A_eff = 0.5`. The
   result decides which creators are worth paying to enrich.
6. **Enrichment.** Fetches recent videos for the first-pass leaders, the top
   20 in that ranking rather than the largest accounts. At the estimated rates,
   the `$0.70` reserve and the job budget allow 20 profile runs, matching
   `MatchingConfig.enrich_top_n`. Leaders beyond that keep their search-derived
   stats, get lower engagement confidence, and carry the `search_stats_only`
   flag.
7. **Judge.** The top 20, or the top k when that is larger, go to one batched
   LLM call. It returns `judge`, `A`, a confidence `c`, and an `account_type` per
   creator. Accounts typed `seller` or `media` are removed, counted, and listed
   in `removed`. Up to two more calls judge the next leaders in their place,
   using search stats when they were not enriched. Only creators sent to the
   judge can be recommended. Creators the reply leaves out
   are asked for once more; any still missing, or everyone if the call fails,
   stay in the list with the `unreviewed` warning flag and
   `judge` capped at a neutral 0.5 so they cannot pass the judged leaders on
   similarity alone. The rubric checks category match, tone
   match, and no risky content. The judge reads up to 10 captions with hashtags
   and ad labels per creator. It also reports formats, style, and a style fit that
   must cite a video. These are shown to the Marketer and move no score. Audience is estimated from text, captions, and
   bio only, never from faces. Risky content becomes a flag.
8. **Output.** Every candidate is scored again now that judged values exist.
   The top k is selected with MMR over `Final`, using caption similarity, so
   near-duplicate creators do not fill the list. Each row carries the scores,
   confidences, flags, `tier_reason`, reason, up to three evidence video links,
   and the profile URL.

A fatal node error sets `stage` to `failed` and ends the graph. A partial
failure, such as one profile that fails to enrich, is recorded in `errors` and
the run continues.

### Service

```mermaid
flowchart LR
    UI["React app<br/>Vite, TanStack Query, EventSource"] -->|"POST /api/jobs"| API[FastAPI]
    API -->|"insert job"| DB[("SQLite<br/>jobs + events")]
    API -->|"submit"| POOL["Thread pool<br/>JOB_WORKERS"]
    POOL -->|"graph.stream"| G[LangGraph pipeline]
    POOL -->|"node events, profile, results"| DB
    UI -->|"GET /api/jobs/{id}/events"| API
    API -->|"SSE, reads events after the cursor"| UI
    G --> EXT["Jina, Apify, LLM API,<br/>local embedder"]
```

`POST /api/jobs` validates the request, stores the job, and hands it to a
thread pool. The worker streams the graph and writes a `started` and a
`finished` or `failed` event for each node, and saves the brand profile as soon
as it exists. The SSE endpoint reads stored events after the client's cursor
(`Last-Event-ID` or `?after=`), so a live stream, a reconnect, and a replay of
a finished job all use the same code. Jobs left running by a previous process
are marked failed on startup.

Every external service in `backend/app/integrations/` is a `Protocol` with a
mock and a live implementation, selected per job by the live flags. Web pages,
Facebook pages, profile enrichment, and embedding vectors are cached on disk.
TikTok search is not cached: every query is a paid Apify run, so a repeated run
still pays for search.

## Scoring

Every score is in `[0, 1]` except the MMR term. The functions live in
`backend/app/scoring.py`, the selection in `backend/app/ranking.py`, and every
tuning value in `backend/app/matching_config.py` (`MatchingConfig`).

```text
F       = 0.4·sim + 0.3·on_topic + 0.3·judge (+0.05 mention bonus, cap 1)
E       = 0.8·percentile_in_tier(ER*) + 0.2·percentile_in_tier(reach)
T       = exp(−|log10(followers) − log10(target_followers)| / σ)
σ       = 0.5 + 0.5·(1 − tier_confidence)
A_eff   = c·A + (1 − c)·0.5
Quality = 0.45·E + 0.30·T + 0.25·A_eff
Final   = F × Quality
MMR     = argmax λ·Final − (1 − λ)·max_similarity_to_selected, λ = 0.7
```

**F — fit.** `sim` is the cosine between the brief and the creator's recent
captions, converted to a percentile within the candidate pool. `on_topic` is
the share of up to 10 recent videos whose caption or hashtags contains a brief
topic term. Brief terms are widened offline with the built-in category lexicon
in `backend/app/topic_lexicon.py`: when the brief's category, products, or
topic keywords mention a common category (skincare/beauty, makeup, haircare,
food, beverage, fashion, fitness, tech/gadget, home, baby/mom, pet, travel, or
finance), matching Thai and English terms for that category are added. A
matching creator bio supplies partial credit, so `on_topic = max(video share,
cfg.on_topic_bio_floor)`; the floor is `0.3` by default.
`judge` is the LLM rubric score. Before the judge stage, `judge` is the `sim`
percentile, so the first pass and the final pass use the same formula.

**E — engagement quality.** For each recent video, `ER = (likes + comments +
shares) / views`, skipping videos with zero views. The median over the videos
is shrunk toward the median ER of the same follower tier:
`ER* = (n·median + k·prior_tier) / (n + k)` with `k = 3`, falling back to the
global pool median when a tier is empty. `ER*` is then ranked within the tier,
because engagement naturally falls as followers grow. The second term is reach,
`median views / followers`, also ranked within the tier; a creator with no
views or followers gets the neutral `0.5`. Engagement confidence is
`n/(n+3)` times `0.6` for search-derived stats or `1.0` for profile stats.
Search results skew viral, so a creator seen only in search is trusted less.

**T — tier fit.** A creator exactly at `target_followers` scores 1. The
distance is measured in log10 followers, and `σ` widens when the brief's tier
guess is unsure.

**A_eff — audience fit.** The judge returns `A` and a confidence `c`, and
`A_eff` shrinks the estimate toward `0.5` when `c` is low. This is an
estimate, not measured demographics.

**Quality and Final.** Quality is a weighted blend of the three non-fit
signals, and Final multiplies it by F.

**Diversity.** The top k is chosen greedily with MMR. The first pick has the
highest `Final`; each later pick maximizes `0.7·Final − 0.3·max similarity to
an already selected creator`. The aim is a mix of tiers and styles rather than
near-duplicates.

### Why these choices

**Why F is multiplicative.** A creator with excellent engagement but the wrong
niche should not outrank an on-niche creator with average engagement. Making F
a multiplier means the fit score acts as a gate on Quality; an off-niche
creator can only rank on engagement if F is not near zero. The tests include
that case.

**Why A is an estimate.** TikTok does not expose audience demographics to
scrapers, so there is no measured age or gender split to use. The judge reads
the creator's text, captions, and bio and returns a guess plus a confidence.
`A_eff` shrinks a low-confidence guess toward neutral instead of pretending it
is precise. Age and gender are never inferred from faces, both for ethics and
for PDPA reasons.

**Why a mention bonus instead of a cap.** A creator who names the brand or a
competitor in their recent text gets `+0.05` on F, capped at 1. Creators who
have never mentioned the brand are not penalized; most good creators for a new
campaign have not. Recent competitor sponsorship is only a flag
(`competitor_sponsored`), not a score penalty.

**How the tier is guessed.** The brief prompt asks the model to use price tier,
sales channels, local versus multinational reach, and the brand page's follower
count. An SME or small online shop maps to micro (~30k), a nationwide Thai
brand of any size to mid (~150k), and a multinational to macro (~800k). The result is
clamped to 10k–2M and shown with `tier_label`, `tier_confidence`, and
`tier_reason`.

**Scores before the judge.** Until the batched judge call runs, `judge = sim` and
`A_eff = 0.5`. That keeps first-pass ranking cheap and confirmatory: it only
decides which 20 candidates deserve profile enrichment. The judge then changes
F for the top 20 or top k, and the final ranking is recomputed for everyone.

**Every tuning value lives in `MatchingConfig`.** The score weighting, shrinkage `k`,
reach blend, tier bounds, stage sizes, MMR lambda, and confidence values are
frozen dataclass fields, validated at construction, instead of numbers spread
through the code.

**The eval harness.** [`eval/README.md`](eval/README.md) documents an offline
NDCG@10 harness over cached candidate pools, with `--ablate F|E|T|A`. It needs
no network. The labels in `eval/golden/` are drafts, not human-checked. On the
example pool, ablating A raises mean NDCG@10 from 0.9134 to 0.9576, so the
current A term is not yet justified by that set. The harness is the tool for
tuning once real labels exist.

### Cost per job

`estimate_run_cost` in `backend/app/integrations/apify.py` charges `$0.005`
per run plus a per-item rate, and caps the result at the run's own
`maxTotalChargeUsd`. The rates are TikTok search `$0.0043` (`$0.003` per result
plus `$0.0013` for the Thai proxy, on the Free plan), profile enrichment
`$0.003`, Facebook page `$0.015`, and Facebook posts `$0.005` per item. The
sync endpoint returns dataset items but not usage, so these are estimates.

| Step | Estimate |
| --- | --- |
| Facebook page, 1 item | `0.005 + 0.015` = $0.020 |
| Facebook posts, 20 items | `0.005 + 20 × 0.005` = $0.105 |
| Facebook subtotal | $0.125 |
| TikTok search, 15 videos per query | `0.005 + 15 × 0.0043` = $0.0695 per run |
| Search runs | 10 runs = $0.695 |
| Profile enrichment, 10 videos per profile | `0.005 + 10 × 0.003` = $0.035 per run |
| Profile runs | 20 runs = $0.70 |
| **Total at the conservative estimate** | **$1.52 of $1.55** |

`MatchingConfig.max_search_queries` caps search at 10 queries per job, so an
eleventh search run is not allowed. Raising `--max-enrich` above 20 would
need `$1.52 + $0.035 = $1.555` for a 21st profile, over the `$1.55` budget, so
the budget guard would skip it. `estimate_default_job_usd` reproduces this table.

These are estimates from the code's own per-item rates. The
[live runs](#live-results) used the previous actors and rates, and the code does
not read Apify's billing, so their real spend is unmeasured.

The reserve matters for enrichment. The `$0.70` reserve guarantees room for 20
profile runs, which equals `MatchingConfig.enrich_top_n` (default 20), so every
first-pass leader the config asks for gets profile stats. Leaders outside the
top 20 keep stats from their search videos, score E with lower confidence, and
carry `search_stats_only`. They are not dropped.

LLM calls per job are up to four for brand research, one brand brief, one to
three for query rounds, and one to three for the judge, which includes up to two
refill calls. That is at most 11, plus any retries and the follow-up for creators
a judge reply leaves out. Each web search also uses about 10,000 Jina tokens. The
CLI and API report a cost log with one line per stage. This one is from the
newest live job:

```text
cost log:
  llm: research_brand 3 calls
  web search: 3 queries
  llm: analyze_brand 1 call
  llm: generate_queries round 1 1 call
  search round 1: est $0.70 (10 queries)
  enrich: est $0.70 (20 profiles)
  llm: judge 2 calls (23 creators)
```

## Live results

These are the six most recent successful live jobs, with every service live and
the newest first. Each one ran one discovery round and logged 20 profile runs.
Apify cost is the logged estimate for search and enrichment, so it leaves out
Facebook (about $0.125).

| Brand | Top k | Run time | Accounts filtered out | Estimated Apify cost | Actor |
| --- | --- | --- | --- | --- | --- |
| InnovestX | 10 | 6 min 40 s | 57 | $1.40 | TikTok Data Extractor |
| Akha Ama Coffee | 7 | 9 min 5 s | 106 | $1.35 | Previous |
| Siam Kubota | 10 | 11 min 0 s | 65 | $1.22 | Previous |
| InnovestX | 10 | 10 min 3 s | 82 | $1.35 | Previous |
| InnovestX | 5 | 11 min 3 s | 58 | $1.35 | Previous |
| Dime | 10 | 11 min 4 s | 82 | $1.28 | Previous |

The newest job started five minutes after the commit that added brand research
and content style analysis, and its cost log uses the current Thai-proxy rate.
The other five ran on earlier commits and previous actors. The results are
uneven, and the [known limitations](#known-limitations) describe what went wrong:

- **InnovestX and Dime** returned mostly finance and investing creators. In the
  newest InnovestX job, the first-ranked creator's caption said "Sponsored by
  Dime" and Dime was a listed competitor, but no `competitor_sponsored` flag was
  raised.
- **Akha Ama** returned coffee creators, but also a coffee shop promoting
  itself, a general lifestyle account, and creators far below the mid-tier
  target.
- **Siam Kubota** returned mostly the brand's own sales channel: the judge
  described all of the top five as Kubota dealers or salespeople.

## Business value

The tool is for an agency's influencer team. It turns a client's two URLs into a
first shortlist they can review, so their time goes to judging and contacting
creators rather than searching for them.

- **Speed.** A live job took about 9-11 minutes on the previous actors, without
  anyone watching it. The one recorded job on the current actor took 6 min 40 s.
- **Known cost per brief.** Apify is budgeted at up to $1.55 per job, with a hard
  check before each run. LLM cost is separate and not tracked.
- **Reasoning the team can check.** Each creator comes with component scores, a
  reason, flags, and evidence videos, so a shortlist can be challenged and
  explained to a client. It can be exported to CSV.
- **Consistent criteria.** Every client goes through the same filters and scoring,
  which also gives the team one place to tune them.
- **Fewer poor fits reaching a person.** Own-brand, competitor, inactive, non-Thai,
  and out-of-range accounts are removed before review, 57-106 per live job.

The time saved has not been measured: there is no manual baseline, and the tool
has not been used on a real campaign. A pilot should compare these on the same
briefs:

1. Hours from brief to approved shortlist, against the current manual process.
2. The share of suggested creators the team accepts.
3. Creator response and booking rates.
4. Campaign results per creator, which can also become ranking labels for
   [the eval harness](#scoring).

## Design decisions and trade-offs

**An OpenAI-compatible LLM, chosen by configuration.** The live client speaks
the chat completions API, so moving between the default
[DeepSeek on OpenRouter](#llm-model), NVIDIA-hosted Nemotron, OpenAI, or a local
server means changing `LLM_BASE_URL`, `LLM_MODEL`, and the key. Every call asks
for one JSON object, validates it with Pydantic, and retries once with the
validation error in the prompt. Nemotron is a reasoning model, so on NVIDIA's
endpoint the client turns thinking off through `chat_template_kwargs`, which cut
a brand-analysis call from 30-45 seconds to about 3. Other models, including the
default, get a plain request unless `LLM_DISABLE_REASONING` is set.

**Local embeddings as a free similarity term.** embeddinggemma runs on CPU
through sentence-transformers, with no per-call cost and no data leaving the
machine. It supplies `sim` for F and the document similarity that MMR uses.
The trade-off is a heavy optional dependency (torch), so it sits behind the
`embed` extra and the default install and tests stay light.

**One judge, not one call per creator.** Only the top 20 or top k see the LLM judge, and
they share one request, with up to two refill requests when the judge removes
accounts. That keeps the expensive stage small and makes the rubric consistent
across candidates.

**An estimated Apify budget plus per-run caps.** Every Apify run sets
`maxItems`, `limit`, and `maxTotalChargeUsd`. On top of that,
`APIFY_JOB_BUDGET_USD` (default $1.55) covers the whole job. A run starts only
if its worst-case estimate, from the item count it asks for, still fits in the
budget. Runs that do not fit are skipped and recorded as warnings, and the job
still finishes with what it has. The cost table is in
[Cost per job](#cost-per-job).

**A reserve so search cannot starve enrichment.** A search run starts only if
the estimated spend, plus that run, plus `APIFY_ENRICH_RESERVE_USD` ($0.70)
fits in the budget, so profile runs always have room. With the defaults this
means at most 10 search runs (the query cap) and 20 profile runs, matching
`MatchingConfig.enrich_top_n`.

**SSE instead of polling from the browser.** Progress is a one-way stream of
small events, which is what server-sent events are for. `EventSource`
reconnects and resends `Last-Event-ID` on its own. Inside the server the stream
reads SQLite every 0.25 seconds, which keeps replay and live delivery on one
code path at the cost of a little latency. Before reconnecting, the frontend
checks that the job still exists, so a stale `?job=` link stops instead of
retrying forever.

**SQLite for jobs and events.** One file, no extra service, and it survives a
restart in the Compose volume. It fits a single-node tool with a few jobs at a
time. Several API replicas would need a shared database and a real queue.

**Demo mode by default.** Without an `APIFY_TOKEN`, the service runs the full
graph on fixtures, so anyone can clone the repo and see the product in a minute
without spending money. The fixtures cover the awkward cases: duplicate handles
with different casing, non-Thai creators, the brand's own account, seller and
media accounts, and a second discovery round.

**Guarding a public-URL fetcher.** The API only accepts Facebook URLs on
facebook.com or fb.com, rejects website URLs on local or private addresses,
and the web reader checks resolved addresses before a direct fetch.

## Ethics and privacy

This section records how the prototype behaves and what must change before real
use. It is not legal advice, and production use needs legal review.

**What the tool does to reduce harm**

- **No inference from faces.** Audience is estimated from text only. The judge
  prompt forbids inferring age or gender from faces, appearance, or photos, and
  its creator cards contain no images.
- **Audience is labeled as an estimate.** Low-confidence estimates shrink toward
  neutral, and the web app marks low-confidence scores as limited data.
- **A person makes the decision.** The output is a shortlist with evidence for a
  marketer to review. A `risky` flag is an LLM's reading of text, not a finding
  about the creator.
- **Secrets stay out of output.** Keys are not logged or put in error messages,
  and `/api/health` reports them only as set or missing.
- **Local by default.** Compose binds to localhost, and demo mode uses only
  fixtures.

**Data handling in live mode**

- **Personal data.** Public handles, nicknames, bios, captions, and avatar links
  can identify a person. This tool treats them as personal data under Thailand's
  [PDPA](#references), even though they are public.
- **Third parties.** Apify receives the Facebook URL, search queries, and creator
  handles. Jina receives the website URL and the web research queries. Creator handles, nicknames, bios, and
  captions go to the configured LLM host. Embeddings stay on the machine.
- **Retention.** Jobs and creator data stay in SQLite, and the API has no delete
  endpoint. Web, Facebook, and profile caches stop being used after 7 days, but
  the files stay on disk, and embedding vectors do not expire.
- **Creator rights.** There is no process to notify creators, handle their
  requests, or remove them from results.

**Platform terms**

Live mode collects data through third-party scrapers, the route the brief
suggests for a prototype. The platforms restrict this:
[TikTok's terms](#references) prohibit automated scripts that collect information
from the service, and [Meta's terms](#references) require express written
permission for automated data collection.

**Before production**

1. Get legal review of the PDPA lawful basis, privacy notice, and platform terms,
   and move to permitted data access wherever it is available.
2. Add a retention period, deletion, and a way to handle creator requests.
3. Add authentication and role-based access to jobs and creator data.
4. Sign data processing terms with vendors, or use a self-hosted LLM.
5. Ask shortlisted creators for their own TikTok analytics, with their agreement,
   for real audience demographics.

## Known limitations

- **The tuning values are not fitted.** The score weighting, the shrinkage constant,
  and the MMR lambda are reasoned starting points. The eval labels are drafts,
  and on the example pool ablating A does not lower NDCG@10, so that term is
  not yet supported by evidence.
- **Seller and media detection can misfire.** The rules match tokens such as
  `shop`, `official`, `news`, and `tv`, sales phrases in the bio, and business
  names in the handle or nickname, so a fan account named after a brand or a
  creator with `tv` in the name is dropped. The judge decides the unclear
  accounts and can be wrong in either direction.
- **The tier guess can still differ by one step.** The LLM reads the brand again
  on every Search, so a brand near a boundary can move between neighbouring
  tiers, which changes T and the shortlist. This variation is accepted; the
  prompt only removes the overlap between large Thai brands and multinationals.
- **Competitor sponsorship can be missed.** The flag needs a competitor the LLM
  listed and a caption with `#ad`, `#sponsored`, or a listed Thai marker. It does
  not recognize "Sponsored by Dime" and does not read TikTok's own ad label, so
  two creators in the newest InnovestX job kept only `mentions_competitor`. That
  mention also adds the `0.05` bonus to F.
- **LLM replies can come back empty.** Two Akha Ama jobs failed at brand analysis
  this way before `LLM_MAX_TOKENS` was raised to 16000. Two later live jobs
  succeeded, but jobs do not store the model, so this does not show that the
  setting alone fixed it.
- **Content style reads text only.** Formats, style, and style fit come from
  captions and hashtags, not from what is said or shown in the videos. Style fit
  has no human-checked eval yet, so it is not part of the score.
- **`on_topic` is keyword-based.** Terms are expanded from a small built-in
  category lexicon with a bio fallback, but a relevant video described in
  unseen words can still be missed, and a passing mention still counts as a
  hit. Adding a new niche means adding its terms to `topic_lexicon.py`.
- **The Apify budget is an estimate.** It is computed from item counts and
  per-item rates measured on this account, not from Apify's billing. If Apify's
  prices rise, real spend can pass the budget until the rates are updated.
- **The live budget caps enrichment.** The 20-profile default fits only at the
  estimated rates. If real prices are higher, the guard skips the last
  profiles, and they keep `search_stats_only` stats.
- **TikTok search is not cached.** Web pages, Facebook pages, profile
  enrichment, and embeddings are cached, but every search query is a paid run.
  Repeating a job without changing the brand still pays for search.
- **No warning when the budget skips the Facebook run.** If the budget stops a
  Facebook run, the reader quietly falls back to the web reader, or keeps the
  page without posts, and the job shows no warning about it.
- **The Facebook web fallback can accept a login wall.** When it reads the page
  through Jina, it does not detect Facebook's login page, so the brand profile
  may get login text instead of the page's "about".
- **Quality is measured on a few brands, without labels.** The
  [live runs](#live-results) cover four brands and no marketer has rated them, so
  recall and real shortlist quality are unmeasured.
- **The brand mention check is literal.** A creator who refers to the brand
  only in speech, an image, or an unlisted nickname does not get the bonus, and
  an alias the LLM misses is not matched.
- **The Thai filter is a heuristic.** A Thai creator with English-only text and
  no TH region is missed, and a non-Thai account that writes in Thai is kept.
- **Audience fit is an estimate.** It comes from the judge reading text, not
  from audience demographics, which TikTok does not expose to scrapers. It is
  never inferred from faces.
- **The API has no authentication.** It is built for local use and binds to
  localhost in Compose.

## Testing

```bash
cd backend && uv run pytest          # 510 tests
cd backend && uv run python ../eval/run_eval.py   # NDCG@10, no network
cd frontend && pnpm lint && pnpm test   # ESLint, Prettier, 84 Vitest tests
```

The tests make no network calls. The backend suite covers the graph routing and
fail-fast behaviour, every live integration against mocked HTTP (Jina and
captcha replies, Apify errors and budgets, LLM JSON repair and retries, and
the embedder), the API and SSE resume, brand-profile cleanup, the new scores,
the hard filters, and MMR. The frontend suite covers the step reducer, the SSE
follower's retry and stop rules, the job-progress hook, the API client,
formatting, the CSV export, and the app flow.

`backend/scripts/` has small live smoke checks for Apify, the LLM, and the
embedder. They need real keys, and the Apify one spends credit.

## Next steps

1. **Human labels and tuning.** Replace the draft eval labels with a marketer's
   labels for a few brands, then use [`eval/README.md`](eval/README.md) to
   tune the scoring values and confirm that each term earns its place.
2. **A pilot.** Run the [business value](#business-value) comparison on real
   briefs.
3. **Real audience data.** Use creator-shared analytics or permitted data access,
   then replace A with measured demographics.
4. **Privacy and access work.** Complete the
   [before-production list](#ethics-and-privacy).
5. **More enrichment within budget.** Raise the profile cap only when the
   estimated spend allows it, and measure whether profile stats change the top
   10 enough to justify the cost.
6. **Better signals.** Detect sponsorship from TikTok's ad label and "Sponsored
   by" captions, stop giving competitor mentions the fit bonus, add comment
   language as a weak audience signal, detect the Facebook login wall, and warn
   when the budget skips a Facebook run.
7. **Real spend tracking.** Read `usageTotalUsd` from Apify's run API after each
   run instead of estimating.
8. **CI.** Run the backend and frontend suites and the Docker build on every
   pull request.
9. **Production shape.** A shared database and a job queue if it ever needs more
   than one node.

## Repo layout

```text
backend/
  app/
    graph.py          # graph wiring, parallel fan-in, discovery loop
    nodes.py          # the pipeline nodes
    matching_config.py # every tuning value for matching v2
    filters.py        # free hard filters and rejection reasons
    scoring.py        # F, E, T, A_eff, Quality, Final, flags, MMR helper
    ranking.py        # first pass, judge pass, RankedKol, MMR selection
    models.py         # MatchRequest, BrandProfile, KolProfile, RankedKol
    state.py          # graph state and reducers
    brand_match.py    # brand and competitor name matching
    brand_tidy.py     # cleanup of the LLM brand profile
    thai.py           # decides whether a creator is Thai
    topic_lexicon.py  # category terms for on_topic
    prompts.py        # prompts for research, the brief, queries, and judge
    cli.py            # CLI (table or JSON)
    api/              # FastAPI app, settings, SQLite store, job runner
    integrations/     # Protocol + mock + live client per service, with caches
    fixtures/         # demo website, Facebook page, creators, Apify samples
  scripts/            # live smoke checks
  tests/
eval/                 # offline NDCG@10 harness, golden labels, cached pools
frontend/
  src/
    api/              # wire types, fetch client, query hooks, SSE follower
    components/       # form, progress, brand profile, results, history
    hooks/ lib/       # job progress, URL state, step reducer, CSV, validation
    styles/tokens.css # design tokens
  Dockerfile          # pnpm build, then nginx
  nginx.conf          # static files, /api proxy, unbuffered event stream
docs/
  reference.md        # HTTP API and live integration details
  screenshots/
docker-compose.yml
.env.example
```

## References

Data collection:

- [Apify TikTok Data Extractor](https://apify.com/clockworks/free-tiktok-scraper)
- [Apify Facebook Pages Scraper](https://apify.com/apify/facebook-pages-scraper)
  and [Facebook Posts Scraper](https://apify.com/apify/facebook-posts-scraper)
- [Jina Reader](https://jina.ai/reader/) and
  [Trafilatura](https://trafilatura.readthedocs.io/)

AI and application:

- [DeepSeek V4.1 Flash on OpenRouter](https://openrouter.ai/deepseek/deepseek-v4.1-flash-1017)
- [NVIDIA Nemotron 3.5 Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)
- [Jina Search](https://s.jina.ai/)
- [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) and
  [Sentence Transformers](https://sbert.net/)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) and
  [FastAPI](https://fastapi.tiangolo.com/)

Law and platform terms:

- [Personal Data Protection Act B.E. 2562 (2019), unofficial English translation, Ministry of Digital Economy and Society](https://www.mdes.go.th/law/detail/3577)
- [Thailand's Personal Data Protection Committee](https://www.pdpc.or.th)
- [TikTok Terms of Service](https://www.tiktok.com/legal/page/row/terms-of-service/en)
- [Meta Automated Data Collection Terms](https://www.facebook.com/legal/automated_data_collection_terms)
