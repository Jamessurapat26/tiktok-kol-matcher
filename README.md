# TikTok KOL Matcher

A LangGraph pipeline that turns a brand's website and Facebook page into a ranked
shortlist of Thai TikTok creators.

Given a brand website and Facebook page, it scrapes the sources, builds a
structured brand profile, generates TikTok search queries, discovers creators,
keeps Thai creators, enriches their profiles, and scores the shortlist on:

| Signal | Weight |
| --- | --- |
| Relevance to the brand | 40% |
| Engagement | 25% |
| Audience fit | 20% |
| Follower tier | 15% |

Each result includes the component scores, a total score, and a short reason for
the match.

## Pipeline

```mermaid
flowchart TD
    START([MatchRequest]) --> W[scrape_website]
    START --> F[scrape_facebook]
    W --> A[analyze_brand]
    F --> A
    A --> G[generate_queries]
    G --> D[discover_kols]
    D -->|"under 30 discovered KOLs before the Thai filter, and under 2 rounds"| G
    D -->|"enough discovered KOLs or 2 rounds"| T[filter_thai]
    T --> E[enrich_kols]
    E --> P[embed_and_prefilter]
    P --> R[llm_rerank]
    R --> O[rank_and_output]
    O --> OUT([Top-k ranked shortlist])
```

The website and Facebook scrapers run in parallel. Discovery loops back to query
generation until it has at least 30 KOLs or reaches the 2-round cap. The 30-KOL
target counts discovered (deduped) KOLs before the Thai filter. `filter_thai`
runs after that target is met and can shrink the list.

## Current status

This is a working skeleton with deterministic mocks. Every external service in
`backend/app/integrations/` has a `Protocol` and a `Mock` implementation, so the
whole pipeline runs offline and produces stable output. Two pieces can run live:

- Web page reading, with `--live-web` (see below).
- TikTok creator search and enrichment, with `--live-kols` (see below). It calls
  Apify and needs `APIFY_TOKEN`. The default stays on the offline fixtures.

Swap in real implementations later for:

- Facebook page scraping
- embeddings and vector similarity
- LLM brand analysis, query generation, and reranking

## Tech stack

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- [LangGraph](https://langchain-ai.github.io/langgraph/) for orchestration
- Pydantic for domain models and structured outputs
- FastAPI (planned) for an HTTP API

## How to run

```bash
cd backend
uv sync
```

Run the CLI:

```bash
uv run python -m app.cli match \
  --website https://ophtus.com \
  --facebook https://www.facebook.com/ophtus \
  --top-k 10
```

Progress is printed to stderr; the ranked table goes to stdout. `--json` returns
the same run as a single JSON object on stdout:

```bash
uv run python -m app.cli match \
  --website https://ophtus.com \
  --facebook https://www.facebook.com/ophtus \
  --top-k 10 \
  --json
```

Tune discovery with `--min-candidates N` and `--max-rounds N`.

## Web reading

The default run uses the offline website fixture and does not touch the network.
`--live-web` fetches the brand site for real and leaves Facebook, TikTok search,
embeddings, and the LLM mocked:

- **Jina Reader** (`https://r.jina.ai/`) is the primary fetcher. Set `JINA_API_KEY`
  to send a bearer token and raise the rate limit. The key is not written to logs
  or error messages. Image links and empty bullets are stripped from its markdown.
- **trafilatura** is the fallback. It fetches the page directly when Jina errors
  or returns fewer than 200 characters. The request uses browser-like
  `Accept`, `Accept-Language`, and `Accept-Encoding` headers and HTTP/2, because
  some storefronts (including ophtus.com) answer 200 to curl and 429 to a bare
  HTTP/1.1 client.
- Successful pages are cached as JSON under `data/cache/web` for 7 days. The cache
  key is the SHA-256 of the normalized URL. Set `KOL_CACHE_DIR` to move the cache
  root (`$KOL_CACHE_DIR/web`). A cache hit skips the network.
- Page text is cut to 20,000 characters before it enters graph state.

```bash
uv run python -m app.cli match \
  --website https://ophtus.com \
  --facebook https://www.facebook.com/ophtus \
  --top-k 5 \
  --live-web
```

## TikTok KOL search (Apify)

The default run uses the offline KOL fixtures and does not call Apify.
`--live-kols` searches TikTok and enriches Thai creators through Apify (at most
`--max-enrich`, default 10, largest accounts first, one paid run each).
Combine it with `--live-web` when both the site and the creator data should be live.

`APIFY_TOKEN` is read from the environment and sent only as an `Authorization`
header. It is not written to logs or error messages. If it is missing, the CLI
exits with code 2 before the pipeline starts.

- **Search** uses [clockworks/tiktok-scraper](https://apify.com/clockworks/tiktok-scraper).
  A query that is only a hashtag (a trailing "tiktok" is ignored, so
  `#Ophtus tiktok` counts) is sent as `hashtags`. Every other query is sent as
  `searchQueries` with `searchSection` set to `/video`, so each item is a video
  with `authorMeta`. Authors are deduped. Rows without a handle, and actor
  `errorCode` items, are skipped and recorded by `discover_kols`.
- **Enrichment** uses [clockworks/tiktok-profile-scraper](https://apify.com/clockworks/tiktok-profile-scraper)
  to fetch recent videos for one profile. The engagement rate for a video is
  `(likes + comments + shares) / views` (`diggCount`, `commentCount`,
  `shareCount`, `playCount`). When `playCount` is missing or zero, that video
  uses the same numerator divided by the profile's follower count. The stored
  value is the median of those rates, capped to the range 0–1.
  `posts_per_week` is `(n - 1) / weeks` between the oldest and newest non-pinned
  video in the sample. `content_summary` is a short line of captions and hashtags.
- **Cost controls.** `--max-results-per-query` (default 10) caps videos per
  search query and videos fetched per profile. `--max-enrich` (default 10)
  caps how many profiles are enriched. Video, cover, subtitle, and avatar
  downloads stay off, and comment scraping stays at zero. Each run sets
  `maxItems`, `limit`, and `maxTotalChargeUsd` (default $0.50) on the sync
  endpoint. Transport errors and HTTP 429/5xx are retried once. Timeouts and
  HTTP 408 are not retried, because the paid run may already be in progress.
  The store lists these actors from about $1–$2 per 1,000 results.
- **Cache.** Enrichment results are stored as JSON under `data/cache/apify` for
  7 days. Set `KOL_CACHE_DIR` to move the root (`$KOL_CACHE_DIR/apify`). A hit
  does not call Apify.

```bash
export APIFY_TOKEN=...
uv run python -m app.cli match \
  --website https://ophtus.com \
  --facebook https://www.facebook.com/ophtus \
  --top-k 5 \
  --live-kols \
  --max-results-per-query 5 \
  --max-enrich 10
```

`backend/scripts/apify_smoke.py` runs one search query with a maximum of 5
results and prints `handle`, `url`, and `followers`. It is the live check to
run once a token is available:

```bash
uv run python scripts/apify_smoke.py "รีวิวแว่นตา"
```

Run the tests:

```bash
uv run pytest
```

## Repo layout

```text
backend/
  app/
    graph.py         # graph wiring, parallel fan-in, discovery loop
    nodes.py         # the ten pipeline nodes
    models.py        # MatchRequest, BrandProfile, KolProfile, RankedKol, ...
    state.py         # MatchState and list reducers
    cli.py           # argparse CLI (table or JSON)
    integrations/    # Protocol + Mock per service; live web reader and Apify KOL clients
    fixtures/        # mock website, Facebook page, KOL records, Apify samples
  scripts/           # apify_smoke.py, one live search when APIFY_TOKEN is set
  tests/
  pyproject.toml
README.md
```
