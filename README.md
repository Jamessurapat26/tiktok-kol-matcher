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
whole pipeline runs offline and produces stable output. Five pieces can run live:

- Web page reading, with `--live-web` (see below).
- Facebook page reading, with `--live-fb` (see below). It calls Apify and needs
  `APIFY_TOKEN`. The default stays on the offline fixture.
- TikTok creator search and enrichment, with `--live-kols` (see below). It calls
  Apify and needs `APIFY_TOKEN`. The default stays on the offline fixtures.
- Brand analysis, query generation, and relevance reranking, with `--live-llm`
  (see below). The default stays on the offline mock.
- Embeddings and the cosine prefilter, with `--live-embed` (see below). The
  default stays on the offline mock and does not download a model.

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
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 10
```

Progress is printed to stderr; the ranked table goes to stdout. `--json` returns
the same run as a single JSON object on stdout:

```bash
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
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
  some storefronts answer 200 to curl and 429 to a bare
  HTTP/1.1 client.
- Successful pages are cached as JSON under `data/cache/web` for 7 days. The cache
  key is the SHA-256 of the normalized URL. Set `KOL_CACHE_DIR` to move the cache
  root (`$KOL_CACHE_DIR/web`). A cache hit skips the network.
- Page text is cut to 20,000 characters before it enters graph state.

```bash
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 5 \
  --live-web
```

## Facebook page (Apify)

The default run uses the offline Facebook fixture and does not call Apify.
`--live-fb` reads the page through Apify. It needs `APIFY_TOKEN`, handled the
same way as `--live-kols`: if the token is missing, the CLI exits with code 2
before the pipeline starts.

- **Profile** uses [apify/facebook-pages-scraper](https://apify.com/apify/facebook-pages-scraper)
  with `startUrls`. The page name, title, `intro`, `categories`, `followers`
  (or `likes` when followers is missing), website (`websites`, then the bare
  `website` domain), and `ad_status` are mapped onto the Facebook page.
- **Recent posts** use [apify/facebook-posts-scraper](https://apify.com/apify/facebook-posts-scraper)
  with `resultsLimit` 20. Up to 20 post texts are kept, each with its
  whitespace collapsed and cut to 1,000 characters. Brand analysis sees each
  post cut to 300 characters.
- **Cost controls.** Each page is two runs, and each run sets `maxItems`,
  `limit`, and `maxTotalChargeUsd` ($0.50). The store lists the page scraper
  from about $5–$12 per 1,000 pages and the posts scraper from about $2–$5 per
  1,000 posts.
- **Fallback.** When the profile run fails or returns no usable item, the page
  URL is read with the live web reader (Jina, then trafilatura, with its own
  cache), even without `--live-web`. The page name comes from the URL, the title
  from the first heading, and the intro from the page text. If that also
  fails, `scrape_facebook` records a warning and brand analysis runs on the
  website alone. When only the posts run fails, the page is kept with no posts.
- **Cache.** Complete Apify results are stored as JSON under
  `data/cache/facebook` for 7 days. The key is the SHA-256 of the normalized
  URL. Set `KOL_CACHE_DIR` to move the root (`$KOL_CACHE_DIR/facebook`). A hit
  does not call Apify. Fallback pages and pages without posts are not cached.
  A failed cache write keeps the fetched page.

```bash
export APIFY_TOKEN=...
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 5 \
  --live-fb
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
  `#examplebrand tiktok` counts) is sent as `hashtags`. Every other query is sent as
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
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
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

## LLM

The default run uses the offline brand analyzer and does not call a model.
`--live-llm` sends brand analysis, TikTok query generation, and relevance
reranking to an OpenAI-compatible chat completions API. Facebook scraping
stays mocked unless `--live-fb` is set. Embeddings stay on the offline mock unless `--live-embed` is
set. Combine the flag with `--live-web` and `--live-kols`
when those sources should be live too. Switching providers is configuration
only: set `LLM_BASE_URL`, `LLM_MODEL`, and a key.

| Variable | Default | Role |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API root, including the `/v1` prefix |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Chat model id |
| `LLM_API_KEY` | unset | Bearer token. If empty, `NVIDIA_API_KEY` is used, then `OPENAI_API_KEY` |
| `LLM_TIMEOUT` | `120` | Request timeout in seconds |
| `LLM_MAX_TOKENS` | `4096` | Completion token budget, shared with any reasoning trace |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_RERANK_BATCH_SIZE` | `5` | Creators scored per rerank request |
| `LLM_DISABLE_REASONING` | `extra_body` for the default endpoint and model, otherwise `off` | Provider-specific reasoning control, below |
| `LLM_NO_THINK_MESSAGE` | `/no_think` | System message used when reasoning control is `system` or `both` |

The default model is a reasoning model. A short prompt can take tens of seconds
and spend most of its completion tokens on `reasoning_content`. With a small
`max_tokens`, `content` comes back null. The client defaults are sized for
that: 4096 completion tokens and a 120 second timeout. Empty or missing
`content` is retried once.

`LLM_DISABLE_REASONING` picks how thinking is turned off. When it is unset, the
default NVIDIA endpoint and model use `extra_body` and every other provider
gets a plain request. The values are provider-specific:

- `off` sends a normal chat request.
- `system` prepends `LLM_NO_THINK_MESSAGE` (default `/no_think`). The default
  model ignores it, so it still thinks.
- `extra_body` sends `chat_template_kwargs.enable_thinking` set to false. On the
  default model this turns thinking off: a brand-analysis call dropped from
  about 30-45 seconds and 2,000-3,700 completion tokens to about 3 seconds and
  250 tokens.
- `both`, and also `true`, `1`, or `on`, does both.

A provider that rejects unknown JSON fields works with the default (`off` for
anything but the NVIDIA default). Point `LLM_BASE_URL` at
`https://api.openai.com/v1` and set `LLM_MODEL` plus `OPENAI_API_KEY` to use
OpenAI with no code change.

Each call asks for one JSON object. Code fences and leading prose are stripped,
the first JSON object is validated with Pydantic, and a validation failure is
retried once with the error in the next prompt. A second failure raises. The
graph records that on `errors`, sets `stage` to `failed`, and stops, the same
way other node failures do.

The model returns relevance from 0 to 1 and a short Thai or English reason.
Engagement, audience fit, and tier fit stay on the existing heuristics. The
total still uses the weights 0.4, 0.25, 0.2, and 0.15, with relevance as the
0.4 term. Query generation returns only the new queries for that round, so the
discovery loop and `last_queries` keep working. Rerank requests are batched.

The API key is not written to logs. It is stripped from error text.

```bash
export LLM_API_KEY=...
uv run python -m app.cli match \
  --website https://example.com \
  --facebook https://www.facebook.com/example \
  --top-k 5 \
  --live-llm
```

`backend/scripts/llm_smoke.py` runs one small brand-analysis call. It exits
with a message and code 2 when no key is set:

```bash
uv run python scripts/llm_smoke.py
```

## Embeddings

The default run uses an offline embedder. It does not download a model and it
does not touch the network. `--live-embed` loads `google/embeddinggemma-2`
with sentence-transformers, on CPU unless `EMBED_DEVICE` is set. The model is
loaded once per process and reused. Vision and audio towers are left unloaded
so only the text tower stays in memory.

The model is Apache 2.0. `HF_TOKEN` is optional.

Install the extra before a live run. It pulls sentence-transformers,
transformers, and the CPU build of torch. A plain `uv sync` does not install
them, so the default install and the tests stay offline.

```bash
cd backend
uv sync --extra embed
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 5 \
  --live-embed
```

| Variable | Default | Role |
| --- | --- | --- |
| `EMBED_MODEL` | `google/embeddinggemma-2` | sentence-transformers model id |
| `EMBED_DEVICE` | `cpu` | device passed to the model |
| `EMBED_DIM` | `768` | Matryoshka size passed as `truncate_dim` (`768`, `512`, `256`, or `128`) |
| `EMBED_DTYPE` | `float32` | `float32` or `bfloat16` |
| `EMBED_BATCH_SIZE` | `32` | texts encoded per call |
| `PREFILTER_LIMIT` | `25` | candidates kept after cosine ranking, before rerank |
| `KOL_CACHE_DIR` | `data/cache` | cache root; embeddings are `$KOL_CACHE_DIR/embeddings` |
| `HF_TOKEN` | unset | optional download token |

The brand text (category, products, audience, keywords, hashtags) is the
query. Each creator (nickname, bio, caption summary, hashtags) is a document.
Queries use the `SearchQuery` prompt (`task: search result | query: `).
Documents use the `Document` prompt (`title: none | text: `). If those prompts
are not on the model, a document is formatted as
`title: {nickname or handle} | text: {content}`.

Similarity is cosine on L2-normalized vectors. `EMBED_DIM` truncates the
vector. The context window is 8192 tokens. Dtype is float32 or bfloat16, not
float16.

`--prefilter-limit` (or `PREFILTER_LIMIT`) sets how many creators go to the
reranker. The default is 25. If `--top-k` is larger, the width is raised to
`--top-k`, so a larger shortlist is not silently cut.

The offline reranker uses that cosine as the relevance score. The live
reranker still scores relevance itself and sees the cosine as
`embedding_similarity`. Engagement, audience fit, and tier fit are unchanged.
The total still uses the weights 0.4, 0.25, 0.2, and 0.15.

Vectors are cached as JSON under `data/cache/embeddings`. The file name is the
SHA-256 of the model, the prompt prefix, the text, and the dimension. Writes
are atomic. A cache hit does not call the model.

`backend/scripts/embed_smoke.py` embeds one query and two short documents:

```bash
uv run python scripts/embed_smoke.py
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
    prompts.py       # LLM prompt text for brand analysis, queries, and rerank
    integrations/    # Protocol + Mock per service; live web, Facebook, Apify, LLM, and embedder
    fixtures/        # mock website, Facebook page, KOL records, Apify samples
  scripts/           # apify_smoke.py, llm_smoke.py, and embed_smoke.py
  tests/
  pyproject.toml
README.md
```
