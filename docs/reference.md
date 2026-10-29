# Reference

Detailed behaviour of the HTTP API and each live integration. The
[README](../README.md) covers setup, architecture, and design decisions.

## HTTP API

| Method and path | Result |
| --- | --- |
| `POST /api/jobs` | 202 `{"job_id": "..."}`. 400 when a requested live flag cannot run. 422 on an invalid body, a `facebook_url` outside facebook.com or fb.com, or a `website_url` on a local or private address. 429 when `MAX_QUEUED_JOBS` jobs are queued or running |
| `GET /api/jobs/{id}` | Status, stage, request, effective flags, errors, brand profile, rejected counts, cost log, and ranked results. 404 when unknown |
| `GET /api/jobs/{id}/events` | Server-sent events, ending with a `done` or `failed` event. 404 when unknown |
| `GET /api/jobs?limit=20` | Recent jobs, newest first. `limit` is 1 to 100 |
| `GET /api/health` | Mode, default flags, the configured models, and which credentials are set |

The examples below come from a local run in demo mode, trimmed.

### Create a job

`top_k` is 1 to 200 and defaults to 10. The live flags are optional. An omitted
flag takes the mode's default.

```bash
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' -d '{
  "website_url": "https://example-brand.co.th",
  "facebook_url": "https://www.facebook.com/example-brand",
  "top_k": 3
}'
```

```json
{"job_id":"14aa56a39f8845cab15a866b34308480"}
```

### Read a job

A job moves from `queued` to `running` to `succeeded` or `failed`.
`brand_profile` is `null` until `analyze_brand` finishes, and is saved before
the next node starts.

```bash
curl localhost:8000/api/jobs/14aa56a39f8845cab15a866b34308480
```

```json
{
  "id": "14aa56a39f8845cab15a866b34308480",
  "status": "succeeded",
  "stage": "done",
  "request": {
    "website_url": "https://example-brand.co.th/",
    "facebook_url": "https://www.facebook.com/example-brand",
    "top_k": 3
  },
  "flags": {"live_web": false, "live_fb": false, "live_kols": false, "live_llm": false, "live_embed": false},
  "errors": [],
  "brand_profile": {
    "category": "gaming eyewear / blue-light glasses",
    "key_products": ["blue-light glasses", "gaming glasses", "clip-on blue-light lenses"],
    "price_tier": "mid",
    "target_audience": {"gender": "any", "age_range": "16-34", "interests": ["gaming", "esports", "tech"]},
    "tone": "energetic, friendly, gamer-coded",
    "topic_keywords_th": ["แว่นกันแสงสีฟ้า", "แว่นเกมมิ่ง"],
    "topic_keywords_en": ["blue light glasses", "gaming glasses", "eyewear"],
    "competitors": [],
    "brand_handles": ["@ExampleBrand", "examplebrand"],
    "brand_names": ["ExampleBrand"],
    "target_followers": 30000,
    "tier_label": "micro",
    "tier_confidence": 0.7,
    "tier_reason": "Mock analysis derived from the website and Facebook page: a small Bangkok online shop, so a micro creator around 30k."
  },
  "rejected": {
    "own_brand_account": 1,
    "competitor_account": 1,
    "shop_media_account": 2,
    "not_thai": 1,
    "inactive": 1,
    "followers_out_of_range": 2
  },
  "cost_log": [
    "llm: analyze_brand 1 call",
    "llm: generate_queries round 1 1 call",
    "search round 1: est $0.52 (8 queries)",
    "llm: generate_queries round 2 1 call",
    "search round 2: est $0.13 (2 queries)",
    "enrich: est $0.70 (20 profiles)",
    "llm: judge 1 call (15 creators)"
  ],
  "results": [
    {
      "kol": {
        "handle": "fpscoach.oat",
        "url": "https://www.tiktok.com/@fpscoach.oat",
        "nickname": "โค้ชโอ๊ต FPS",
        "followers": 36000,
        "bio": "สอนเล่นเกมยิง ตั้งค่าเมาส์ จอ แว่น",
        "language": "th",
        "region": "TH",
        "posts_per_week": 2.24,
        "content_summary": "Recent posts: สอนคอมโบตัวใหม่ ... รีวิวแว่นเกมมิ่ง ... แว่นกันแสงสีฟ้า",
        "videos": [
          {
            "url": "https://www.tiktok.com/@fpscoach.oat/video/7326318781316695081",
            "views": 68043,
            "likes": 4188,
            "comments": 178,
            "shares": 340,
            "posted_at": "2026-10-06T08:07:09Z",
            "caption": "สรุปให้: แว่นกันแสงสีฟ้า รุ่น Lumen ใส่เล่นเกมทั้งคืน ..."
          }
        ],
        "stats_source": "profile",
        "source": "apify"
      },
      "scores": {
        "F": 0.8017,
        "E": 0.8318,
        "T": 0.8853,
        "A_eff": 0.7457,
        "F_conf": 0.9,
        "E_conf": 0.75,
        "T_conf": 0.7,
        "A_conf": 0.4913,
        "sim": 0.8375,
        "on_topic": 0.5556,
        "judge": 1.0,
        "er_star": 0.0761,
        "reach": 1.2289,
        "judged": true
      },
      "quality": 0.8263,
      "final": 0.6624,
      "flags": [],
      "reason": "Matched brief topics: แว่นกันแสงสีฟ้า, แว่นเกมมิ่ง, เกมเมอร์. Audience cues: gaming, eyewear.",
      "evidence_urls": [
        "https://www.tiktok.com/@fpscoach.oat/video/7326318781316695081",
        "https://www.tiktok.com/@fpscoach.oat/video/7386802432291829589"
      ],
      "profile_url": "https://www.tiktok.com/@fpscoach.oat"
    }
  ],
  "created_at": "2026-10-27T06:17:33.008856+00:00",
  "updated_at": "2026-10-27T06:17:33.057882+00:00"
}
```

`rejected` counts hard-filter drops by reason. `cost_log` has one line per paid
stage. `flags` can contain `mentions_brand`, `mentions_competitor`,
`competitor_sponsored`, `low_reach`, `activity_unknown`, `search_stats_only`,
and `risky_content`.

### Stream progress

```bash
curl -N localhost:8000/api/jobs/14aa56a39f8845cab15a866b34308480/events
```

Each event carries its sequence number as the SSE `id`. Node events have type
`node`, and the last event is `done` or `failed`, after which the server closes
the stream. A `finished` event carries a short summary such as `discovered+22`.

```text
event: node
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":1,"type":"node","node":"scrape_facebook","status":"started","summary":"","timestamp":"2026-10-27T06:17:33.020539+00:00"}
id: 1

event: node
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":3,"type":"node","node":"scrape_facebook","status":"finished","summary":"facebook=ExampleBrand","timestamp":"2026-10-27T06:17:33.024247+00:00"}
id: 3

event: node
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":10,"type":"node","node":"discover_kols","status":"finished","summary":"round=1 discovered+22","timestamp":"2026-10-27T06:17:33.034594+00:00"}
id: 10

event: done
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":25,"type":"done","node":null,"status":null,"summary":"ranked 3 KOLs","timestamp":"2026-10-27T06:17:33.057947+00:00"}
id: 25
```

To resume after a dropped connection, send the last `id` you received as
`Last-Event-ID` (an `EventSource` does this on its own) or pass `?after=N`. The
stream then starts after that event:

```bash
curl -N -H 'Last-Event-ID: 22' localhost:8000/api/jobs/14aa56a39f8845cab15a866b34308480/events
```

```text
event: node
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":23,"type":"node","node":"rank_and_output","status":"started","summary":"","timestamp":"2026-10-27T06:17:33.055289+00:00"}
id: 23

event: node
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":24,"type":"node","node":"rank_and_output","status":"finished","summary":"results=3","timestamp":"2026-10-27T06:17:33.056399+00:00"}
id: 24

event: done
data: {"job_id":"14aa56a39f8845cab15a866b34308480","seq":25,"type":"done","node":null,"status":null,"summary":"ranked 3 KOLs","timestamp":"2026-10-27T06:17:33.057947+00:00"}
id: 25
```

### List jobs

```bash
curl 'localhost:8000/api/jobs?limit=5'
```

The response is a JSON array of jobs in the same shape as `GET /api/jobs/{id}`,
newest first.

### Check health

```bash
curl localhost:8000/api/health
```

```json
{
  "status": "ok",
  "mode": "demo",
  "defaults": {"live_web": false, "live_fb": false, "live_kols": false, "live_llm": false, "live_embed": false},
  "credentials": {
    "APIFY_TOKEN": false,
    "LLM_API_KEY": false,
    "NVIDIA_API_KEY": false,
    "OPENAI_API_KEY": false,
    "JINA_API_KEY": false
  },
  "models": {
    "llm_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "llm_host": "integrate.api.nvidia.com",
    "embed_model": "google/embeddinggemma-2"
  }
}
```

`defaults` are the flags an omitted request field takes. `models` shows what
the live LLM and embedder would use. `llm_host` is the host of `LLM_BASE_URL`
only, never the full URL.

## Live integrations

Each section below covers one `--live-*` flag. The API flag of the same name
runs the same code.

### Web reading

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

### Facebook page (Apify)

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
  whitespace collapsed and cut to 1,000 characters. Brand analysis sees only
  the newest 3 posts, each cut to 200 characters, and treats them as weak
  signals for tone and current campaigns rather than as product evidence.
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

### TikTok KOL search (Apify)

The default run uses the offline KOL fixtures and does not call Apify.
`--live-kols` searches TikTok and enriches the first-pass leaders through Apify
(one paid run each, up to `--max-enrich`, which overrides
`MatchingConfig.enrich_top_n`, default 20). Combine it with
`--live-web` when both the site and the creator data should be live.

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
  to fetch recent videos for one profile. Each video keeps its views, likes,
  comments, shares, date, caption, and URL, and the profile is marked
  `stats_source: "profile"`. The scorer computes
  `(likes + comments + shares) / views` per video and takes the median; the
  profile itself does not store a pre-computed rate. `posts_per_week` is
  `(n - 1) / weeks` between the oldest and newest non-pinned video in the
  sample. `content_summary` is a short line of captions and hashtags.
- **Cost controls.** `--max-results-per-query` overrides
  `TIKTOK_RESULTS_PER_QUERY` (default 15) for search only; profile enrichment
  always uses `TIKTOK_PROFILE_VIDEOS` (default 10). `--max-enrich` overrides
  `MatchingConfig.enrich_top_n` (default 20), which caps how many first-pass
  leaders are enriched. Video, cover, subtitle, and
  avatar downloads stay off, and comment scraping stays at zero. Each run sets
  `maxItems`, `limit`, and `maxTotalChargeUsd` (default $0.50) on the sync
  endpoint. Transport errors and HTTP 429/5xx are retried once. Timeouts and
  HTTP 408 are not retried, because the paid run may already be in progress.
  The store lists these actors from about $1–$2 per 1,000 results.
- **Cost control.** Each run has its own cap (`APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD`,
  `APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD`, and `APIFY_FACEBOOK_MAX_CHARGE_USD`),
  and one `APIFY_JOB_BUDGET_USD` budget covers the whole job across Facebook,
  search, and enrichment. The sync endpoint returns dataset items but no usage,
  so spend is estimated from the items returned plus a small per-run amount.
  The budget is soft: the run that crosses it finishes, then later calls are
  skipped and recorded as warnings instead of failing the job. Cache hits do
  not call Apify. TikTok search is not cached, so every query is a paid run.
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
  --max-enrich 20
```

`backend/scripts/apify_smoke.py` runs one search query with a maximum of 5
results and prints `handle`, `url`, and `followers`. It is the live check to
run once a token is available:

```bash
uv run python scripts/apify_smoke.py "รีวิวแว่นตา"
```

### LLM

The default run uses the offline brand analyzer and does not call a model.
`--live-llm` sends brand analysis, TikTok query generation, and creator judging
to an OpenAI-compatible chat completions API. Facebook scraping
stays mocked unless `--live-fb` is set. Embeddings stay on the offline mock unless `--live-embed` is
set. Combine the flag with `--live-web` and `--live-kols`
when those sources should be live too. Switching providers is configuration
only: set `LLM_BASE_URL`, `LLM_MODEL`, and a key.

| Variable | Default | Role |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API root, including the `/v1` prefix |
| `LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Chat model id |
| `LLM_API_KEY` | unset | Bearer token. If empty, `NVIDIA_API_KEY` is used, then `OPENAI_API_KEY` |
| `LLM_TIMEOUT` | `60` | Request timeout in seconds |
| `LLM_MAX_TOKENS` | `4096` | Completion token budget, shared with any reasoning trace |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_DISABLE_REASONING` | `extra_body` for the default endpoint and model, otherwise `off` | Provider-specific reasoning control, below |
| `LLM_NO_THINK_MESSAGE` | `/no_think` | System message used when reasoning control is `system` or `both` |

The default model is a reasoning model. A short prompt can take tens of seconds
and spend most of its completion tokens on `reasoning_content`. With a small
`max_tokens`, `content` comes back null. The client defaults are sized for
that: 4096 completion tokens and a 60 second timeout. Empty or missing
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

The model returns three things for each creator: `judge` (0–1), `audience`
(0–1), and `audience_confidence` (0–1), plus `risky` and a short reason. The
judge rubric covers category match, tone match, and no risky content. Audience
is inferred from text, captions, and bio only, never from faces. Only the top
15 candidates by first-pass score are sent, in one request. The returned
values replace the `judge = sim` placeholder and the neutral `A_eff = 0.5`,
and the final ranking is recomputed. Query generation returns only the new
queries for that round, so the discovery loop and `last_queries` keep working.

The API key is not written to logs. It is stripped from error text.

```bash
export LLM_API_KEY=...
uv run python -m app.cli match \
  --website https://example-brand.co.th \
  --facebook https://www.facebook.com/example-brand \
  --top-k 5 \
  --live-llm
```

`backend/scripts/llm_smoke.py` runs one small brand-analysis call. It exits
with a message and code 2 when no key is set:

```bash
uv run python scripts/llm_smoke.py
```

### Embeddings

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
| `KOL_CACHE_DIR` | `data/cache` | cache root; embeddings are `$KOL_CACHE_DIR/embeddings` |
| `HF_TOKEN` | unset | optional download token |

The brand text (category, key products, target audience, topic keywords, and
competitors) is the query. Each creator (nickname, bio, caption summary, and
hashtags found in them) is a document.
Queries use the `SearchQuery` prompt (`task: search result | query: `).
Documents use the `Document` prompt (`title: none | text: `). If those prompts
are not on the model, a document is formatted as
`title: {nickname or handle} | text: {content}`.

Similarity is cosine on L2-normalized vectors. `EMBED_DIM` truncates the
vector. The context window is 8192 tokens. Dtype is float32 or bfloat16, not
float16.

The brand-to-creator cosine is converted to a percentile within the candidate
pool and becomes the `sim` term of F. Before the judge stage that percentile
also stands in for `judge`. MMR uses document-to-document cosine to keep
near-duplicates out of the final list. Engagement, tier fit, and audience fit
do not use embeddings.

Vectors are cached as JSON under `data/cache/embeddings`. The file name is the
SHA-256 of the model, the prompt prefix, the text, and the dimension. Writes
are atomic. A cache hit does not call the model.

`backend/scripts/embed_smoke.py` embeds one query and two short documents:

```bash
uv run python scripts/embed_smoke.py
```
