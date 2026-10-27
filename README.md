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
whole pipeline runs offline and produces stable output. Web page reading can run
live with `--live-web` (see below). Swap in real implementations later for:

- Facebook page scraping
- TikTok creator search via Apify (profiles and posts)
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
  or returns fewer than 200 characters.
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
    integrations/    # Protocol + Mock per service; web_reader_live.py is the opt-in reader
    fixtures/        # mock website, Facebook page, and KOL records
  tests/
  pyproject.toml
README.md
```
