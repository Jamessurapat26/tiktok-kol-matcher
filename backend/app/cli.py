import argparse
import importlib.util
import json
import os
import sys
from typing import Any

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from .graph import (
    MAX_SEARCH_ROUNDS,
    MIN_CANDIDATES,
    PREFILTER_N,
    build_graph,
    initial_state,
)
from .integrations import build_integrations
from .integrations.apify import ApifyConfigError
from .integrations.embeddings import EmbedConfigError
from .integrations.llm_live import LlmConfigError, llm_api_key_from_env
from .models import MatchRequest, RankedKol

_LIST_FIELDS = ("discovered", "candidates", "shortlisted", "ranked", "results")
_ADDITIVE_FIELDS = {"discovered"}

_TABLE_ROW = (
    "{rank:>4}  {handle:<24} {followers:>11} {er:>7} {relevance:>9} "
    "{engagement:>10} {audience:>8} {tier:<6} {total:>6}  {reason}"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Match a brand to TikTok KOLs with the LangGraph pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser(
        "match",
        help="run the matching pipeline for a brand",
    )
    match.add_argument("--website", required=True, help="brand website URL")
    match.add_argument("--facebook", required=True, help="brand Facebook page URL")
    match.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="number of ranked KOLs to return (default: 10)",
    )
    match.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the final state as JSON instead of a table",
    )
    match.add_argument(
        "--min-candidates",
        type=int,
        default=MIN_CANDIDATES,
        help=f"discovery target before filtering (default: {MIN_CANDIDATES})",
    )
    match.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_SEARCH_ROUNDS,
        help=f"maximum discovery rounds (default: {MAX_SEARCH_ROUNDS})",
    )
    match.add_argument(
        "--live-web",
        action="store_true",
        help=(
            "read the brand website live (Jina, then trafilatura) "
            "instead of the offline mock"
        ),
    )
    match.add_argument(
        "--live-kols",
        action="store_true",
        help=(
            "search and enrich TikTok KOLs via Apify "
            "(requires APIFY_TOKEN; default stays on the offline fixtures)"
        ),
    )
    match.add_argument(
        "--max-results-per-query",
        type=int,
        default=10,
        help=(
            "Apify videos per search query and per profile enrichment "
            "(default: 10)"
        ),
    )
    match.add_argument(
        "--max-enrich",
        type=int,
        default=10,
        help=(
            "with --live-kols, enrich at most this many candidates (largest "
            "accounts first); each one is a paid Apify run (default: 10)"
        ),
    )
    match.add_argument(
        "--live-llm",
        action="store_true",
        help=(
            "run brand analysis, query generation, and reranking through an "
            "OpenAI-compatible chat API (LLM_API_KEY, NVIDIA_API_KEY, or "
            "OPENAI_API_KEY; default stays on the offline mock)"
        ),
    )
    match.add_argument(
        "--live-embed",
        action="store_true",
        help=(
            "embed brand and creator text locally with sentence-transformers "
            "(EMBED_MODEL, default google/embeddinggemma-2; EMBED_DEVICE, "
            "default cpu; EMBED_DIM, default 768). Requires the embed extra. "
            "The default stays on the offline mock"
        ),
    )
    match.add_argument(
        "--prefilter-limit",
        type=int,
        default=None,
        help=(
            "candidates kept after cosine ranking, before rerank "
            f"(default: {PREFILTER_N}, or PREFILTER_LIMIT). "
            "Raised to --top-k when that is larger"
        ),
    )
    return parser


def _summarize_update(
    update: dict[str, Any],
    *,
    node: str | None = None,
    discovered_total: int | None = None,
    discovered_before: int | None = None,
) -> str:
    """Turn one node update into a short, single-line progress summary.

    ``discover_kols`` reports the merged unique total from the values snapshot
    (after the reducer dedupes), plus how many of those are new this round.
    """
    parts: list[str] = []

    if "search_rounds" in update:
        parts.append(f"round={update['search_rounds']}")
    if "search_queries" in update:
        parts.append(f"queries+{len(update['search_queries'] or [])}")

    show_merged_discovered = (
        node == "discover_kols" and discovered_total is not None
    )
    for key in _LIST_FIELDS:
        if key == "discovered" and show_merged_discovered:
            added = discovered_total - (discovered_before or 0)
            parts.append(f"discovered={discovered_total} (+{added} new)")
            continue
        if key not in update:
            continue
        count = len(update[key] or [])
        separator = "+" if key in _ADDITIVE_FIELDS else "="
        parts.append(f"{key}{separator}{count}")

    if "website_content" in update:
        content = update["website_content"] or ""
        parts.append(f"website={len(content)} chars")
    if "facebook_page" in update:
        page = update["facebook_page"]
        name = getattr(page, "page_name", None)
        parts.append(f"facebook={name}" if name else "facebook=ok")
    if "brand_profile" in update:
        parts.append("brand=profile")

    errors = update.get("errors") or []
    if errors:
        parts.append(f"errors={len(errors)}")

    if not parts:
        stage = update.get("stage")
        parts.append(f"stage={stage}" if stage else "ok")

    return " ".join(parts)


def _run_pipeline(
    graph: Any,
    request: MatchRequest,
    *,
    show_progress: bool,
) -> dict[str, Any]:
    final_state: dict[str, Any] = {}
    step = 0
    # Updates are buffered until the matching values event so discover_kols can
    # print the merged discovered length instead of the raw update size.
    pending: list[tuple[str, dict[str, Any]]] = []
    discovered_before = 0

    def flush(snapshot: dict[str, Any]) -> None:
        nonlocal step, discovered_before
        discovered_now = len(snapshot.get("discovered") or [])
        for node, update in pending:
            step += 1
            if show_progress:
                summary = _summarize_update(
                    update,
                    node=node,
                    discovered_total=discovered_now,
                    discovered_before=discovered_before,
                )
                print(f"[{step:>2}] {node:<18} {summary}", file=sys.stderr)
        pending.clear()
        discovered_before = discovered_now

    for mode, payload in graph.stream(
        initial_state(request),
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            pending.extend(payload.items())
            continue
        if mode != "values":
            continue
        final_state = payload
        flush(payload)

    if pending:
        flush(final_state)

    return final_state


def _follower_tier(followers: int) -> str:
    if followers >= 1_000_000:
        return "mega"
    if followers >= 100_000:
        return "macro"
    if followers >= 10_000:
        return "micro"
    return "nano"


def _truncate(text: str, limit: int = 50) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _print_table(results: list[RankedKol]) -> None:
    print(
        _TABLE_ROW.format(
            rank="rank",
            handle="handle",
            followers="followers",
            er="ER %",
            relevance="relevance",
            engagement="engagement",
            audience="audience",
            tier="tier",
            total="total",
            reason="reason",
        )
    )
    print("-" * 132)

    for rank, item in enumerate(results, start=1):
        kol = item.kol
        engagement_rate = (kol.median_engagement_rate or 0.0) * 100
        print(
            _TABLE_ROW.format(
                rank=rank,
                handle=_truncate(kol.handle, 24),
                followers=f"{kol.followers:,}",
                er=f"{engagement_rate:.2f}%",
                relevance=f"{item.scores.relevance:.2f}",
                engagement=f"{item.scores.engagement:.2f}",
                audience=f"{item.scores.audience_fit:.2f}",
                tier=_follower_tier(kol.followers),
                total=f"{item.total_score:.3f}",
                reason=_truncate(item.reason),
            )
        )

    if not results:
        print("No results.")


def _json_payload(final_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": final_state.get("stage", "failed"),
        "search_rounds": final_state.get("search_rounds", 0),
        "search_queries": final_state.get("search_queries") or [],
        "counts": {
            "discovered": len(final_state.get("discovered") or []),
            "candidates": len(final_state.get("candidates") or []),
            "shortlisted": len(final_state.get("shortlisted") or []),
            "ranked": len(final_state.get("ranked") or []),
        },
        "results": [
            result.model_dump(mode="json")
            for result in final_state.get("results") or []
        ],
        "errors": final_state.get("errors") or [],
    }


def _resolve_prefilter_limit(args: argparse.Namespace) -> int:
    """CLI flag wins, then ``PREFILTER_LIMIT``, then the graph default."""
    if args.prefilter_limit is not None:
        if args.prefilter_limit < 1:
            raise ValueError("--prefilter-limit must be at least 1")
        return args.prefilter_limit
    raw = os.environ.get("PREFILTER_LIMIT", "").strip()
    if not raw:
        return PREFILTER_N
    try:
        value = int(raw, 10)
    except ValueError:
        raise ValueError("PREFILTER_LIMIT must be an integer") from None
    if value < 1:
        raise ValueError("PREFILTER_LIMIT must be at least 1")
    return value


def _cmd_match(args: argparse.Namespace) -> int:
    if args.max_rounds < 0:
        print("error: --max-rounds must be zero or greater", file=sys.stderr)
        return 2
    if args.max_results_per_query < 1:
        print(
            "error: --max-results-per-query must be at least 1",
            file=sys.stderr,
        )
        return 2
    if args.max_enrich < 1:
        print("error: --max-enrich must be at least 1", file=sys.stderr)
        return 2
    try:
        prefilter_limit = _resolve_prefilter_limit(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.live_kols and not os.environ.get("APIFY_TOKEN", "").strip():
        print(
            "error: --live-kols requires APIFY_TOKEN in the environment",
            file=sys.stderr,
        )
        return 2
    if args.live_llm and not llm_api_key_from_env():
        print(
            "error: --live-llm requires LLM_API_KEY, NVIDIA_API_KEY, or "
            "OPENAI_API_KEY in the environment",
            file=sys.stderr,
        )
        return 2
    # find_spec only, so the model is neither imported nor loaded here.
    if args.live_embed and importlib.util.find_spec("sentence_transformers") is None:
        print(
            "error: --live-embed requires sentence-transformers. "
            "From the backend directory run `uv sync --extra embed`",
            file=sys.stderr,
        )
        return 2

    try:
        request = MatchRequest(
            website_url=args.website,
            facebook_url=args.facebook,
            top_k=args.top_k,
        )
    except ValidationError as exc:
        print(f"error: invalid match request:\n{exc}", file=sys.stderr)
        return 2

    graph_kwargs: dict[str, Any] = {
        "min_candidates": args.min_candidates,
        "max_search_rounds": args.max_rounds,
        "prefilter_n": prefilter_limit,
    }
    if args.live_kols:
        graph_kwargs["enrich_limit"] = args.max_enrich
    # Default stays on build_graph's mocked bundle so the CLI never opens a
    # socket or loads a model unless a live flag is set.
    if args.live_web or args.live_kols or args.live_llm or args.live_embed:
        try:
            graph_kwargs["integrations"] = build_integrations(
                live_web=args.live_web,
                live_kols=args.live_kols,
                live_llm=args.live_llm,
                live_embed=args.live_embed,
                max_results_per_query=args.max_results_per_query,
            )
        except (ApifyConfigError, LlmConfigError, EmbedConfigError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    graph = build_graph(**graph_kwargs)
    try:
        final_state = _run_pipeline(graph, request, show_progress=not args.as_json)
    except GraphRecursionError as exc:
        detail = str(exc).strip()
        message = "error: pipeline exceeded the recursion limit"
        if detail:
            message = f"{message}: {detail}"
        print(message, file=sys.stderr)
        return 1

    errors = list(final_state.get("errors") or [])
    stage = final_state.get("stage")
    if args.as_json:
        print(
            json.dumps(
                _json_payload(final_state),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_table(final_state.get("results") or [])

    # Warnings stay in the JSON body. Failures also go to stderr, and the
    # human table always lists them.
    if stage != "done" or not args.as_json:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)

    return 0 if stage == "done" else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "match":
        return _cmd_match(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
