import argparse
import importlib.util
import json
import os
import sys
from dataclasses import replace
from typing import Any

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from .graph import (
    MAX_SEARCH_ROUNDS,
    MIN_CANDIDATES,
    build_graph,
    initial_state,
)
from .integrations import build_integrations
from .integrations.apify import ApifyConfigError, DiscoveryLimits
from .integrations.embeddings import EmbedConfigError
from .integrations.llm_live import LlmConfigError, llm_api_key_from_env
from .matching_config import DEFAULT_CONFIG
from .models import BrandProfile, MatchRequest, RankedKol

_LIST_FIELDS = ("discovered", "candidates", "ranked", "results")
_ADDITIVE_FIELDS = {"discovered"}
_MAX_EVIDENCE_LINES = 3

_TABLE_ROW = (
    "{rank:>4}  {handle:<22} {followers:>11} {final:>6} {fit:>6} {eng:>6} "
    "{tier:>6} {aud:>6} {conf:<17} {flags:<28} {profile}"
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
        default=None,
        help=(
            "discovery target of Thai creators "
            f"(default: DISCOVERY_MIN_THAI_CANDIDATES or {MIN_CANDIDATES})"
        ),
    )
    match.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help=(
            "maximum discovery rounds "
            f"(default: DISCOVERY_MAX_ROUNDS or {MAX_SEARCH_ROUNDS})"
        ),
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
        "--live-fb",
        action="store_true",
        help=(
            "read the Facebook page and recent posts via Apify, falling back "
            "to the live web reader (requires APIFY_TOKEN; default stays on "
            "the offline fixture)"
        ),
    )
    match.add_argument(
        "--max-results-per-query",
        type=int,
        default=None,
        help=(
            "Apify videos per search query; overrides TIKTOK_RESULTS_PER_QUERY "
            "for search only (default: TIKTOK_RESULTS_PER_QUERY or 15). "
            "Profile enrichment always uses TIKTOK_PROFILE_VIDEOS (default: 10)"
        ),
    )
    match.add_argument(
        "--max-enrich",
        type=int,
        default=None,
        help=(
            "override how many first-pass leaders to enrich; each one is a "
            "paid Apify run when live KOLs are enabled "
            f"(default: MatchingConfig.enrich_top_n or {DEFAULT_CONFIG.enrich_top_n})"
        ),
    )
    match.add_argument(
        "--live-llm",
        action="store_true",
        help=(
            "run brand analysis, query generation, and judging through an "
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
    return parser


def summarize_update(
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
                summary = summarize_update(
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


def _truncate(text: str, limit: int = 50) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _brief_line(brief: BrandProfile | None) -> str:
    if brief is None:
        return "brief: unavailable"
    return (
        f"brief: tier={brief.tier_label} target={brief.target_followers} "
        f"conf={brief.tier_confidence:.2f} — {brief.tier_reason}"
    )


def _confidence_text(item: RankedKol) -> str:
    scores = item.scores
    return (
        f"{scores.F_conf:.1f}/{scores.E_conf:.1f}/"
        f"{scores.T_conf:.1f}/{scores.A_conf:.1f}"
    )


def _print_table(
    results: list[RankedKol],
    brief: BrandProfile | None,
) -> None:
    print(_brief_line(brief))
    print(
        _TABLE_ROW.format(
            rank="rank",
            handle="handle",
            followers="followers",
            final="final",
            fit="F",
            eng="E",
            tier="T",
            aud="A_eff",
            conf="conf F/E/T/A",
            flags="flags",
            profile="profile_url",
        )
    )
    print("-" * 170)

    for rank, item in enumerate(results, start=1):
        kol = item.kol
        print(
            _TABLE_ROW.format(
                rank=rank,
                handle=_truncate(kol.handle, 22),
                followers=f"{kol.followers:,}",
                final=f"{item.final:.3f}",
                fit=f"{item.scores.F:.2f}",
                eng=f"{item.scores.E:.2f}",
                tier=f"{item.scores.T:.2f}",
                aud=f"{item.scores.A_eff:.2f}",
                conf=_confidence_text(item),
                flags=_truncate(",".join(item.flags) or "-", 28),
                profile=item.profile_url,
            )
        )

    if not results:
        print("No results.")
        return

    print()
    for rank, item in enumerate(results, start=1):
        print(f"  {rank}. {item.reason}")
        for url in item.evidence_urls[:_MAX_EVIDENCE_LINES]:
            print(f"     {url}")


def _result_payload(item: RankedKol) -> dict[str, Any]:
    return item.model_dump(mode="json")


def _json_payload(final_state: dict[str, Any]) -> dict[str, Any]:
    brief = final_state.get("brand_profile")
    if isinstance(brief, BrandProfile):
        brief = brief.model_dump(mode="json")
    return {
        "stage": final_state.get("stage", "failed"),
        "search_rounds": final_state.get("search_rounds", 0),
        "search_queries": final_state.get("search_queries") or [],
        "counts": {
            "discovered": len(final_state.get("discovered") or []),
            "candidates": len(final_state.get("candidates") or []),
            "ranked": len(final_state.get("ranked") or []),
            "results": len(final_state.get("results") or []),
        },
        "brief": brief,
        "rejected": final_state.get("rejected") or {},
        "results": [
            _result_payload(result)
            for result in final_state.get("results") or []
        ],
        "cost_log": final_state.get("cost_log") or [],
        "errors": final_state.get("errors") or [],
    }


def _print_rejected(rejected: dict[str, int]) -> None:
    if rejected:
        counts = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejected.items())
        )
    else:
        counts = "(none)"
    print(f"rejected: {counts}")


def _print_cost_log(lines: list[str]) -> None:
    print("cost log:")
    for line in lines:
        print(f"  {line}")
    if not lines:
        print("  (none)")


def _cmd_match(args: argparse.Namespace) -> int:
    try:
        discovery = DiscoveryLimits.from_env()
    except ApifyConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    min_candidates = (
        args.min_candidates
        if args.min_candidates is not None
        else discovery.min_thai_candidates
    )
    max_rounds = (
        args.max_rounds
        if args.max_rounds is not None
        else discovery.max_rounds
    )
    if max_rounds < 0:
        print("error: --max-rounds must be zero or greater", file=sys.stderr)
        return 2
    if (
        args.max_results_per_query is not None
        and args.max_results_per_query < 1
    ):
        print(
            "error: --max-results-per-query must be at least 1",
            file=sys.stderr,
        )
        return 2
    if args.max_enrich is not None and args.max_enrich < 1:
        print("error: --max-enrich must be at least 1", file=sys.stderr)
        return 2
    if args.live_kols and not os.environ.get("APIFY_TOKEN", "").strip():
        print(
            "error: --live-kols requires APIFY_TOKEN in the environment",
            file=sys.stderr,
        )
        return 2
    if args.live_fb and not os.environ.get("APIFY_TOKEN", "").strip():
        print(
            "error: --live-fb requires APIFY_TOKEN in the environment",
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
        "min_candidates": min_candidates,
        "max_search_rounds": max_rounds,
    }
    if args.max_enrich is not None:
        graph_kwargs["cfg"] = replace(
            DEFAULT_CONFIG,
            enrich_top_n=args.max_enrich,
        )
    # Default stays on build_graph's mocked bundle so the CLI never opens a
    # socket or loads a model unless a live flag is set.
    if args.live_web or args.live_kols or args.live_llm or args.live_embed or args.live_fb:
        try:
            graph_kwargs["integrations"] = build_integrations(
                live_web=args.live_web,
                live_kols=args.live_kols,
                live_llm=args.live_llm,
                live_embed=args.live_embed,
                live_fb=args.live_fb,
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
        _print_table(
            final_state.get("results") or [],
            final_state.get("brand_profile"),
        )
        _print_rejected(final_state.get("rejected") or {})
        _print_cost_log(final_state.get("cost_log") or [])

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
