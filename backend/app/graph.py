from langgraph.graph import END, START, StateGraph

from .integrations import Integrations, mock_integrations
from .matching_config import DEFAULT_CONFIG, MatchingConfig
from .models import MatchRequest
from .nodes import make_nodes
from .state import MatchState

# Discovery keeps searching until this many Thai creators are found.
MIN_CANDIDATES = 30
MAX_SEARCH_ROUNDS = 3

_NODE_ORDER = (
    "scrape_website",
    "scrape_facebook",
    "analyze_brand",
    "generate_queries",
    "discover_kols",
    "hard_filter",
    "first_pass_rank",
    "enrich_kols",
    "judge_and_rank",
    "rank_and_output",
)


def _stop_if_failed(next_node: str):
    """Return a router that stops the graph after a fatal node failure.

    A node exception caught by ``_guarded`` sets ``stage`` to ``"failed"``.
    That is fatal: the graph must stop immediately instead of running the
    next node. Warnings only append to ``errors`` and leave ``stage`` alone,
    so the router continues to ``next_node``.
    """

    def router(state: MatchState) -> str:
        if state.stage == "failed":
            return END
        return next_node

    router.__name__ = f"stop_if_failed_before_{next_node}"
    return router


def build_graph(
    integrations: Integrations | None = None,
    *,
    min_candidates: int = MIN_CANDIDATES,
    max_search_rounds: int = MAX_SEARCH_ROUNDS,
    cfg: MatchingConfig = DEFAULT_CONFIG,
    enrich_limit: int | None = None,
):
    """Build and compile the KOL-matching graph."""
    integrations = integrations or mock_integrations()
    nodes = make_nodes(
        integrations,
        min_candidates=min_candidates,
        max_search_rounds=max_search_rounds,
        cfg=cfg,
        enrich_limit=enrich_limit,
    )

    graph = StateGraph(MatchState)
    for name in _NODE_ORDER:
        graph.add_node(name, nodes[name])

    # Both scrapers start together; the list form makes analyze_brand wait for
    # both of them to finish (fan-in).
    graph.add_edge(START, "scrape_website")
    graph.add_edge(START, "scrape_facebook")
    graph.add_edge(["scrape_website", "scrape_facebook"], "analyze_brand")

    def _edge(source: str, target: str) -> None:
        graph.add_conditional_edges(
            source,
            _stop_if_failed(target),
            {target: target, END: END},
        )

    # Fatal node failures (stage "failed") end the graph. discover_kols uses
    # its own router; rank_and_output is the sink.
    _edge("analyze_brand", "generate_queries")
    _edge("generate_queries", "discover_kols")
    graph.add_conditional_edges(
        "discover_kols",
        nodes["route_after_discover"],
        {
            "generate_queries": "generate_queries",
            "hard_filter": "hard_filter",
            END: END,
        },
    )
    _edge("hard_filter", "first_pass_rank")
    _edge("first_pass_rank", "enrich_kols")
    _edge("enrich_kols", "judge_and_rank")
    _edge("judge_and_rank", "rank_and_output")
    graph.add_edge("rank_and_output", END)

    return graph.compile()


def initial_state(request: MatchRequest) -> MatchState:
    """Create the starting state for one matching request."""
    return MatchState(request=request)
