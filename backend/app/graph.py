from langgraph.graph import END, START, StateGraph

from .integrations import Integrations, mock_integrations
from .models import MatchRequest
from .nodes import make_nodes
from .state import MatchState

MIN_CANDIDATES = 30
MAX_SEARCH_ROUNDS = 2
PREFILTER_N = 25

_NODE_ORDER = (
    "scrape_website",
    "scrape_facebook",
    "analyze_brand",
    "generate_queries",
    "discover_kols",
    "filter_thai",
    "enrich_kols",
    "embed_and_prefilter",
    "llm_rerank",
    "rank_and_output",
)


def build_graph(
    integrations: Integrations | None = None,
    *,
    min_candidates: int = MIN_CANDIDATES,
    max_search_rounds: int = MAX_SEARCH_ROUNDS,
    prefilter_n: int = PREFILTER_N,
):
    """Build and compile the mock KOL-matching graph."""
    integrations = integrations or mock_integrations()
    nodes = make_nodes(
        integrations,
        min_candidates=min_candidates,
        max_search_rounds=max_search_rounds,
        prefilter_n=prefilter_n,
    )

    graph = StateGraph(MatchState)
    for name in _NODE_ORDER:
        graph.add_node(name, nodes[name])

    # Both scrapers start together; the list form makes analyze_brand wait for
    # both of them to finish (fan-in).
    graph.add_edge(START, "scrape_website")
    graph.add_edge(START, "scrape_facebook")
    graph.add_edge(["scrape_website", "scrape_facebook"], "analyze_brand")

    graph.add_edge("analyze_brand", "generate_queries")
    graph.add_edge("generate_queries", "discover_kols")
    graph.add_conditional_edges(
        "discover_kols",
        nodes["route_after_discover"],
        {
            "generate_queries": "generate_queries",
            "filter_thai": "filter_thai",
        },
    )
    graph.add_edge("filter_thai", "enrich_kols")
    graph.add_edge("enrich_kols", "embed_and_prefilter")
    graph.add_edge("embed_and_prefilter", "llm_rerank")
    graph.add_edge("llm_rerank", "rank_and_output")
    graph.add_edge("rank_and_output", END)

    return graph.compile()


def initial_state(request: MatchRequest) -> MatchState:
    """Create the starting state for one matching request."""
    return MatchState(request=request)
