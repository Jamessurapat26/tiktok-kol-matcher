import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import BrandProfile, FacebookPage, KolProfile, MatchRequest, RankedKol

_SOURCE_PRIORITY: dict[str, int] = {"apify": 3, "seed": 2, "cache": 1}


def merge_kols(left: list[KolProfile], right: list[KolProfile]) -> list[KolProfile]:
    """Fan-in reducer: dedupe by handle, keeping the highest-priority source.

    LangGraph passes node updates to reducers as raw values, so list items may be
    dicts even though the state annotation says ``KolProfile``. Coerce them here.
    """
    merged: dict[str, KolProfile] = {}
    for raw in [*left, *right]:
        kol = raw if isinstance(raw, KolProfile) else KolProfile.model_validate(raw)
        key = kol.handle.strip().lower().lstrip("@")
        current = merged.get(key)
        if (
            current is None
            or _SOURCE_PRIORITY[kol.source] > _SOURCE_PRIORITY[current.source]
        ):
            merged[key] = kol
    return list(merged.values())


class MatchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: MatchRequest
    # Pipeline stages, in graph order.
    stage: Literal[
        "received",
        "scraping",
        "analyzing_brand",
        "generating_queries",
        "discovering_kols",
        "filtering_thai",
        "enriching_kols",
        "embedding_prefilter",
        "reranking",
        "outputting",
        "done",
        "failed",
    ] = "received"

    website_content: str | None = None
    facebook_page: FacebookPage | None = None
    brand_profile: BrandProfile | None = None

    # Accumulates across rounds. generate_queries returns only the new queries for
    # the current round; it can read the prior list on state to avoid repeats.
    search_queries: Annotated[list[str], operator.add] = Field(default_factory=list)
    search_rounds: int = Field(default=0, ge=0)

    # Raw output of discover_kols, accumulated across query rounds. The reducer
    # dedupes by normalized handle (strip whitespace, lowercase, drop leading "@")
    # with priority apify > seed > cache.
    discovered: Annotated[list[KolProfile], merge_kols] = Field(default_factory=list)
    # filter_thai output. Plain overwrite (not reducer-backed) so the list can shrink.
    candidates: list[KolProfile] = Field(default_factory=list)
    # embed_and_prefilter output: the top 20-30 candidates sent to llm_rerank.
    shortlisted: list[KolProfile] = Field(default_factory=list)

    # Every candidate scored by llm_rerank, sorted by total_score.
    ranked: list[RankedKol] = Field(default_factory=list)
    # Top-k slice of ranked, returned to the caller.
    results: list[RankedKol] = Field(default_factory=list)

    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
