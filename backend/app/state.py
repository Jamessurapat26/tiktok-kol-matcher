import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    BrandProfile,
    FacebookPage,
    KolProfile,
    MatchRequest,
    RankedKol,
    normalize_handle,
    outranks,
)


# Identity fields always come from the winning source. Optional fields that are
# None on the winner are filled from a strictly lower-priority row. Zero is a
# real value (followers, posts_per_week), so it is not treated as missing.
_WINNER_FIELDS = frozenset({"handle", "url", "source"})


def _fill_missing(winner: KolProfile, donor: KolProfile) -> KolProfile:
    """Copy fields that are None on ``winner`` from a lower-priority ``donor``."""
    updates: dict[str, object] = {}
    for name in type(winner).model_fields:
        if name in _WINNER_FIELDS:
            continue
        if getattr(winner, name) is None and getattr(donor, name) is not None:
            updates[name] = getattr(donor, name)
    if not updates:
        return winner
    return winner.model_copy(update=updates)


def merge_kols(left: list[KolProfile], right: list[KolProfile]) -> list[KolProfile]:
    """Fan-in reducer: dedupe by handle, keeping the highest-priority source.

    The winning row keeps its own values. Fields that are ``None`` on the winner
    are filled from strictly lower-priority rows (including values those rows
    already inherited). Equal priority keeps the earlier row unchanged.

    LangGraph passes node updates to reducers as raw values, so list items may be
    dicts even though the state annotation says ``KolProfile``. Coerce them here.
    """
    merged: dict[str, KolProfile] = {}
    for raw in [*left, *right]:
        kol = raw if isinstance(raw, KolProfile) else KolProfile.model_validate(raw)
        key = normalize_handle(kol.handle)
        current = merged.get(key)
        if current is None:
            merged[key] = kol
        elif outranks(kol.source, current.source):
            merged[key] = _fill_missing(kol, current)
        elif outranks(current.source, kol.source):
            merged[key] = _fill_missing(current, kol)
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
    # Queries from the latest round only. Overwritten each round so discover_kols
    # searches this list instead of the accumulated search_queries history.
    last_queries: list[str] = Field(default_factory=list)
    search_rounds: int = Field(default=0, ge=0)

    # Raw output of discover_kols, accumulated across query rounds. The reducer
    # dedupes by normalized handle (strip whitespace, lowercase, drop leading "@")
    # with priority apify > seed > cache. None fields are filled from lower-priority rows.
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
