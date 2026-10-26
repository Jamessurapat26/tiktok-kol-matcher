import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import BrandProfile, FacebookPage, KolProfile, MatchRequest, RankedKol

_SOURCE_PRIORITY: dict[str, int] = {"apify": 3, "seed": 2, "cache": 1}


def merge_kols(left: list[KolProfile], right: list[KolProfile]) -> list[KolProfile]:
    """Fan-in reducer: dedupe by handle, keeping the highest-priority source."""
    merged: dict[str, KolProfile] = {}
    for kol in [*left, *right]:
        key = kol.handle.lower()
        current = merged.get(key)
        if (
            current is None
            or _SOURCE_PRIORITY[kol.source] > _SOURCE_PRIORITY[current.source]
        ):
            merged[key] = kol
    return list(merged.values())


def merge_ranked(left: list[RankedKol], right: list[RankedKol]) -> list[RankedKol]:
    """Fan-in reducer: dedupe by handle, keeping the highest-scoring copy."""
    merged: dict[str, RankedKol] = {}
    for ranked in [*left, *right]:
        key = ranked.kol.handle.lower()
        current = merged.get(key)
        if current is None or ranked.total_score > current.total_score:
            merged[key] = ranked
    return list(merged.values())


class MatchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: MatchRequest
    stage: Literal[
        "received",
        "enriching",
        "profiling",
        "searching",
        "scoring",
        "done",
        "failed",
    ] = "received"

    facebook_page: FacebookPage | None = None
    brand_profile: BrandProfile | None = None

    candidates: Annotated[list[KolProfile], merge_kols] = Field(default_factory=list)
    ranked: Annotated[list[RankedKol], merge_ranked] = Field(default_factory=list)
    results: list[RankedKol] = Field(default_factory=list)

    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
