from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScoreBreakdown(DomainModel):
    relevance: float = Field(ge=0, le=1)
    engagement: float = Field(ge=0, le=1)
    audience_fit: float = Field(ge=0, le=1)
    tier_fit: float = Field(ge=0, le=1)
    audience_confidence: Literal["low", "medium", "high"]


class MatchWeights(DomainModel):
    # Frozen so the sum==1 invariant cannot be broken by attribute assignment.
    model_config = ConfigDict(frozen=True, extra="forbid")

    relevance: float = Field(default=0.4, ge=0)
    engagement: float = Field(default=0.25, ge=0)
    audience: float = Field(default=0.2, ge=0)
    tier: float = Field(default=0.15, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Fill defaults and normalize the raw input dict before validation.

        Normalizing here (rather than in a mode="after" validator) keeps the
        invariant at construction without mutating an already-built model. No
        model is instantiated in this validator, so it cannot recurse.
        """
        if isinstance(data, cls):
            data = data.model_dump()
        if not isinstance(data, Mapping):
            return data

        try:
            values = {
                "relevance": float(data.get("relevance", 0.4)),
                "engagement": float(data.get("engagement", 0.25)),
                "audience": float(data.get("audience", 0.2)),
                "tier": float(data.get("tier", 0.15)),
            }
        except (TypeError, ValueError):
            # Let field validation coerce/report non-numeric input.
            return data

        if any(value < 0 for value in values.values()):
            # Leave negatives to the fields' ge=0 constraint.
            return data

        total = sum(values.values())
        if total <= 0:
            raise ValueError("weights must not all be zero")

        normalized = {name: value / total for name, value in values.items()}
        # Preserve unknown keys so extra="forbid" still rejects them.
        return {**data, **normalized}

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "MatchWeights":
        """Validate ``update`` so ``model_copy(update=...)`` cannot break the sum."""
        if not update:
            return super().model_copy(deep=deep)
        return type(self).model_validate({**self.model_dump(), **update})

    def apply(self, scores: ScoreBreakdown) -> float:
        return round(
            scores.relevance * self.relevance
            + scores.engagement * self.engagement
            + scores.audience_fit * self.audience
            + scores.tier_fit * self.tier,
            4,
        )


class MatchRequest(DomainModel):
    website_url: HttpUrl
    facebook_url: HttpUrl
    top_k: int = Field(default=20, ge=1, le=200)
    weights: MatchWeights = Field(default_factory=MatchWeights)


class FacebookPage(DomainModel):
    page_name: str
    title: str
    url: HttpUrl
    categories: list[str] = Field(default_factory=list)
    intro: str | None = None
    followers: int = Field(default=0, ge=0)
    price_range: str | None = None
    website: HttpUrl | None = None
    ad_running: bool = False


class TargetAudience(DomainModel):
    age_min: int | None
    age_max: int | None
    gender: Literal["male", "female", "any"]
    interests: list[str]
    location: str


class BrandProfile(DomainModel):
    """OpenAI strict structured output: every field is required (no defaults)."""

    category: str
    products: list[str]
    target_audience: TargetAudience
    tone: str
    content_style: str
    keywords_th: list[str]
    keywords_en: list[str]
    hashtags: list[str]
    preferred_tiers: list[Literal["nano", "micro", "macro", "mega"]]
    reasoning: str


class KolProfile(DomainModel):
    handle: str
    url: HttpUrl
    nickname: str | None = None
    followers: int = Field(default=0, ge=0)
    bio: str | None = None
    language: str | None = None
    region: str | None = None
    median_engagement_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Median engagement per post as a fraction (0.043 = 4.3%).",
    )
    posts_per_week: float = Field(default=0, ge=0)
    content_summary: str | None = None
    source: Literal["apify", "seed", "cache"]


class RankedKol(DomainModel):
    kol: KolProfile
    scores: ScoreBreakdown
    total_score: float = Field(ge=0, le=1)
    reason: str
