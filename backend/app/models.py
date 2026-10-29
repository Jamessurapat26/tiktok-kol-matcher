from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from .matching_config import DEFAULT_CONFIG


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


MAX_RECENT_POSTS = 20
MAX_POST_CHARS = 1_000


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
    recent_posts: list[str] = Field(default_factory=list)

    @field_validator("recent_posts")
    @classmethod
    def _cap_posts(cls, posts: list[str]) -> list[str]:
        """Collapse whitespace, drop blank posts, clip each, keep the newest few."""
        compact = (" ".join(post.split())[:MAX_POST_CHARS].rstrip() for post in posts)
        return [post for post in compact if post][:MAX_RECENT_POSTS]


class TargetAudience(DomainModel):
    gender: Literal["male", "female", "any"]
    age_range: str
    interests: list[str]


class BrandProfile(DomainModel):
    """Structured brand brief returned by the brand-analysis LLM call.

    ``brand_names`` has a default so stored profiles and fixtures written
    before that field existed stay valid. ``target_followers`` is boundary
    validated because the value comes from the model.
    """

    category: str
    key_products: list[str]
    price_tier: Literal["budget", "mid", "premium", "luxury"]
    target_audience: TargetAudience
    tone: str
    topic_keywords_th: list[str]
    topic_keywords_en: list[str]
    competitors: list[str]
    brand_handles: list[str]
    brand_names: list[str] = Field(default_factory=list)
    target_followers: int
    tier_label: Literal["nano", "micro", "mid", "macro"]
    tier_confidence: float = Field(ge=0, le=1)
    tier_reason: str

    @field_validator("target_followers")
    @classmethod
    def _clamp_target_followers(cls, value: int) -> int:
        return min(
            max(value, DEFAULT_CONFIG.target_followers_min),
            DEFAULT_CONFIG.target_followers_max,
        )


def normalize_handle(handle: str) -> str:
    """Comparable handle key: strip, lowercase, and drop leading '@'s."""
    return handle.strip().lower().lstrip("@")


# Higher wins when the same creator is seen from more than one source.
# Unknown sources stay at 0 so they never replace a known row.
SOURCE_PRIORITY: dict[str, int] = {"apify": 3, "seed": 2, "cache": 1}


def outranks(new_source: str, current_source: str) -> bool:
    """Return True when ``new_source`` has strictly higher priority.

    Equal priority keeps the earlier record. Unknown sources have priority 0.
    """
    return SOURCE_PRIORITY.get(new_source, 0) > SOURCE_PRIORITY.get(current_source, 0)


class VideoStats(DomainModel):
    """One recent video's public counts and metadata."""

    url: str | None = None
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    posted_at: datetime | None = None
    caption: str = ""

    @field_validator("posted_at")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


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
    videos: list[VideoStats] = Field(default_factory=list)
    stats_source: Literal["search", "profile"] = "search"
    source: Literal["apify", "seed", "cache"]

    @field_validator("handle")
    @classmethod
    def _clean_handle(cls, value: str) -> str:
        # Preserve case. Strip again after removing '@' so "  @Bar " -> "Bar".
        cleaned = value.strip().lstrip("@").strip()
        if not cleaned:
            raise ValueError("handle must not be empty")
        return cleaned


class RankedKol(DomainModel):
    kol: KolProfile
    scores: ScoreBreakdown
    total_score: float = Field(ge=0, le=1)
    reason: str
