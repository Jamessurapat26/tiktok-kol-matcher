from typing import Literal

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
    relevance: float = Field(default=0.4, ge=0, le=1)
    engagement: float = Field(default=0.25, ge=0, le=1)
    audience: float = Field(default=0.2, ge=0, le=1)
    tier: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "MatchWeights":
        total = self.relevance + self.engagement + self.audience + self.tier
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")
        return self

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
    age_range: str
    gender: Literal["male", "female", "any"] = "any"
    interests: list[str] = Field(default_factory=list)
    location: str


class BrandProfile(DomainModel):
    category: str
    products: list[str] = Field(default_factory=list)
    target_audience: TargetAudience
    tone: str
    content_style: str
    keywords_th: list[str] = Field(default_factory=list)
    keywords_en: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    preferred_tiers: list[str] = Field(default_factory=list)
    reasoning: str


class KolProfile(DomainModel):
    handle: str
    url: HttpUrl
    nickname: str | None = None
    followers: int = Field(default=0, ge=0)
    bio: str | None = None
    language: str
    region: str
    median_engagement_rate: float = Field(
        ge=0,
        le=1,
        description="Median engagement per post as a fraction (0.043 = 4.3%).",
    )
    posts_per_week: float = Field(default=0, ge=0)
    content_summary: str
    source: Literal["apify", "seed", "cache"]


class RankedKol(DomainModel):
    kol: KolProfile
    scores: ScoreBreakdown
    total_score: float = Field(ge=0, le=1)
    reason: str
