import re
from typing import Literal, Protocol

from ..models import BrandProfile, FacebookPage, KolProfile, ScoreBreakdown, TargetAudience

_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+")
_TIER_ORDER: tuple[str, ...] = ("nano", "micro", "macro", "mega")
# Engagement rates at or above this fraction score 1.0. 0.15 keeps a 9% rate
# and a 14% rate from both saturating, which a 0.08 ceiling used to do.
ENGAGEMENT_CEILING = 0.15


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def _kol_topics(kol: KolProfile) -> set[str]:
    summary = kol.content_summary or ""
    if summary.casefold().startswith("topics:"):
        raw_topics = summary.split(":", 1)[1]
        return {topic.strip().casefold() for topic in raw_topics.split(",") if topic.strip()}
    return _tokens(summary)


def _tier_for(followers: int) -> str:
    if followers >= 1_000_000:
        return "mega"
    if followers >= 100_000:
        return "macro"
    if followers >= 10_000:
        return "micro"
    return "nano"


class BrandAnalyzer(Protocol):
    def analyze_brand(
        self,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandProfile:
        """Build a structured brand profile from the scraped sources."""
        ...

    def generate_queries(
        self,
        brand: BrandProfile,
        previous_queries: list[str],
        round: int,
    ) -> list[str]:
        """Return only queries that were not used in earlier rounds."""
        ...

    def rerank(
        self,
        brand: BrandProfile,
        kols: list[KolProfile],
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        """Score each KOL and return a short reason for the match."""
        ...


class ChatClient(Protocol):
    """OpenAI-compatible chat completions client.

    ``complete`` returns the assistant text. Missing or blank content is an
    error: implementations retry that case once, then raise.
    """

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant message content."""
        ...


def heuristic_score(
    brand: BrandProfile,
    kol: KolProfile,
) -> tuple[ScoreBreakdown, str]:
    """Score one KOL with the offline token, engagement, audience, and tier rules.

    The live reranker replaces ``relevance`` and the reason. Engagement,
    audience fit, and tier fit stay on this breakdown so the 0.4/0.25/0.2/0.15
    weights still apply the same way.
    """
    brand_tokens = _tokens(
        " ".join(
            [
                *brand.keywords_en,
                *brand.keywords_th,
                *brand.hashtags,
                *brand.products,
                brand.category,
            ]
        )
    )
    matched = sorted(_kol_topics(kol) & brand_tokens)
    relevance = min(1.0, 0.2 + 0.2 * len(matched)) if matched else 0.05

    engagement_rate = kol.median_engagement_rate or 0.02
    engagement = min(1.0, engagement_rate / ENGAGEMENT_CEILING)

    thai_language = (kol.language or "").casefold() in {"th", "thai"}
    thai_region = (kol.region or "").upper() == "TH"
    if thai_language and thai_region:
        audience_fit: float = 1.0
        confidence: Literal["low", "medium", "high"] = "high"
    elif thai_language or thai_region:
        audience_fit = 0.75
        confidence = "medium"
    else:
        audience_fit = 0.35
        confidence = "low"

    tier = _tier_for(kol.followers)
    if not brand.preferred_tiers:
        tier_fit = 0.5
    elif tier in brand.preferred_tiers:
        tier_fit = 1.0
    else:
        distance = min(
            abs(_TIER_ORDER.index(tier) - _TIER_ORDER.index(preferred))
            for preferred in brand.preferred_tiers
        )
        tier_fit = 0.5 if distance == 1 else 0.15

    scores = ScoreBreakdown(
        relevance=round(relevance, 4),
        engagement=round(engagement, 4),
        audience_fit=round(audience_fit, 4),
        tier_fit=round(tier_fit, 4),
        audience_confidence=confidence,
    )
    if matched:
        reason = (
            f"Matched topics: {', '.join(matched[:4])}. "
            f"Tier: {tier}; audience fit: {confidence}."
        )
    else:
        reason = (
            "No direct topic overlap; broad creator fit. "
            f"Tier: {tier}; audience fit: {confidence}."
        )
    return scores, reason


class MockBrandAnalyzer:
    """MOCK: replace with real implementation (LLM calls)."""

    def analyze_brand(
        self,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandProfile:
        sources = []
        if website_content:
            sources.append("website")
        if facebook_page is not None:
            sources.append("Facebook page")

        if sources:
            reasoning = (
                f"Mock analysis derived from the {' and '.join(sources)}: "
                "a Bangkok gaming eyewear brand selling blue-light glasses to "
                "young Thai gamers, students and screen workers."
            )
        else:
            reasoning = "No source content was available; mock brand defaults used."

        return BrandProfile(
            category="gaming eyewear / blue-light glasses",
            products=[
                "blue-light glasses",
                "gaming glasses",
                "clip-on blue-light lenses",
            ],
            target_audience=TargetAudience(
                age_min=16,
                age_max=34,
                gender="any",
                interests=["gaming", "esports", "tech", "lifestyle", "eyewear"],
                location="Thailand (Bangkok)",
            ),
            tone="energetic, friendly, gamer-coded",
            content_style="short-form reviews, gameplay commentary and desk setups",
            keywords_th=[
                "แว่นกันแสงสีฟ้า",
                "แว่นเกมมิ่ง",
                "ถนอมสายตา",
                "เกมเมอร์",
                "เทคโนโลยี",
                "ไลฟ์สไตล์",
            ],
            keywords_en=[
                "blue light glasses",
                "gaming glasses",
                "eyewear",
                "screen protection",
                "gaming",
                "tech",
                "esports",
                "lifestyle",
            ],
            hashtags=[
                "#Ophtus",
                "#BlueLightGlasses",
                "#GamingGlasses",
                "#แว่นกันแสงสีฟ้า",
                "#เกมเมอร์",
            ],
            preferred_tiers=["micro", "macro", "mega"],
            reasoning=reasoning,
        )

    def generate_queries(
        self,
        brand: BrandProfile,
        previous_queries: list[str],
        round: int,
    ) -> list[str]:
        if round <= 1:
            # Round 1: broad Thai + English product and audience keywords.
            candidates = [
                *(f"{keyword} thailand" for keyword in brand.keywords_en[:3]),
                *(f"{keyword} รีวิว" for keyword in brand.keywords_th[:3]),
                *(
                    f"{interest} creator thailand"
                    for interest in brand.target_audience.interests[:2]
                ),
            ]
        else:
            # Round 2: hashtag-led and adjacent creator searches.
            candidates = [
                *(f"{hashtag} tiktok" for hashtag in brand.hashtags[:4]),
                *(
                    f"{keyword} creator thailand"
                    for keyword in brand.keywords_en[3:6]
                ),
                *(f"{hashtag} รีวิว" for hashtag in brand.hashtags[3:5]),
            ]

        previous = {query.strip().casefold() for query in previous_queries}
        new_queries: list[str] = []
        for query in candidates:
            key = query.strip().casefold()
            if key and key not in previous:
                previous.add(key)
                new_queries.append(query)
        return new_queries

    def rerank(
        self,
        brand: BrandProfile,
        kols: list[KolProfile],
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        results: list[tuple[KolProfile, ScoreBreakdown, str]] = []
        for kol in kols:
            scores, reason = heuristic_score(brand, kol)
            results.append((kol, scores, reason))
        return results
