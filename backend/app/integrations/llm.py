import re
from typing import Literal, Protocol

from ..models import BrandProfile, FacebookPage, KolProfile, ScoreBreakdown, TargetAudience

_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+")
_TIER_ORDER: tuple[str, ...] = ("nano", "micro", "mid", "macro")
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
    if followers >= 500_000:
        return "macro"
    if followers >= 100_000:
        return "mid"
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
        *,
        similarities: list[float] | None = None,
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        """Score each KOL and return a short reason for the match.

        ``similarities`` are cosine scores from the embedder, aligned with
        ``kols``. The mock uses them as relevance. A live model still scores
        relevance itself and can read the same scores in its prompt.
        """
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
    *,
    relevance: float | None = None,
) -> tuple[ScoreBreakdown, str]:
    """Score one KOL with the offline engagement, audience, and tier rules.

    When ``relevance`` is omitted, relevance is the offline topic overlap.
    When the embedder passes a cosine score, that score is the relevance so
    the prefilter and the mock reranker agree. The live reranker still
    replaces relevance and the reason. Engagement, audience fit, and tier fit
    stay on this breakdown so the 0.4/0.25/0.2/0.15 weights still apply.
    """
    brand_tokens = _tokens(
        " ".join(
            [
                *brand.topic_keywords_en,
                *brand.topic_keywords_th,
                *brand.key_products,
                brand.category,
            ]
        )
    )
    matched = sorted(_kol_topics(kol) & brand_tokens)
    relevance_from_embedding = relevance is not None
    if relevance is None:
        relevance = min(1.0, 0.2 + 0.2 * len(matched)) if matched else 0.05
    else:
        relevance = min(1.0, max(0.0, float(relevance)))
        matched = []

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
    preferred = brand.tier_label
    if tier == preferred:
        tier_fit = 1.0
    else:
        distance = abs(_TIER_ORDER.index(tier) - _TIER_ORDER.index(preferred))
        tier_fit = 0.5 if distance == 1 else 0.15

    scores = ScoreBreakdown(
        relevance=round(relevance, 4),
        engagement=round(engagement, 4),
        audience_fit=round(audience_fit, 4),
        tier_fit=round(tier_fit, 4),
        audience_confidence=confidence,
    )
    if relevance_from_embedding:
        reason = (
            f"Embedding similarity {relevance:.2f}. "
            f"Tier: {tier}; audience fit: {confidence}."
        )
    elif matched:
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
            tier_reason = (
                f"Mock analysis derived from the {' and '.join(sources)}: "
                "a small Bangkok online shop, so a micro creator around 30k."
            )
        else:
            tier_reason = "No source content was available; mock brand defaults used."

        return BrandProfile(
            category="gaming eyewear / blue-light glasses",
            brand_names=["ExampleBrand"],
            key_products=[
                "blue-light glasses",
                "gaming glasses",
                "clip-on blue-light lenses",
            ],
            price_tier="mid",
            target_audience=TargetAudience(
                gender="any",
                age_range="16-34",
                interests=["gaming", "esports", "tech", "lifestyle", "eyewear"],
            ),
            tone="energetic, friendly, gamer-coded",
            topic_keywords_th=[
                "แว่นกันแสงสีฟ้า",
                "แว่นเกมมิ่ง",
                "ถนอมสายตา",
                "เกมเมอร์",
                "เทคโนโลยี",
            ],
            topic_keywords_en=[
                "blue light glasses",
                "gaming glasses",
                "eyewear",
                "screen protection",
                "gaming",
                "tech",
                "esports",
            ],
            competitors=["OtherLens", "FocusGear"],
            brand_handles=["@ExampleBrand", "examplebrand"],
            target_followers=30_000,
            tier_label="micro",
            tier_confidence=0.7,
            tier_reason=tier_reason,
        )

    def generate_queries(
        self,
        brand: BrandProfile,
        previous_queries: list[str],
        round: int,
    ) -> list[str]:
        if round <= 1:
            # Round 1: broad topic and competitor searches.
            candidates = [
                *(f"{keyword} thailand" for keyword in brand.topic_keywords_en[:3]),
                *(f"{keyword} รีวิว" for keyword in brand.topic_keywords_th[:3]),
                *(f"{competitor} รีวิว" for competitor in brand.competitors[:2]),
            ]
        else:
            # Round 2: adjacent topic and competitor searches.
            candidates = [
                *(
                    f"{keyword} creator thailand"
                    for keyword in brand.topic_keywords_en[3:6]
                ),
                *(f"{keyword} creator" for keyword in brand.topic_keywords_th[3:6]),
                *(f"{competitor} tiktok" for competitor in brand.competitors[2:4]),
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
        *,
        similarities: list[float] | None = None,
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        if similarities is not None and len(similarities) != len(kols):
            raise ValueError("similarities must have one score per creator")
        results: list[tuple[KolProfile, ScoreBreakdown, str]] = []
        for index, kol in enumerate(kols):
            override = None if similarities is None else similarities[index]
            scores, reason = heuristic_score(brand, kol, relevance=override)
            results.append((kol, scores, reason))
        return results
