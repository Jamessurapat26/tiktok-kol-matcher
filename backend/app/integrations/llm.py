from typing import Protocol

from ..brand_match import normalize_for_match
from ..models import (
    BrandProfile,
    ContentAnalysis,
    FacebookPage,
    Judgement,
    KolProfile,
    TargetAudience,
)
from ..prompts import judge_videos

_RISKY_WORDS = ("gambling", "พนัน", "casino", "บาคาร่า")
_AUDIENCE_HITS_FOR_ONE = 2
_TOPIC_HITS_FOR_ONE = 3
_FORMAT_WORDS = {
    "review": ("รีวิว", "review"),
    "tutorial": ("สอน", "วิธี", "how to", "tutorial"),
    "unboxing": ("แกะกล่อง", "unbox"),
    "comparison": ("เทียบ", " vs "),
    "vlog": ("vlog", "ชีวิตประจำวัน"),
    "comedy": ("ตลก", "ฮา", "comedy"),
}
_FORMATS_FOR_FULL_FIT = 2


class BrandAnalyzer(Protocol):
    def analyze_brand(
        self,
        website_content: str | None,
        facebook_page: FacebookPage | None,
        research: str | None = None,
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

    def judge(
        self,
        brief: BrandProfile,
        kols: list[KolProfile],
    ) -> list[Judgement]:
        """Judge all given creators in one batched call.

        The returned list contains only creators the model judged. A missing
        creator is unjudged, not an error.
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


def _brief_terms(brief: BrandProfile) -> list[str]:
    category_parts = brief.category.replace("/", ",").split(",")
    terms: list[str] = []
    seen: set[str] = set()
    for raw in (
        *brief.topic_keywords_th,
        *brief.topic_keywords_en,
        *brief.key_products,
        *category_parts,
    ):
        term = normalize_for_match(str(raw))
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _matches(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in text]


def _judgement_reason(matched: list[str], audience_hits: list[str]) -> str:
    if matched:
        topics = ", ".join(matched[:3])
        reason = f"เนื้อหาตรงกับหัวข้อของแบรนด์: {topics}"
        if audience_hits:
            reason += f" และกลุ่มผู้ชมสนใจ: {', '.join(audience_hits[:2])}"
        return reason
    if audience_hits:
        return f"กลุ่มผู้ชมสนใจ: {', '.join(audience_hits[:2])}"
    return "ไม่พบหัวข้อของแบรนด์ในเนื้อหาของ KOL"


def _mock_content(brief: BrandProfile, kol: KolProfile) -> ContentAnalysis | None:
    """Read formats from caption keywords, so demo mode can show this section."""
    formats: list[str] = []
    evidence: list[str] = []
    for video in judge_videos(kol):
        caption = video.caption.casefold()
        for name, words in _FORMAT_WORDS.items():
            if any(word in caption for word in words):
                if name not in formats:
                    formats.append(name)
                if video.url and video.url not in evidence:
                    evidence.append(video.url)
    if not formats:
        return None
    shared = [name for name in formats if name in brief.content_formats]
    return ContentAnalysis(
        formats=formats[:3],
        style=f"โพสต์แนว {', '.join(formats[:3])} ตามแคปชันล่าสุด",
        style_fit=min(1.0, len(shared) / _FORMATS_FOR_FULL_FIT),
        style_reason=(
            f"รูปแบบตรงกับแบรนด์: {', '.join(shared)}"
            if shared
            else "รูปแบบคอนเทนต์ต่างจากที่แบรนด์ต้องการ"
        ),
        evidence_urls=evidence[:3],
    )


class MockBrandAnalyzer:
    """MOCK: replace with real implementation (LLM calls)."""

    def analyze_brand(
        self,
        website_content: str | None,
        facebook_page: FacebookPage | None,
        research: str | None = None,
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
            tier_reason_th = (
                "ร้านออนไลน์ขนาดเล็กในกรุงเทพฯ จึงเหมาะกับ KOL ระดับ Micro "
                "ที่มีผู้ติดตามราว 30K"
            )
        else:
            tier_reason = "No source content was available; mock brand defaults used."
            tier_reason_th = "ไม่มีข้อมูลแบรนด์ จึงใช้ค่าตัวอย่าง"

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
            content_formats=["review", "comparison", "tutorial"],
            content_style="Hands-on gear reviews with real gaming setups.",
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
            tier_label="micro",
            tier_confidence=0.7,
            tier_reason=tier_reason,
            category_th="แว่นเกมมิ่ง / แว่นกรองแสงสีฟ้า",
            key_products_th=["แว่นกรองแสงสีฟ้า", "แว่นเกมมิ่ง", "คลิปออนกรองแสงสีฟ้า"],
            tone_th="สนุก เป็นกันเอง สไตล์เกมเมอร์",
            content_style_th="รีวิวอุปกรณ์จากการใช้งานจริงในเซ็ตอัปเล่นเกม",
            interests_th=["เกม", "อีสปอร์ต", "เทคโนโลยี", "ไลฟ์สไตล์", "แว่นตา"],
            tier_reason_th=tier_reason_th,
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

    def judge(
        self,
        brief: BrandProfile,
        kols: list[KolProfile],
    ) -> list[Judgement]:
        terms = _brief_terms(brief)
        interests = [
            term
            for term in (
                normalize_for_match(str(interest))
                for interest in brief.target_audience.interests
            )
            if len(term) >= 2
        ]
        judgements: list[Judgement] = []
        for kol in kols:
            captions = " ".join(video.caption for video in kol.videos)
            profile_text = " ".join(
                part for part in (kol.nickname, kol.bio, captions) if part
            )
            text = normalize_for_match(profile_text)
            matched = _matches(text, terms)
            audience_text = normalize_for_match(
                " ".join(part for part in (kol.bio, captions) if part)
            )
            audience_hits = _matches(audience_text, interests)
            risky = any(word in captions.casefold() for word in _RISKY_WORDS)
            judgements.append(
                Judgement(
                    handle=kol.handle,
                    judge=min(1.0, len(matched) / _TOPIC_HITS_FOR_ONE),
                    audience=min(1.0, len(audience_hits) / _AUDIENCE_HITS_FOR_ONE),
                    audience_confidence=min(0.9, len(profile_text) / 1500),
                    reason=_judgement_reason(matched, audience_hits),
                    risky=risky,
                    content=_mock_content(brief, kol),
                )
            )
        return judgements
