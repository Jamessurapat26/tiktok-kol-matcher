"""Prompt text for the live brand analyzer.

The model must put one JSON object in the visible reply. Reasoning models can
spend the completion budget on a hidden trace and return an empty reply, so
the instructions say the JSON belongs in the answer itself.
"""

from __future__ import annotations

from .models import BrandProfile, FacebookPage, KolProfile

_SOURCE_CHARS = 12_000
_BIO_CHARS = 300
_SUMMARY_CHARS = 400
MAX_PROMPT_POSTS = 3
MAX_PROMPT_POST_CHARS = 200

BRAND_ANALYSIS_SYSTEM = """\
You extract a structured brand profile for a brand. Reply with one JSON object
and nothing else. Do not wrap the JSON in markdown. Put the JSON in the visible
reply itself, not only in a reasoning trace.

Use the sources in this order of trust:
1. The website text and the Facebook page name, categories, intro, and website
   are the primary sources. Use them for products, category, and audience.
2. Recent Facebook posts are a weak signal for tone and current campaigns only.
   Never make a post the sole source of a product. Posts often mention items
   that are not the brand's own products, such as giveaways, collaborations,
   partner products, and memes. Leave those items out.

The object must contain exactly these keys:
- category: short product category
- products: the brand's own core product lines or categories, at most 8 items.
  Each item is a short noun phrase of at most 4 words.
  Prefer the primary sources over posts.
- target_audience: object with age_min (integer or null), age_max (integer or
  null), gender ("male", "female", or "any"), interests (at most 8 short
  phrases of at most 3 words each), and location
- tone: how the brand speaks
- content_style: the kind of social content that fits the brand
- keywords_th: Thai search phrases in Thai script, at most 12 items.
  Each is a short search phrase of at most 3 words, written
  without sentence structure, without verbs with clauses, and without
  punctuation.
- keywords_en: English search phrases, at most 12 items. Each is a
  short search phrase of at most 3 words, without sentence
  structure or punctuation.
- hashtags: at most 10 hashtags. Each starts with # and has no
  spaces.
- preferred_tiers: list using only "nano", "micro", "macro", and "mega"
- reasoning: one or two sentences on how you read the sources

Give a best-estimate age_min and age_max from the brand type and audience cues.
Use null for an age bound only when there is truly no signal. Prefer the
primary sources over guesses. If Facebook text is missing, say so in reasoning
and still return the profile.
"""

# Each query becomes one paid Apify actor run, so the live analyzer enforces
# this bound even when the model ignores the prompt.
MAX_QUERIES = 10

QUERY_SYSTEM = f"""\
You write TikTok search queries that find creators for a brand. Reply with one
JSON object and nothing else, in the visible reply, not only in a reasoning
trace. Do not wrap it in markdown.

Shape: {{"queries": ["...", "..."]}}

Each query is a short TikTok search string. Mix Thai (Thai script) and
English. Do not repeat any query from the previous list, even with different
casing or extra spaces. Round 1 is broad product and audience searches. Later
rounds expand into hashtags, adjacent topics, and creator-style searches that
were not used yet. Return between 6 and {MAX_QUERIES} queries.
"""

RERANK_SYSTEM = """\
You score how relevant each TikTok creator is to a brand. Reply with one JSON
object and nothing else, in the visible reply, not only in a reasoning trace.
Do not wrap it in markdown.

Shape: {"scores": [{"handle": "...", "relevance": 0.0, "reason": "..."}]}

relevance is a number from 0 to 1. 1 means the creator clearly covers the
brand's products or audience. 0 means unrelated. The goal is to find creators
who could promote the brand, so score the brand's own account, other shops, and
competing brands low, however closely their products match. reason is one
short sentence in Thai or English. Include every handle from the input exactly
once. Copy each handle unchanged.
"""

_RERANK_SIMILARITY_NOTE = """
Each creator includes embedding_similarity, the cosine similarity between the
brand and that creator. Higher means closer. Use it as a hint and still set
relevance from the creator's content.
"""

JSON_RETRY_USER = (
    "That response failed validation:\n"
    "{error}\n\n"
    "Reply again with only one JSON object that matches the schema. "
    "No markdown, and put the JSON in the visible reply."
)


def brand_messages(
    website_content: str | None,
    facebook_page: FacebookPage | None,
) -> list[dict[str, str]]:
    website = _clip(website_content, _SOURCE_CHARS) or "(no website text)"
    user = (
        "Primary source: Website text\n"
        f"{website}\n\n"
        "Primary source: Facebook page (name, categories, intro, and website)\n"
        f"{_facebook_text(facebook_page)}"
    )
    return [
        {"role": "system", "content": BRAND_ANALYSIS_SYSTEM},
        {"role": "user", "content": user},
    ]


def query_messages(
    brand: BrandProfile,
    previous_queries: list[str],
    round: int,
) -> list[dict[str, str]]:
    previous = "\n".join(f"- {query}" for query in previous_queries) or "(none)"
    user = (
        f"Round: {round}\n\n"
        "Brand:\n"
        f"{_brand_summary(brand)}\n\n"
        "Queries already used (do not repeat these):\n"
        f"{previous}"
    )
    return [
        {"role": "system", "content": QUERY_SYSTEM},
        {"role": "user", "content": user},
    ]


def rerank_messages(
    brand: BrandProfile,
    kols: list[KolProfile],
    similarities: list[float] | None = None,
) -> list[dict[str, str]]:
    if similarities is not None and len(similarities) != len(kols):
        raise ValueError("similarities must have one score per creator")
    if similarities is None:
        cards = "\n\n".join(_kol_card(kol) for kol in kols)
        system = RERANK_SYSTEM
    else:
        cards = "\n\n".join(
            _kol_card(kol, similarity) for kol, similarity in zip(kols, similarities, strict=True)
        )
        system = RERANK_SYSTEM + _RERANK_SIMILARITY_NOTE
    user = (
        "Brand:\n"
        f"{_brand_summary(brand)}\n\n"
        "Creators:\n"
        f"{cards}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _brand_summary(brand: BrandProfile) -> str:
    audience = brand.target_audience
    return "\n".join(
        [
            f"category: {brand.category}",
            f"products: {', '.join(brand.products)}",
            f"tone: {brand.tone}",
            f"content_style: {brand.content_style}",
            (
                f"audience: {audience.gender}, ages {audience.age_min}-{audience.age_max}, "
                f"{audience.location}"
            ),
            f"interests: {', '.join(audience.interests)}",
            f"keywords_th: {', '.join(brand.keywords_th)}",
            f"keywords_en: {', '.join(brand.keywords_en)}",
            f"hashtags: {' '.join(brand.hashtags)}",
            f"preferred_tiers: {', '.join(brand.preferred_tiers)}",
        ]
    )


def _facebook_text(page: FacebookPage | None) -> str:
    if page is None:
        return "(no Facebook page was scraped)"
    categories = ", ".join(page.categories)
    lines = [
        f"name: {page.page_name}",
        f"categories: {categories}",
        f"intro: {_clip(page.intro, _SOURCE_CHARS)}",
        f"website: {page.website or ''}",
        f"title: {page.title}",
        f"followers: {page.followers}",
        f"price_range: {page.price_range or ''}",
    ]
    # recent_posts is newest first. Posts are a weak signal, so only the newest
    # few are sent, and each is clipped hard to keep them from dominating.
    posts = page.recent_posts[:MAX_PROMPT_POSTS]
    if posts:
        lines.append(
            "weak_signals_recent_posts (weak signal only: use for tone and"
            " current campaigns; never as the sole source of a product):"
        )
        lines.extend(
            f"- {_clip(post, MAX_PROMPT_POST_CHARS)}" for post in posts
        )
    return "\n".join(lines)


def _kol_card(kol: KolProfile, similarity: float | None = None) -> str:
    if kol.median_engagement_rate is None:
        rate = "unknown"
    else:
        rate = f"{kol.median_engagement_rate:.4f}"
    lines = [
        f"handle: {kol.handle}",
        f"nickname: {kol.nickname or ''}",
        f"followers: {kol.followers}",
        f"language: {kol.language or ''}",
        f"region: {kol.region or ''}",
        f"engagement_rate: {rate}",
        f"bio: {_clip(kol.bio, _BIO_CHARS)}",
        f"content: {_clip(kol.content_summary, _SUMMARY_CHARS)}",
    ]
    if similarity is not None:
        lines.append(f"embedding_similarity: {similarity:.4f}")
    return "\n".join(lines)


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
