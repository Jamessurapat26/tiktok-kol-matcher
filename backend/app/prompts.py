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
- brand_names: the brand's own proper names and common aliases or
  abbreviations as written in social posts, in Latin and Thai script, at
  most 8 items. Proper names only, never generic words like the category.
- key_products: the brand's own core product lines or categories, at most 8
  items. Each item is a short noun phrase of at most 4 words.
  Prefer the primary sources over posts.
- price_tier: one of "budget", "mid", "premium", or "luxury".
- target_audience: object with gender ("male", "female", or "any"),
  age_range (a short string such as "18-34" or "any"), and interests
  (at most 8 short phrases of at most 3 words each)
- tone: how the brand speaks
- topic_keywords_th: Thai search phrases in Thai script, at most 12 items.
  Each is a short search phrase of at most 3 words, written
  without sentence structure, without verbs with clauses, and without
  punctuation.
- topic_keywords_en: English search phrases, at most 12 items. Each is a
  short search phrase of at most 3 words, without sentence
  structure or punctuation.
- competitors: names of competing brands, at most 8 items. Proper names only.
- brand_handles: the brand's own social handles, at most 8 items, with or
  without a leading @.
- target_followers: your best integer estimate of the creator follower count
  this brand should target. It must be between 10000 and 2000000.
- tier_label: one of "nano", "micro", "mid", or "macro".
- tier_confidence: a number from 0 to 1 for how sure you are of the tier.
- tier_reason: one short sentence explaining the tier choice.

Tier guidance:
- SME or small online shop -> micro, about 30,000 followers.
- Nationwide Thai brand -> mid, about 150,000 followers.
- Multinational or very large brand -> macro, about 800,000 followers.
Use the price tier, the sales channels (online-only, small stores, or
nationwide convenience stores), whether the brand is local or multinational,
and the brand page follower count when it is available. Show the tier_label
and tier_reason in the JSON.

Prefer the primary sources over guesses. If Facebook text is missing, still
return the best brief you can from the website.
"""

# Each query becomes one paid Apify actor run, so the live analyzer enforces
# this bound even when the model ignores the prompt.
MAX_QUERIES = 10

QUERY_SYSTEM = f"""\
You write TikTok search queries that find creators for a brand. Reply with one
JSON object and nothing else, in the visible reply, not only in a reasoning
trace. Do not wrap it in markdown.

Shape: {{"queries": ["...", "..."]}}

Each query is a short TikTok search string built from the brand's topic
keywords or competitor names. Mix Thai (Thai script) and English. Never use
the brand's own name, brand names, or brand handles in a query. Do not repeat
any query from the previous list, even with different casing or extra spaces.
Round 1 is broad topic and competitor searches. Later rounds expand into
adjacent topics and creator-style searches that were not used yet. Return
between 6 and {MAX_QUERIES} queries.
"""

JUDGE_SYSTEM = """\
You judge how well TikTok creators fit a brand. Reply with one JSON object and
nothing else, in the visible reply, not only in a reasoning trace. Do not wrap it
in markdown.

Shape: {"creators": [{"handle": "...", "judge": 0.0, "audience": 0.0,
"audience_confidence": 0.0, "reason": "...", "risky": false}]}

Rubric:
- judge is 0 to 1: score content fit to the brand using category match, tone
  match, and brand safety (no risky content). 1 means the creator clearly covers
  the brand's category in a fitting tone with no risky content. 0 means unrelated
  or clearly unsuitable. A creator not mentioning the brand is normal and must
  NOT lower the judge score; brand mentions are not required. The brand's own
  account, other shops, and competing brands are low fits however closely their
  products match.
- audience is 0 to 1: does the creator's content, captions, and bio suggest an
  audience that matches target_audience? Use text/content/bio ONLY. Never infer
  age or gender from faces, appearance, or photos.
- audience_confidence is 0 to 1: how much the available text supports the
  audience estimate.
- risky is true when the creator's content shows gambling, casino, or similar
  unsafe content.
- reason is one short sentence. Only say a creator posts about the brand when a
  brand name appears in their text; never claim that without that evidence. This
  rule only governs the reason wording, not the judge score. Copy every handle
  from the input unchanged. Include each creator you judge exactly once.
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


def judge_messages(
    brief: BrandProfile,
    kols: list[KolProfile],
) -> list[dict[str, str]]:
    cards = "\n\n".join(_kol_card(kol) for kol in kols)
    user = (
        "Brand:\n"
        f"{_brand_summary(brief)}\n\n"
        "Creators:\n"
        f"{cards}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _brand_summary(brand: BrandProfile) -> str:
    audience = brand.target_audience
    return "\n".join(
        [
            f"category: {brand.category}",
            f"brand_names: {', '.join(brand.brand_names)}",
            f"key_products: {', '.join(brand.key_products)}",
            f"price_tier: {brand.price_tier}",
            f"tone: {brand.tone}",
            (
                f"audience: {audience.gender}, ages {audience.age_range}, "
                f"interests {', '.join(audience.interests)}"
            ),
            f"topic_keywords_th: {', '.join(brand.topic_keywords_th)}",
            f"topic_keywords_en: {', '.join(brand.topic_keywords_en)}",
            f"competitors: {', '.join(brand.competitors)}",
            f"tier: {brand.tier_label} (target {brand.target_followers} followers)",
            f"tier_reason: {brand.tier_reason}",
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


def _kol_card(kol: KolProfile) -> str:
    captions = [
        _clip(video.caption, MAX_PROMPT_POST_CHARS)
        for video in kol.videos[:3]
        if video.caption.strip()
    ]
    lines = [
        f"handle: {kol.handle}",
        f"nickname: {kol.nickname or ''}",
        f"followers: {kol.followers}",
        f"language: {kol.language or ''}",
        f"region: {kol.region or ''}",
        f"stats_source: {kol.stats_source}",
        f"bio: {_clip(kol.bio, _BIO_CHARS)}",
        f"content: {_clip(kol.content_summary, _SUMMARY_CHARS)}",
    ]
    if captions:
        lines.append("recent_captions: " + " | ".join(captions))
    return "\n".join(lines)


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
