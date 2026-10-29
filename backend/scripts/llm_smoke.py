#!/usr/bin/env python3
"""Run one live brand-analysis call and print a short profile.

Requires ``LLM_API_KEY``, ``NVIDIA_API_KEY``, or ``OPENAI_API_KEY``. Exits
with code 2 and a message when none of those is set. The key is never printed.

    cd backend && uv run python scripts/llm_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integrations.llm_live import (
    LlmError,
    LiveBrandAnalyzer,
    llm_api_key_from_env,
    redact_secret,
)

# A generic page, long enough to be a real extraction and short enough for one call.
_SAMPLE_WEBSITE = """
A small shop in Bangkok sells blue-light glasses for people who game and study
on screens. The tone is friendly and practical. Products include gaming glasses
and clip-on lenses. Customers are mostly young adults in Thailand.
""".strip()


def main(_argv: list[str] | None = None) -> int:
    key = llm_api_key_from_env()
    if not key:
        print(
            "error: set LLM_API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY",
            file=sys.stderr,
        )
        return 2

    analyzer = LiveBrandAnalyzer()
    try:
        profile = analyzer.analyze_brand(_SAMPLE_WEBSITE, None)
    except LlmError as exc:
        print(f"error: {redact_secret(str(exc), key)}", file=sys.stderr)
        return 1
    finally:
        analyzer.close()

    def clean(text: str) -> str:
        return redact_secret(text, key)

    print(f"category: {clean(profile.category)}")
    print(f"topic_keywords_th: {clean(', '.join(profile.topic_keywords_th))}")
    print(f"topic_keywords_en: {clean(', '.join(profile.topic_keywords_en))}")
    print(f"tier: {clean(profile.tier_label)} ({profile.target_followers} followers)")
    print(f"tier_reason: {clean(profile.tier_reason)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
