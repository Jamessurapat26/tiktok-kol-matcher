#!/usr/bin/env python3
"""Build the cached candidate pool for the example-brand evaluation.

Runs the fully mocked graph (no live flags, no network) and writes the brief,
aliases, candidates, and a fixed ``now`` timestamp to
``eval/pools/example-brand.json``. The committed JSON is what the evaluator
reads; rerun this only when the fixtures or the mock pipeline change.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.brand_match import brand_aliases, normalized_aliases  # noqa: E402
from app.graph import build_graph, initial_state  # noqa: E402
from app.models import MatchRequest  # noqa: E402

BRAND_SLUG = "example-brand"
WEBSITE_URL = "https://example-brand.co.th"
FACEBOOK_URL = "https://facebook.com/examplebrand"
POOL_PATH = _EVAL_DIR / "pools" / f"{BRAND_SLUG}.json"


def build_pool() -> dict:
    """Run the mock graph and return the cacheable pool document."""
    request = MatchRequest(
        website_url=WEBSITE_URL,
        facebook_url=FACEBOOK_URL,
        top_k=10,
    )
    final = build_graph().invoke(initial_state(request))
    if final["stage"] != "done":
        raise RuntimeError(
            f"mock graph did not finish: stage={final['stage']} "
            f"errors={final['errors']}"
        )

    brief = final["brand_profile"]
    if brief is None:
        raise RuntimeError("mock graph produced no brand profile")

    now = datetime.now(UTC).replace(microsecond=0)
    return {
        "brand": BRAND_SLUG,
        "brief": brief.model_dump(mode="json"),
        "brand_aliases": brand_aliases(brief, WEBSITE_URL, FACEBOOK_URL),
        "competitor_aliases": normalized_aliases(brief.competitors),
        "now": now.isoformat().replace("+00:00", "Z"),
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in final["candidates"]
        ],
    }


def main() -> int:
    pool = build_pool()
    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {POOL_PATH.relative_to(_REPO_ROOT)} "
        f"({len(pool['candidates'])} candidates, now={pool['now']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
