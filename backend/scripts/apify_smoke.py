#!/usr/bin/env python3
"""Run one live Apify TikTok search and print mapped profiles.

Requires ``APIFY_TOKEN``. Sends at most 5 results for a single query (default
``blue light glasses``) and prints ``handle``, ``url``, and ``followers``.
The token is never printed.

    cd backend && uv run python scripts/apify_smoke.py
    uv run python scripts/apify_smoke.py "blue light glasses"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integrations.apify import ApifyConfigError, ApifyError
from app.integrations.kol_search_live import ApifyKolSearch


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    query = args[0] if args else "blue light glasses"
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        print("error: APIFY_TOKEN is not set", file=sys.stderr)
        return 2

    try:
        rows = ApifyKolSearch(token=token, max_results=5).search([query], round=1)
    except (ApifyConfigError, ApifyError) as exc:
        print(f"error: {str(exc).replace(token, '[redacted]')}", file=sys.stderr)
        return 1

    printed = 0
    for row in rows:
        if row.get("errorCode") or row.get("error"):
            detail = str(row.get("error") or "").replace(token, "[redacted]")
            print(f"skipped: {row.get('errorCode')}: {detail}", file=sys.stderr)
            continue
        print(f"{row.get('handle')}\t{row.get('url')}\t{row.get('followers')}")
        printed += 1

    if printed == 0:
        print("error: no profiles mapped", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
