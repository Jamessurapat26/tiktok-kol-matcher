#!/usr/bin/env python3
"""Embed one query and two documents with the local embeddinggemma model.

Requires the embed extra (``uv sync --extra embed``). The default model is
``google/embeddinggemma-2`` on CPU. It is Apache 2.0. ``HF_TOKEN`` is
optional. Exits 1 when the extra is missing or the model
fails to load.

    cd backend && uv run python scripts/embed_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integrations.embeddings import EmbedError  # noqa: E402
from app.integrations.embeddings_live import SentenceTransformerEmbedder  # noqa: E402

_QUERY = "blue light glasses for people who game and study"
_DOCUMENTS = (
    "A creator reviews gaming glasses and desk setups.",
    "A creator films noodle recipes and street food.",
)


def main(_argv: list[str] | None = None) -> int:
    embedder = SentenceTransformerEmbedder.from_env()
    try:
        scores = embedder.similarity(_QUERY, list(_DOCUMENTS))
    except EmbedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"model: {embedder.model_name}")
    print(f"device: {embedder.device}")
    print(f"dim: {embedder.dim}")
    print(f"dtype: {embedder.dtype}")
    for document, score in zip(_DOCUMENTS, scores, strict=True):
        print(f"{score:.4f}  {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
