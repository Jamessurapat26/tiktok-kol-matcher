"""Embedding protocol, offline mock, and on-disk vector cache.

The mock embedder is a deterministic hashed bag-of-words. It uses the same
query and document prefixes as embeddinggemma, L2-normalizes the vectors, and
scores them with cosine similarity. It does not download a model.

The live embedder lives in ``embeddings_live`` and is not imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Protocol

from ..models import BrandProfile, KolProfile

# embeddinggemma-2 retrieval prompts. SearchQuery is the query prompt.
# Document is ``title: none | text: ``. The manual document form uses the
# creator's nickname or handle in the title slot.
QUERY_PROMPT_NAME = "SearchQuery"
DOCUMENT_PROMPT_NAME = "Document"
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "
ALLOWED_EMBED_DIMS = (128, 256, 512, 768)
DEFAULT_EMBED_DIM = 768
MAX_EMBED_TOKENS = 8192

_TOKEN_RE = re.compile(r"[\w\u0E00-\u0E7F]+")
_HASHTAG_RE = re.compile(r"#[\w\u0E00-\u0E7F]+")
_MOCK_DIM = 256


class EmbedError(RuntimeError):
    """The embedding backend failed or is unavailable."""


class EmbedConfigError(EmbedError):
    """Live embedder settings are invalid."""


class Embedder(Protocol):
    def similarity(
        self,
        query: str,
        documents: list[str],
        *,
        titles: list[str] | None = None,
    ) -> list[float]:
        """Return one cosine similarity per document, in input order.

        ``query`` uses the SearchQuery prompt. Each document uses the Document
        prompt, or a manual ``title: ... | text: `` prompt when that is how the
        backend formats documents. ``titles`` aligns with ``documents`` and is
        the nickname or handle for that manual form. Scores are rounded to 6
        decimals.
        """
        ...


def l2_normalize(vector: list[float]) -> list[float]:
    """Return a copy of ``vector`` scaled to unit length.

    A zero vector is returned unchanged so cosine similarity can treat it as
    no overlap.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two vectors.

    Equal dimensions are required. A zero vector on either side scores 0.
    Values just outside [-1, 1] from floating-point error are clipped.
    """
    if not left or len(left) != len(right):
        raise EmbedError(
            f"embedding dimensions do not match ({len(left)} and {len(right)})"
        )
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_sq += left_value * left_value
        right_sq += right_value * right_value
    if left_sq == 0.0 or right_sq == 0.0:
        return 0.0
    score = dot / math.sqrt(left_sq * right_sq)
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return score


def hashed_bow(text: str, dim: int = _MOCK_DIM) -> list[float]:
    """L2-normalized bag-of-words vector using a stable hash, not ``hash()``.

    Each token increments one bucket. The bucket is the first 4 bytes of
    SHA-256, so the vector is the same across processes.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")
    vector = [0.0] * dim
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).casefold()
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        vector[index] += 1.0
    return l2_normalize(vector)


def clamp_unit(value: float) -> float:
    """Clamp a cosine score into the relevance range 0..1."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


class MockEmbedder:
    """MOCK: deterministic hashed bag-of-words vectors and cosine similarity.

    Offline stand-in for embeddinggemma. Same query and document prefixes,
    L2-normalized vectors, and cosine similarity as the live embedder.
    """

    def __init__(self, dim: int = _MOCK_DIM) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim

    def similarity(
        self,
        query: str,
        documents: list[str],
        *,
        titles: list[str] | None = None,
    ) -> list[float]:
        if not documents:
            return []
        if titles is not None and len(titles) != len(documents):
            raise ValueError("titles must have one entry per document")
        # Offline vectors use the Document prompt (``title: none``). ``titles``
        # is accepted so the same call works for the live manual prompt.
        query_vector = hashed_bow(QUERY_PREFIX + query, self.dim)
        scores: list[float] = []
        for document in documents:
            document_vector = hashed_bow(DOCUMENT_PREFIX + document, self.dim)
            scores.append(round(cosine_similarity(query_vector, document_vector), 6))
        return scores


def brand_query_text(brand: BrandProfile) -> str:
    """Build the brand retrieval query.

    Uses category, products, audience, keywords, and hashtags. Tone and
    content style are left out so the query stays on what the creator covers.
    """
    audience = brand.target_audience
    if audience.age_min is None and audience.age_max is None:
        ages = "unspecified"
    else:
        low = "" if audience.age_min is None else str(audience.age_min)
        high = "" if audience.age_max is None else str(audience.age_max)
        ages = f"{low}-{high}"
    products = ", ".join(item.strip() for item in brand.products if item.strip())
    interests = ", ".join(item.strip() for item in audience.interests if item.strip())
    keywords = ", ".join(
        item.strip()
        for item in [*brand.keywords_th, *brand.keywords_en]
        if item.strip()
    )
    hashtags = " ".join(item.strip() for item in brand.hashtags if item.strip())
    lines = [
        brand.category.strip(),
        f"products: {products}" if products else "",
        (
            f"audience: {audience.gender}, ages {ages}, "
            f"{audience.location.strip()}, interests {interests}"
        ),
        f"keywords: {keywords}" if keywords else "",
        f"hashtags: {hashtags}" if hashtags else "",
    ]
    return "\n".join(line for line in lines if line.strip())


def kol_document_text(kol: KolProfile) -> str:
    """Build one creator document.

    Uses nickname, bio, and ``content_summary`` (recent captions and the
    caption summary). Hashtags found in the bio or summary are repeated on
    their own line so they stay visible to the embedder.
    """
    nickname = (kol.nickname or "").strip()
    bio = (kol.bio or "").strip()
    summary = (kol.content_summary or "").strip()
    seen: set[str] = set()
    tags: list[str] = []
    for tag in _HASHTAG_RE.findall(f"{bio}\n{summary}"):
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    lines: list[str] = []
    if nickname:
        lines.append(nickname)
    if bio:
        lines.append(bio)
    if summary:
        lines.append(summary)
    if tags:
        lines.append("hashtags: " + " ".join(tags))
    return "\n".join(lines)


def kol_title(kol: KolProfile) -> str:
    """Title for a manual document prompt: nickname, otherwise handle."""
    nickname = (kol.nickname or "").strip()
    if nickname:
        return nickname
    return kol.handle.strip() or "none"


def document_prefix(title: str | None = None) -> str:
    """Manual document prompt. An empty title uses ``title: none | text: ``."""
    cleaned = " ".join((title or "").split())
    if not cleaned or cleaned.casefold() == "none":
        return DOCUMENT_PREFIX
    return f"title: {cleaned} | text: "


def embedding_cache_key(model: str, prefix: str, text: str, dim: int) -> str:
    """SHA-256 of the model name, prompt prefix, raw text, and vector size.

    The payload is a JSON array so a delimiter inside one field cannot be
    confused with the next field. ``dim`` is part of the key so a 768-d vector
    is not reused as a 256-d vector.
    """
    payload = json.dumps(
        [model, prefix, text, int(dim)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_embedding_cache_dir() -> Path:
    """Embedding cache: ``$KOL_CACHE_DIR/embeddings`` or ``<repo>/data/cache/embeddings``."""
    override = os.environ.get("KOL_CACHE_DIR")
    if override:
        return Path(override) / "embeddings"
    # This file is backend/app/integrations/embeddings.py.
    return Path(__file__).resolve().parents[3] / "data" / "cache" / "embeddings"


class EmbeddingCache:
    """JSON file cache of normalized embedding vectors.

    One file per key. Writes go to a temporary file in the same directory and
    are renamed into place. A missing, unreadable, or mismatched file is a miss.
    Vectors do not expire: the key already includes the model, the prefix,
    and the dimension.
    """

    def __init__(self, cache_dir: Path, *, clock=time.time) -> None:
        self.cache_dir = cache_dir
        self._clock = clock

    def path_for(self, model: str, prefix: str, text: str, dim: int) -> Path:
        return self.cache_dir / f"{embedding_cache_key(model, prefix, text, dim)}.json"

    def get(self, model: str, prefix: str, text: str, dim: int) -> list[float] | None:
        path = self.path_for(model, prefix, text, dim)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("model") != model
            or payload.get("prefix") != prefix
            or payload.get("text") != text
            or payload.get("dim") != dim
        ):
            return None
        vector = payload.get("vector")
        if not isinstance(vector, list) or not vector:
            return None
        cleaned: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number):
                return None
            cleaned.append(number)
        return cleaned

    def put(
        self,
        model: str,
        prefix: str,
        text: str,
        dim: int,
        vector: list[float],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(model, prefix, text, dim)
        payload = {
            "model": model,
            "prefix": prefix,
            "text": text,
            "dim": dim,
            "vector": vector,
            "fetched_at": self._clock(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
