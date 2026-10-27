"""Local embeddinggemma embedder backed by sentence-transformers.

The packages are an optional extra (``uv sync --extra embed``) and are imported
on the first embed call. The loaded model is kept for the process and reused.

``EMBED_MODEL`` defaults to ``google/embeddinggemma-2``. ``EMBED_DEVICE``
defaults to ``cpu``. ``EMBED_DIM`` defaults to 768 and is passed as
``truncate_dim`` (also 512, 256, or 128). Dtype is float32 unless
``EMBED_DTYPE=bfloat16``. float16 is rejected.

The model is Apache 2.0. ``HF_TOKEN`` is optional. The loader
asks for the text tower only and caps the context at 8192 tokens. This module
redacts ``HF_TOKEN`` if a library error echoes it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .embeddings import (
    ALLOWED_EMBED_DIMS,
    DEFAULT_EMBED_DIM,
    DOCUMENT_PREFIX,
    DOCUMENT_PROMPT_NAME,
    MAX_EMBED_TOKENS,
    QUERY_PREFIX,
    QUERY_PROMPT_NAME,
    EmbedConfigError,
    EmbedError,
    EmbeddingCache,
    cosine_similarity,
    default_embedding_cache_dir,
    document_prefix,
    l2_normalize,
)

DEFAULT_EMBED_MODEL = "google/embeddinggemma-2"
DEFAULT_EMBED_DEVICE = "cpu"
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_EMBED_DTYPE = "float32"

_MISSING_PACKAGE = (
    "sentence-transformers is not installed. "
    "From the backend directory run `uv sync --extra embed`. "
    "The default install does not include it."
)

_TEXT_ONLY_CONFIG = {"vision_config": None, "audio_config": None}

_MODELS: dict[tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def clear_sentence_transformer_cache() -> None:
    """Drop process-local models. The next embed loads again."""
    with _MODEL_LOCK:
        _MODELS.clear()


def _redact(text: str) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not text or not token:
        return text
    redacted = text.replace(token, "[redacted]")
    encoded = quote(token, safe="")
    if encoded and encoded != token:
        redacted = redacted.replace(encoded, "[redacted]")
    return redacted


def _clip(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw, 10)
    except ValueError:
        raise EmbedConfigError(f"{name} must be an integer") from None
    if value < 1:
        raise EmbedConfigError(f"{name} must be >= 1")
    return value


def _env_dim() -> int:
    raw = os.environ.get("EMBED_DIM", "").strip()
    if not raw:
        return DEFAULT_EMBED_DIM
    try:
        value = int(raw, 10)
    except ValueError:
        raise EmbedConfigError("EMBED_DIM must be an integer") from None
    if value not in ALLOWED_EMBED_DIMS:
        raise EmbedConfigError("EMBED_DIM must be 128, 256, 512, or 768")
    return value


def _env_dtype() -> str:
    raw = os.environ.get("EMBED_DTYPE", "").strip().lower() or DEFAULT_EMBED_DTYPE
    aliases = {
        "float32": "float32",
        "fp32": "float32",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    if raw in {"float16", "fp16", "half"}:
        raise EmbedConfigError("EMBED_DTYPE must be float32 or bfloat16, not float16")
    canonical = aliases.get(raw)
    if canonical is None:
        raise EmbedConfigError("EMBED_DTYPE must be float32 or bfloat16")
    return canonical


def _prompt_text(model: Any, name: str) -> str | None:
    """Return the model's built-in prompt for ``name``, if it has one."""
    prompts = getattr(model, "prompts", None)
    if not isinstance(prompts, dict) or name not in prompts:
        return None
    value = prompts[name]
    if isinstance(value, str) and value.strip():
        return value
    for attr in ("prompt", "text"):
        text = getattr(value, attr, None)
        if isinstance(text, str) and text.strip():
            return text
    return None


def _rows(encoded: Any, count: int) -> list[list[float]]:
    """Turn an encode result into ``count`` float vectors."""
    if hasattr(encoded, "tolist"):
        data = encoded.tolist()
    else:
        data = list(encoded)
    if (
        count == 1
        and data
        and isinstance(data[0], (int, float))
        and not isinstance(data[0], bool)
    ):
        data = [data]
    rows: list[list[float]] = []
    for row in data:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise EmbedError("embedder returned a vector that is not a sequence")
        cleaned: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbedError("embedder returned a non-numeric vector")
            number = float(value)
            if number != number or number in {float("inf"), float("-inf")}:
                raise EmbedError("embedder returned a non-finite value")
            cleaned.append(number)
        if not cleaned:
            raise EmbedError("embedder returned an empty vector")
        rows.append(cleaned)
    if len(rows) != count:
        raise EmbedError(f"embedder returned {len(rows)} vectors for {count} texts")
    return rows


def _encode(
    model: Any,
    texts: list[str],
    *,
    prompt_name: str | None,
    dim: int,
) -> list[list[float]]:
    """Encode one batch with L2 normalization and Matryoshka truncation."""
    kwargs: dict[str, Any] = {
        "normalize_embeddings": True,
        "truncate_dim": dim,
        "show_progress_bar": False,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    try:
        encoded = model.encode(texts, **kwargs)
    except EmbedError:
        raise
    except Exception as exc:
        raise EmbedError(f"Embedding failed: {_clip(_redact(str(exc)))}") from exc
    return _rows(encoded, len(texts))


def _torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "float16":
        raise EmbedConfigError("EMBED_DTYPE must be float32 or bfloat16, not float16")
    try:
        import torch
    except ImportError as exc:
        raise EmbedError(_MISSING_PACKAGE) from exc
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if dtype is getattr(torch, "float16", object()):
        raise EmbedConfigError("EMBED_DTYPE must be float32 or bfloat16, not float16")
    return dtype


def _load_model(model_name: str, device: str, dtype_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbedError(_MISSING_PACKAGE) from exc
    dtype = _torch_dtype(dtype_name)
    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            config_kwargs=dict(_TEXT_ONLY_CONFIG),
            model_kwargs={"dtype": dtype},
        )
    except EmbedError:
        raise
    except Exception as exc:
        detail = _clip(_redact(f"{type(exc).__name__}: {exc}"))
        raise EmbedError(
            f"Could not load embedding model {model_name} on device {device}: {detail}"
        ) from exc
    try:
        model.max_seq_length = MAX_EMBED_TOKENS
    except (AttributeError, TypeError):
        pass
    return model


def _shared_model(model_name: str, device: str, dtype_name: str) -> Any:
    """Load once per process for this model, device, and dtype, then reuse it."""
    key = (model_name, device, dtype_name)
    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached
        model = _load_model(model_name, device, dtype_name)
        _MODELS[key] = model
        return model


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers.

    The model is loaded on the first ``similarity`` call and reused for the
    rest of the process. Vectors are L2-normalized, truncated to ``dim``, and
    cached as JSON.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str = DEFAULT_EMBED_DEVICE,
        *,
        dim: int = DEFAULT_EMBED_DIM,
        dtype: str = DEFAULT_EMBED_DTYPE,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_dir: Path | None = None,
        model: Any = None,
    ) -> None:
        cleaned_model = model_name.strip()
        cleaned_device = device.strip() or DEFAULT_EMBED_DEVICE
        if not cleaned_model:
            raise ValueError("model_name must not be empty")
        if dim not in ALLOWED_EMBED_DIMS:
            raise ValueError("dim must be 128, 256, 512, or 768")
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model_name = cleaned_model
        self.device = cleaned_device
        self.dim = dim
        self.dtype = dtype
        self.batch_size = batch_size
        self._cache = EmbeddingCache(
            cache_dir if cache_dir is not None else default_embedding_cache_dir()
        )
        # An injected model is used as-is and is not stored in the process cache.
        self._model = model

    @classmethod
    def from_env(cls, *, cache_dir: Path | None = None) -> SentenceTransformerEmbedder:
        """Build from the ``EMBED_*`` environment variables.

        This does not load the model and does not import sentence-transformers.
        """
        model_name = os.environ.get("EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL
        device = os.environ.get("EMBED_DEVICE", "").strip() or DEFAULT_EMBED_DEVICE
        return cls(
            model_name=model_name,
            device=device,
            dim=_env_dim(),
            dtype=_env_dtype(),
            batch_size=_env_positive_int("EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE),
            cache_dir=cache_dir,
        )

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
        model = (
            self._model
            if self._model is not None
            else _shared_model(self.model_name, self.device, self.dtype)
        )
        query_vector = self._vectors(model, "query", [query], None)[0]
        document_vectors = self._vectors(model, "document", documents, titles)
        return [
            round(cosine_similarity(query_vector, document), 6)
            for document in document_vectors
        ]

    def _vectors(
        self,
        model: Any,
        role: str,
        texts: list[str],
        titles: list[str] | None,
    ) -> list[list[float]]:
        prompt_name = QUERY_PROMPT_NAME if role == "query" else DOCUMENT_PROMPT_NAME
        named = _prompt_text(model, prompt_name)
        use_prompt = named is not None
        prefixes: list[str] = []
        for index in range(len(texts)):
            if use_prompt:
                prefixes.append(named or "")
            elif role == "query":
                prefixes.append(QUERY_PREFIX)
            else:
                title = None if titles is None else titles[index]
                prefixes.append(document_prefix(title))

        found: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, text in enumerate(texts):
            cached = self._cache.get(self.model_name, prefixes[index], text, self.dim)
            if cached is None:
                missing.append(index)
            else:
                found[index] = cached

        for start in range(0, len(missing), self.batch_size):
            batch_indexes = missing[start : start + self.batch_size]
            batch_texts = [texts[index] for index in batch_indexes]
            if use_prompt:
                rows = _encode(model, batch_texts, prompt_name=prompt_name, dim=self.dim)
            else:
                prompted = [
                    prefixes[index] + texts[index] for index in batch_indexes
                ]
                rows = _encode(model, prompted, prompt_name=None, dim=self.dim)
            for index, row in zip(batch_indexes, rows, strict=True):
                normalized = l2_normalize(row)
                self._cache.put(
                    self.model_name,
                    prefixes[index],
                    texts[index],
                    self.dim,
                    normalized,
                )
                found[index] = normalized

        if any(row is None for row in found):
            raise EmbedError("embedding cache returned an incomplete batch")
        return [row for row in found if row is not None]
