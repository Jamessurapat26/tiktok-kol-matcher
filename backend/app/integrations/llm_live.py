"""Live brand analyzer backed by an OpenAI-compatible chat completions API.

The provider is selected with settings, not code. ``LLM_BASE_URL``,
``LLM_MODEL``, and the API key are enough to point the same client at the
default NVIDIA endpoint or at any other ``/v1`` chat API.

The default model is a reasoning model. It can return ``reasoning_content``
and a null ``content`` when the completion budget is used up by the trace, so
``max_tokens`` defaults to 4096, the timeout defaults to 120 seconds, and
empty content is retried once. ``LLM_DISABLE_REASONING`` controls thinking:
``extra_body`` sends ``chat_template_kwargs.enable_thinking=false``, ``system``
prepends a ``/no_think`` message, ``both`` does both, and ``off`` sends a plain
request. When unset it is ``extra_body`` for the default NVIDIA endpoint and
model (measured about 10x faster) and ``off`` for any other provider.

The API key is never logged. It is removed from every error string.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import quote

from openai import OpenAI
from pydantic import Field, ValidationError

from ..brand_tidy import tidy_brand_profile
from ..models import (
    BrandProfile,
    DomainModel,
    FacebookPage,
    KolProfile,
    ScoreBreakdown,
    normalize_handle,
)
from ..prompts import (
    JSON_RETRY_USER,
    MAX_QUERIES,
    brand_messages,
    query_messages,
    rerank_messages,
)
from .llm import ChatClient, heuristic_score

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RERANK_BATCH_SIZE = 5
DEFAULT_NO_THINK_MESSAGE = "/no_think"

_KEY_ENV_VARS = ("LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")
ReasoningMode = Literal["off", "system", "extra_body", "both"]

_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class LlmError(RuntimeError):
    """The chat API failed or returned output we cannot use."""


class LlmConfigError(LlmError):
    """Live LLM settings are missing or invalid."""


class LlmOutputError(LlmError):
    """The model reply was empty or not valid structured JSON after retry."""


@dataclass(frozen=True)
class LlmSettings:
    """Chat-completions settings. ``api_key`` is omitted from ``repr``."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = field(default="", repr=False)
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    reasoning: ReasoningMode = "off"
    no_think_message: str = DEFAULT_NO_THINK_MESSAGE
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE

    @classmethod
    def from_env(cls, *, api_key: str | None = None) -> LlmSettings:
        base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise LlmConfigError("LLM_BASE_URL must start with http:// or https://")

        model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
        timeout = _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT, minimum=0.0, positive=True)
        max_tokens = _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS, minimum=1)
        temperature = _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE, minimum=0.0)
        batch_size = _env_int(
            "LLM_RERANK_BATCH_SIZE",
            DEFAULT_RERANK_BATCH_SIZE,
            minimum=1,
        )
        reasoning = _reasoning_mode(os.environ.get("LLM_DISABLE_REASONING"))
        if reasoning is None:
            # Unset: turn thinking off only for the default endpoint and model,
            # where it is known to work. Other providers get a plain request.
            is_default = base_url == DEFAULT_BASE_URL and model == DEFAULT_MODEL
            reasoning = "extra_body" if is_default else "off"
        no_think = os.environ.get("LLM_NO_THINK_MESSAGE", "").strip() or DEFAULT_NO_THINK_MESSAGE
        return cls(
            base_url=base_url,
            model=model,
            api_key=llm_api_key_from_env(api_key),
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning,
            no_think_message=no_think,
            rerank_batch_size=batch_size,
        )


def llm_api_key_from_env(explicit: str | None = None) -> str:
    """Resolve the bearer token.

    An explicit value is used as given (after stripping), including when it is
    empty. Otherwise the first non-empty value among ``LLM_API_KEY``,
    ``NVIDIA_API_KEY``, and ``OPENAI_API_KEY`` wins.
    """
    if explicit is not None:
        return explicit.strip()
    for name in _KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def redact_secret(text: str, secret: str) -> str:
    """Replace ``secret`` and its URL-encoded form with ``[redacted]``."""
    if not text or not secret:
        return text
    redacted = text.replace(secret, "[redacted]")
    encoded = quote(secret, safe="")
    if encoded and encoded != secret:
        redacted = redacted.replace(encoded, "[redacted]")
    return redacted


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object, ignoring fences and surrounding prose."""
    raw = text.strip()
    fenced = _FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        if start < 0:
            raise LlmOutputError("response was not a JSON object") from None
        try:
            value, _end = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError as exc:
            raise LlmOutputError(
                f"response was not a JSON object: {exc.msg}"
            ) from None
    if not isinstance(value, dict):
        raise LlmOutputError("response JSON must be an object")
    return value


class QueryBatch(DomainModel):
    queries: list[str] = Field(min_length=1)


class RelevanceScore(DomainModel):
    handle: str = Field(min_length=1)
    relevance: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=400)


class RelevanceBatch(DomainModel):
    scores: list[RelevanceScore] = Field(min_length=1)


class OpenAIChatClient:
    """Chat completions via the official OpenAI SDK.

    ``max_retries`` is 0 so the only automatic retry is the empty-content
    retry below. Timeouts and HTTP errors are not retried: a reasoning call
    can already sit for most of the timeout.
    """

    def __init__(
        self,
        settings: LlmSettings,
        *,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        if not settings.api_key:
            raise LlmConfigError(_MISSING_KEY)
        self._settings = settings
        self._api_key = settings.api_key
        if client is not None:
            self._client = client
            return
        kwargs: dict[str, Any] = {
            "api_key": settings.api_key,
            "base_url": settings.base_url,
            "timeout": settings.timeout,
            "max_retries": 0,
        }
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = OpenAI(**kwargs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        prepared = _with_reasoning_message(messages, self._settings)
        extra = _reasoning_extra_body(self._settings)
        saw_reasoning = False
        for _attempt in range(2):
            message = self._create(prepared, extra)
            text = _message_text(message)
            if text is not None:
                return text
            if _has_reasoning(message):
                saw_reasoning = True
        detail = ""
        if saw_reasoning:
            detail = " The response included reasoning_content but content was empty."
        raise LlmOutputError(
            "LLM returned empty content after one retry."
            + detail
            + " A reasoning model can use the completion budget before it writes"
            " an answer; increase LLM_MAX_TOKENS or set LLM_DISABLE_REASONING."
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def _create(self, messages: list[dict[str, str]], extra: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
        }
        if extra:
            kwargs["extra_body"] = extra
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Drop the cause. OpenAI exceptions can keep the request, including
            # the Authorization header, on the traceback context.
            raise self._request_error(exc) from None
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return None
        return choices[0].message

    def _request_error(self, exc: BaseException) -> LlmError:
        message = redact_secret(f"{type(exc).__name__}: {exc}", self._api_key)
        message = " ".join(message.split())
        if len(message) > 500:
            message = message[:497] + "..."
        return LlmError(f"LLM request failed: {message}")


class LiveBrandAnalyzer:
    """Brand analysis, query generation, and relevance scoring via chat completions.

    Engagement, audience fit, and tier fit stay on ``heuristic_score``. Only
    relevance and the short reason come from the model, and the graph still
    applies the request weights to the full breakdown.
    """

    def __init__(
        self,
        settings: LlmSettings | None = None,
        *,
        api_key: str | None = None,
        chat: ChatClient | None = None,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        if settings is None:
            settings = LlmSettings.from_env(api_key=api_key)
        elif api_key is not None:
            settings = replace(settings, api_key=api_key.strip())
        if chat is None and not settings.api_key:
            raise LlmConfigError(_MISSING_KEY)
        self._settings = settings
        self._chat = chat or OpenAIChatClient(
            settings,
            client=client,
            http_client=http_client,
        )

    def close(self) -> None:
        close = getattr(self._chat, "close", None)
        if close is not None:
            close()

    def analyze_brand(
        self,
        website_content: str | None,
        facebook_page: FacebookPage | None,
    ) -> BrandProfile:
        profile = self._structured(
            brand_messages(website_content, facebook_page),
            BrandProfile,
            coerce=_coerce_brand_lists,
        )
        return tidy_brand_profile(profile)

    def generate_queries(
        self,
        brand: BrandProfile,
        previous_queries: list[str],
        round: int,
    ) -> list[str]:
        previous = {query.strip().casefold() for query in previous_queries}

        def accept(batch: QueryBatch) -> None:
            if not _new_queries(batch.queries, set(previous)):
                raise ValueError(
                    "no new queries; every query repeats an earlier round"
                )

        batch = self._structured(
            query_messages(brand, previous_queries, round),
            QueryBatch,
            accept,
        )
        fresh = _new_queries(batch.queries, set(previous))
        if not fresh:
            raise LlmOutputError(
                f"LLM returned no new search queries for round {round}"
            )
        return fresh[:MAX_QUERIES]

    def rerank(
        self,
        brand: BrandProfile,
        kols: list[KolProfile],
        *,
        similarities: list[float] | None = None,
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        if similarities is not None and len(similarities) != len(kols):
            raise ValueError("similarities must have one score per creator")
        if not kols:
            return []
        scored: list[tuple[KolProfile, ScoreBreakdown, str]] = []
        size = self._settings.rerank_batch_size
        for start in range(0, len(kols), size):
            batch = kols[start : start + size]
            batch_scores = None if similarities is None else similarities[start : start + size]
            scored.extend(self._rerank_batch(brand, batch, batch_scores))
        return scored

    def _rerank_batch(
        self,
        brand: BrandProfile,
        kols: list[KolProfile],
        similarities: list[float] | None = None,
    ) -> list[tuple[KolProfile, ScoreBreakdown, str]]:
        def accept(batch: RelevanceBatch) -> None:
            seen = {normalize_handle(item.handle) for item in batch.scores}
            missing = [kol.handle for kol in kols if normalize_handle(kol.handle) not in seen]
            if missing:
                raise ValueError(f"missing scores for: {', '.join(missing)}")

        batch = self._structured(
            rerank_messages(brand, kols, similarities),
            RelevanceBatch,
            accept,
        )
        by_score = {normalize_handle(item.handle): item for item in batch.scores}
        results: list[tuple[KolProfile, ScoreBreakdown, str]] = []
        for kol in kols:
            item = by_score[normalize_handle(kol.handle)]
            heuristic, _heuristic_reason = heuristic_score(brand, kol)
            scores = heuristic.model_copy(
                update={"relevance": round(float(item.relevance), 4)}
            )
            results.append((kol, scores, item.reason))
        return results

    def _structured(
        self,
        messages: list[dict[str, str]],
        model: type[Any],
        accept: Callable[[Any], None] | None = None,
        coerce: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> Any:
        conversation = list(messages)
        last_error = "invalid output"
        for attempt in range(2):
            # Empty content raises here and is not treated as a JSON retry.
            text = self._chat.complete(conversation)
            try:
                data = parse_json_object(text)
                if coerce is not None:
                    data = coerce(data)
                parsed = model.model_validate(data)
                if accept is not None:
                    accept(parsed)
                return parsed
            except (LlmOutputError, ValidationError, ValueError) as exc:
                last_error = _clip_error(self._redact(str(exc)))
                if attempt >= 1:
                    break
                conversation = [
                    *messages,
                    {"role": "assistant", "content": self._redact(text)[:4000]},
                    {
                        "role": "user",
                        "content": JSON_RETRY_USER.format(error=last_error),
                    },
                ]
        raise LlmOutputError(
            f"LLM structured output failed after one retry: {last_error}"
        )

    def _redact(self, text: str) -> str:
        return redact_secret(text, self._settings.api_key)


_MISSING_KEY = (
    "No LLM API key is set. Export LLM_API_KEY, NVIDIA_API_KEY, or "
    "OPENAI_API_KEY before using the live LLM."
)


def _with_reasoning_message(
    messages: list[dict[str, str]],
    settings: LlmSettings,
) -> list[dict[str, str]]:
    if settings.reasoning not in {"system", "both"}:
        return list(messages)
    marker = settings.no_think_message
    if (
        messages
        and messages[0].get("role") == "system"
        and messages[0].get("content") == marker
    ):
        return list(messages)
    return [{"role": "system", "content": marker}, *messages]


def _reasoning_extra_body(settings: LlmSettings) -> dict[str, Any] | None:
    """Provider-specific body. The default NVIDIA model reads ``enable_thinking``."""
    if settings.reasoning not in {"extra_body", "both"}:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _message_text(message: Any) -> str | None:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            else:
                piece = getattr(part, "text", None)
                if isinstance(piece, str):
                    parts.append(piece)
        text = "".join(parts)
    else:
        text = str(content)
    if not text.strip():
        return None
    return text.strip()


def _has_reasoning(message: Any) -> bool:
    if message is None:
        return False
    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning:
        extra = getattr(message, "__pydantic_extra__", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content")
    return bool(reasoning and str(reasoning).strip())


_BRAND_LIST_FIELDS = (
    "products",
    "keywords_th",
    "keywords_en",
    "hashtags",
    "preferred_tiers",
)


def _as_list(value: Any) -> Any:
    """Turn a delimited string into a list. Anything else passes through.

    Models sometimes return ``"#a #b"`` or ``"a, b"`` where the schema wants a
    list. Splitting on commas, semicolons, and newlines keeps multi-word terms;
    a string with none of those (hashtags) splits on whitespace.
    """
    if not isinstance(value, str):
        return value
    parts = re.split(r"[,;\n]", value)
    if len(parts) == 1:
        parts = value.split()
    return [part.strip() for part in parts if part.strip()]


def _coerce_brand_lists(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for name in _BRAND_LIST_FIELDS:
        if name in out:
            out[name] = _as_list(out[name])
    audience = out.get("target_audience")
    if isinstance(audience, dict) and "interests" in audience:
        out["target_audience"] = {
            **audience,
            "interests": _as_list(audience["interests"]),
        }
    return out


def _new_queries(queries: list[str], previous: set[str]) -> list[str]:
    fresh: list[str] = []
    seen = set(previous)
    for query in queries:
        text = query.strip()
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append(text)
    return fresh


def _clip_error(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) > 1500:
        return compact[:1497] + "..."
    return compact


def _env_float(name: str, default: float, *, minimum: float, positive: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise LlmConfigError(f"{name} must be a number") from None
    if not math.isfinite(value) or value < minimum or (positive and value <= 0):
        bound = "> 0" if positive else f">= {minimum}"
        raise LlmConfigError(f"{name} must be {bound}")
    return value


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw, 10)
    except ValueError:
        raise LlmConfigError(f"{name} must be an integer") from None
    if value < minimum:
        raise LlmConfigError(f"{name} must be >= {minimum}")
    return value


def _reasoning_mode(raw: str | None) -> ReasoningMode | None:
    value = (raw or "").strip().casefold()
    if not value:
        return None
    if value in {"0", "false", "off", "no", "none"}:
        return "off"
    if value in {"1", "true", "on", "yes", "both"}:
        return "both"
    if value in {"system", "message", "no_think"}:
        return "system"
    if value in {"extra_body", "extra-body", "chat_template_kwargs"}:
        return "extra_body"
    raise LlmConfigError(
        "LLM_DISABLE_REASONING must be off, system, extra_body, or both"
    )
