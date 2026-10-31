"""Live brand analyzer backed by an OpenAI-compatible chat completions API.

The provider is selected with settings, not code. ``LLM_BASE_URL``,
``LLM_MODEL``, and the API key are enough to point the same client at the
default NVIDIA endpoint or at any other ``/v1`` chat API.

The default model is a reasoning model. It can return ``reasoning_content``
and a null ``content`` when the completion budget is used up by the trace, so
``max_tokens`` defaults to 16000 and empty content is retried once. Each request
attempt times out after 60 seconds. Transport failures (timeouts, connection
errors, 408, 429, and 5xx) are retried twice with doubling backoff, so the
worst case for one call with the defaults is 3 x 60 s + 2 s + 4 s = 186 s.
``LLM_DISABLE_REASONING`` controls thinking: ``extra_body`` sends
``chat_template_kwargs.enable_thinking=false``, ``system`` prepends a
``/no_think`` message, ``both`` does both, and ``off`` sends a plain request.
When unset it is ``extra_body`` for the default NVIDIA endpoint and model
(measured about 10x faster) and ``off`` for any other provider.

The API key is never logged. It is removed from every error string.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import quote

import openai
from openai import OpenAI
from pydantic import Field, ValidationError

from ..brand_match import MIN_ALIAS_CHARS, normalize_for_match
from ..brand_tidy import tidy_brand_profile
from ..models import (
    BrandProfile,
    ContentAnalysis,
    DomainModel,
    FacebookPage,
    Judgement,
    KolProfile,
    known_content_formats,
    normalize_handle,
)
from ..prompts import (
    JSON_RETRY_USER,
    MAX_QUERIES,
    brand_messages,
    judge_messages,
    judge_videos,
    query_messages,
)
from .llm import ChatClient

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NO_THINK_MESSAGE = "/no_think"

_MAX_RETRY_AFTER = 30.0
_RETRYABLE_STATUS = frozenset({408, 429})

logger = logging.getLogger(__name__)

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
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    reasoning: ReasoningMode = "off"
    no_think_message: str = DEFAULT_NO_THINK_MESSAGE

    @classmethod
    def from_env(cls, *, api_key: str | None = None) -> LlmSettings:
        base_url = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise LlmConfigError("LLM_BASE_URL must start with http:// or https://")

        model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
        timeout = _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT, minimum=0.0, positive=True)
        max_retries = _env_int("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0)
        retry_backoff = _env_float(
            "LLM_RETRY_BACKOFF",
            DEFAULT_RETRY_BACKOFF,
            minimum=0.0,
        )
        max_tokens = _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS, minimum=1)
        temperature = _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE, minimum=0.0)
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
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning,
            no_think_message=no_think,
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


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_STATUS or (
            isinstance(status, int) and status >= 500
        )
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    if not isinstance(exc, openai.APIStatusError):
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except (AttributeError, TypeError):
        return None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


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


class JudgementItem(DomainModel):
    handle: str = Field(min_length=1)
    judge: float
    audience: float
    audience_confidence: float
    reason: str = ""
    risky: bool = False
    account_type: Literal["kol", "seller", "media"] = "kol"
    # Display only, so a malformed value must not fail the whole batch of scores.
    content_formats: Any = None
    content_style: Any = None
    style_fit: Any = None
    style_reason: Any = None
    style_evidence: Any = None


class JudgementBatch(DomainModel):
    creators: list[JudgementItem] = Field(min_length=1)


class OpenAIChatClient:
    """Chat completions via the official OpenAI SDK.

    The SDK's own ``max_retries`` is 0. This class retries transient transport
    failures itself: timeouts, connection errors, 408, 429, and 5xx, up to
    ``settings.max_retries`` times with doubling backoff. Other HTTP errors and
    non-OpenAI exceptions are not retried. Empty content has its own retry in
    ``complete``.
    """

    def __init__(
        self,
        settings: LlmSettings,
        *,
        client: Any = None,
        http_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.api_key:
            raise LlmConfigError(_MISSING_KEY)
        self._settings = settings
        self._api_key = settings.api_key
        self._sleep = sleep
        self.retry_count = 0
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
        attempts = self._settings.max_retries + 1
        retries = 0
        for attempt in range(attempts):
            try:
                completion = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not _is_transient(exc) or attempt >= attempts - 1:
                    # Drop the cause. OpenAI exceptions can keep the request,
                    # including the Authorization header, on the traceback
                    # context.
                    raise self._request_error(exc, retries=retries) from None
                retries += 1
                self.retry_count += 1
                delay = self._retry_delay(exc, retries)
                logger.warning(
                    "LLM request failed (%s); retry %d of %d in %.1fs",
                    self._short_error(exc),
                    retries,
                    self._settings.max_retries,
                    delay,
                )
                self._sleep(delay)
                continue
            choices = getattr(completion, "choices", None) or []
            if not choices:
                return None
            return choices[0].message
        raise AssertionError("unreachable")

    def _retry_delay(self, exc: BaseException, retry: int) -> float:
        delay = self._settings.retry_backoff * 2 ** (retry - 1)
        retry_after = _retry_after_seconds(exc)
        if retry_after is None:
            return delay
        return min(max(delay, retry_after), _MAX_RETRY_AFTER)

    def _short_error(self, exc: BaseException) -> str:
        text = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if status is not None:
            text += f" status {status}"
        return redact_secret(text, self._api_key)

    def _request_error(self, exc: BaseException, *, retries: int = 0) -> LlmError:
        message = redact_secret(f"{type(exc).__name__}: {exc}", self._api_key)
        message = " ".join(message.split())
        if len(message) > 500:
            message = message[:497] + "..."
        if retries > 0:
            message += f" (after {retries} retries)"
        return LlmError(f"LLM request failed: {message}")


class LiveBrandAnalyzer:
    """Brand analysis, query generation, and batched creator judging."""

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
        research: str | None = None,
    ) -> BrandProfile:
        profile = self._structured(
            brand_messages(website_content, facebook_page, research),
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
            new = _new_queries(batch.queries, set(previous))
            if not new:
                raise ValueError(
                    "no new queries; every query repeats an earlier round"
                )
            if all(_contains_brand_alias(query, brand) for query in new):
                raise ValueError(
                    "every query names the brand; search for the topic instead"
                )

        batch = self._structured(
            query_messages(brand, previous_queries, round),
            QueryBatch,
            accept,
        )
        fresh = [
            query
            for query in _new_queries(batch.queries, set(previous))
            if not _contains_brand_alias(query, brand)
        ]
        return fresh[:MAX_QUERIES]

    def judge(
        self,
        brief: BrandProfile,
        kols: list[KolProfile],
    ) -> list[Judgement]:
        """Judge all creators in one chat call and clamp every score.

        Creators the model leaves out are asked for once more in a smaller
        call. Any still missing are returned unjudged; the caller reports them.
        """
        if not kols:
            return []
        by_handle = self._judge_items(brief, kols)
        missing = [
            kol for kol in kols if normalize_handle(kol.handle) not in by_handle
        ]
        if missing:
            logger.warning(
                "judge skipped %d of %d creators; asking again", len(missing), len(kols)
            )
            by_handle.update(self._judge_items(brief, missing))

        judgements: list[Judgement] = []
        for kol in kols:
            item = by_handle.get(normalize_handle(kol.handle))
            if item is None:
                continue
            judgements.append(
                Judgement(
                    handle=kol.handle,
                    judge=_unit(item.judge),
                    audience=_unit(item.audience),
                    audience_confidence=_unit(item.audience_confidence),
                    reason=item.reason.strip(),
                    risky=item.risky,
                    account_type=item.account_type,
                    content=_content_analysis(item, kol),
                )
            )
        return judgements

    def _judge_items(
        self,
        brief: BrandProfile,
        kols: list[KolProfile],
    ) -> dict[str, JudgementItem]:
        batch = self._structured(judge_messages(brief, kols), JudgementBatch)
        known = {normalize_handle(kol.handle) for kol in kols}
        by_handle: dict[str, JudgementItem] = {}
        for item in batch.creators:
            key = normalize_handle(item.handle)
            if key in known and key not in by_handle:
                by_handle[key] = item
        return by_handle

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


def _content_analysis(item: JudgementItem, kol: KolProfile) -> ContentAnalysis | None:
    """Build the display fields. A style score must cite a video the card showed."""
    videos = judge_videos(kol)
    if not videos:
        return None
    evidence: list[str] = []
    numbers = item.style_evidence if isinstance(item.style_evidence, list) else []
    for number in numbers:
        if isinstance(number, int) and 1 <= number <= len(videos):
            url = videos[number - 1].url
            if url and url not in evidence:
                evidence.append(url)
    style_fit: float | None = None
    if evidence and isinstance(item.style_fit, (int, float)):
        style_fit = _unit(item.style_fit)
    formats = known_content_formats(item.content_formats)
    style = _text(item.content_style)
    if style_fit is None and not formats and not style:
        return None
    return ContentAnalysis(
        formats=formats,
        style=style,
        style_fit=style_fit,
        style_reason=_text(item.style_reason) if style_fit is not None else "",
        evidence_urls=evidence[:3],
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


_BRAND_LIST_FIELDS = (
    "brand_names",
    "key_products",
    "topic_keywords_th",
    "topic_keywords_en",
    "competitors",
    "retailers",
    "brand_handles",
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
    # The follower target comes from tier_label in code, never from the model,
    # which tends to copy the brand page's own follower count.
    out.pop("target_followers", None)
    for name in _BRAND_LIST_FIELDS:
        if name in out:
            out[name] = _as_list(out[name])
    if "content_formats" in out:
        out["content_formats"] = known_content_formats(
            _as_list(out["content_formats"])
        )
    audience = out.get("target_audience")
    if isinstance(audience, dict) and "interests" in audience:
        out["target_audience"] = {
            **audience,
            "interests": _as_list(audience["interests"]),
        }
    return out


def _contains_brand_alias(query: str, brand: BrandProfile) -> bool:
    """True when a query repeats the brand's own name or handle.

    Uses the same minimum alias length as the graph's recall guard, so a short
    brand name such as "Sun" cannot knock out every topic query.
    """
    text = normalize_for_match(query)
    return any(
        alias in text
        for alias in (
            normalize_for_match(value)
            for value in (*brand.brand_names, *brand.brand_handles)
        )
        if len(alias) >= MIN_ALIAS_CHARS
    )


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


def _unit(value: float) -> float:
    """Clamp a model score into [0, 1]; non-finite values become 0."""
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


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
