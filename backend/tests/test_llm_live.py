"""Offline tests for the OpenAI-compatible LLM client.

The official SDK talks HTTP through httpx2, so wire tests use
``httpx2.MockTransport``. A fake ``chat.completions`` client covers failures
that must redact the key even when the exception text is not an HTTP body.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import httpx2
import openai
import pytest

from app.cli import main
from app.graph import build_graph, initial_state
from app.integrations import build_integrations, mock_integrations
from app.integrations.embeddings import MockEmbedder
from app.integrations.facebook import MockFacebookScraper
from app.integrations.kol_search import MockKolSearch
from app.integrations.llm import MockBrandAnalyzer, heuristic_score
from app.integrations.llm_live import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LlmConfigError,
    LlmError,
    LlmOutputError,
    LlmSettings,
    LiveBrandAnalyzer,
    OpenAIChatClient,
    llm_api_key_from_env,
    parse_json_object,
    redact_secret,
)
from app.integrations.web_reader import MockWebReader
from app.models import (
    BrandProfile,
    FacebookPage,
    KolProfile,
    MatchRequest,
    MatchWeights,
    TargetAudience,
)
from app.nodes import make_nodes
from app.state import MatchState

_BACKEND = Path(__file__).resolve().parents[1]
_KEY = "sk-test-llm-key-do-not-leak"
_KEY_ENVS = (
    "LLM_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_RETRY_BACKOFF",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_DISABLE_REASONING",
    "LLM_NO_THINK_MESSAGE",
    "LLM_RERANK_BATCH_SIZE",
)

_BRAND_JSON = {
    "category": "blue-light glasses",
    "products": ["blue-light glasses"],
    "target_audience": {
        "age_min": 18,
        "age_max": 30,
        "gender": "any",
        "interests": ["gaming"],
        "location": "Thailand",
    },
    "tone": "friendly",
    "content_style": "short reviews",
    "keywords_th": ["แว่น"],
    "keywords_en": ["blue light glasses"],
    "hashtags": ["#glasses"],
    "preferred_tiers": ["micro"],
    "reasoning": "The site sells glasses.",
}


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def _completion(content, *, reasoning=None, finish="stop") -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
    }


class _Queue:
    def __init__(self, items: list[tuple[int, dict]]) -> None:
        self.items = list(items)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body = self.items.pop(0)
        return httpx2.Response(status, json=body)


def _analyzer(handler, **overrides) -> LiveBrandAnalyzer:
    settings = LlmSettings(
        base_url=overrides.get("base_url", "https://llm.test/v1"),
        model=overrides.get("model", "test-model"),
        api_key=overrides.get("api_key", _KEY),
        timeout=overrides.get("timeout", DEFAULT_TIMEOUT),
        max_retries=overrides.get("max_retries", DEFAULT_MAX_RETRIES),
        retry_backoff=overrides.get("retry_backoff", 0.0),
        max_tokens=overrides.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=overrides.get("temperature", DEFAULT_TEMPERATURE),
        reasoning=overrides.get("reasoning", "off"),
        no_think_message=overrides.get("no_think_message", "/no_think"),
        rerank_batch_size=overrides.get("rerank_batch_size", DEFAULT_RERANK_BATCH_SIZE),
    )
    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return LiveBrandAnalyzer(settings=settings, http_client=http)


def _brand() -> BrandProfile:
    return BrandProfile.model_validate(_BRAND_JSON)


def _page() -> FacebookPage:
    return FacebookPage(
        page_name="Sample Page",
        title="Glasses",
        url="https://www.facebook.com/example",
        categories=["Product/service"],
        intro="Blue-light glasses for long screen days.",
        followers=10,
    )


def _kol(handle: str = "Alpha", **overrides) -> KolProfile:
    data = {
        "handle": handle,
        "url": f"https://www.tiktok.com/@{handle}",
        "nickname": handle,
        "followers": 12000,
        "bio": "reviews gadgets",
        "language": "th",
        "region": "TH",
        "median_engagement_rate": 0.05,
        "content_summary": "topics: gaming, tech",
        "source": "apify",
    }
    data.update(overrides)
    return KolProfile.model_validate(data)


def _request() -> MatchRequest:
    return MatchRequest(
        website_url="https://example.com",
        facebook_url="https://www.facebook.com/example",
        top_k=3,
    )


def _body(request: httpx2.Request) -> dict:
    return json.loads(request.content)


class _FakeCompletions:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs) -> None:
        self.calls += 1
        raise self.error


class _ScriptedCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _fake_completion(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _status_error(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    message: str | None = None,
) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://llm.test/v1/chat/completions")
    response = httpx2.Response(status, headers=headers, request=request)
    return openai.APIStatusError(
        message or f"HTTP {status}",
        response=response,
        body=None,
    )


class _ScriptedChat:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        if not self._replies:
            raise LlmOutputError("LLM returned empty content after one retry")
        return self._replies.pop(0)


def test_parse_json_object_strips_fences_and_keeps_the_first_object():
    raw = 'note\n```json\n{"a": 1, "b": {"c": 2}}\n```\n{"d": 3}'
    assert parse_json_object(raw) == {"a": 1, "b": {"c": 2}}


def test_parse_json_object_rejects_an_array():
    with pytest.raises(LlmOutputError, match="must be an object"):
        parse_json_object("[1, 2]")


def test_redact_secret_covers_the_raw_and_encoded_forms():
    secret = "sk-test/llm+key"
    encoded = quote(secret, safe="")
    assert encoded != secret
    text = f"header {secret} query {encoded} tail"
    assert redact_secret(text, secret) == "header [redacted] query [redacted] tail"
    assert secret not in redact_secret(text, secret)


def test_default_settings_and_key_fallback(monkeypatch):
    _clear_llm_env(monkeypatch)
    settings = LlmSettings.from_env()
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.model == DEFAULT_MODEL
    assert settings.timeout == DEFAULT_TIMEOUT == 60
    assert settings.max_retries == DEFAULT_MAX_RETRIES == 2
    assert settings.retry_backoff == DEFAULT_RETRY_BACKOFF == 2.0
    assert settings.max_tokens == DEFAULT_MAX_TOKENS == 4096
    assert settings.temperature == DEFAULT_TEMPERATURE
    assert settings.reasoning == "extra_body"
    assert settings.rerank_batch_size == 5
    assert settings.api_key == ""

    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    assert llm_api_key_from_env() == "oa-key"
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    assert LlmSettings.from_env().api_key == "nv-key"
    monkeypatch.setenv("LLM_API_KEY", "llm-key-secret")
    resolved = LlmSettings.from_env()
    assert resolved.api_key == "llm-key-secret"
    assert "llm-key-secret" not in repr(resolved)
    assert LlmSettings.from_env(api_key="explicit-key").api_key == "explicit-key"
    assert LlmSettings.from_env(api_key="  ").api_key == ""


def test_settings_overrides_and_invalid_values(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_RERANK_BATCH_SIZE", "3")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "system")
    monkeypatch.setenv("LLM_NO_THINK_MESSAGE", "do not think")
    settings = LlmSettings.from_env()
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"
    assert settings.timeout == 30
    assert settings.max_tokens == 2048
    assert settings.temperature == 0
    assert settings.rerank_batch_size == 3
    assert settings.reasoning == "system"
    assert settings.no_think_message == "do not think"

    monkeypatch.delenv("LLM_DISABLE_REASONING")
    assert LlmSettings.from_env().reasoning == "off"
    monkeypatch.setenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    assert LlmSettings.from_env().reasoning == "off"  # custom model
    monkeypatch.setenv("LLM_MODEL", DEFAULT_MODEL)
    assert LlmSettings.from_env().reasoning == "extra_body"
    monkeypatch.setenv("LLM_DISABLE_REASONING", "off")
    assert LlmSettings.from_env().reasoning == "off"
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    monkeypatch.setenv("LLM_DISABLE_REASONING", "1")
    assert LlmSettings.from_env().reasoning == "both"
    monkeypatch.setenv("LLM_DISABLE_REASONING", "extra_body")
    assert LlmSettings.from_env().reasoning == "extra_body"
    monkeypatch.setenv("LLM_DISABLE_REASONING", "off")
    assert LlmSettings.from_env().reasoning == "off"

    monkeypatch.setenv("LLM_TIMEOUT", "nope")
    with pytest.raises(LlmConfigError, match="LLM_TIMEOUT"):
        LlmSettings.from_env()
    monkeypatch.setenv("LLM_TIMEOUT", "0")
    with pytest.raises(LlmConfigError, match="LLM_TIMEOUT"):
        LlmSettings.from_env()
    monkeypatch.delenv("LLM_TIMEOUT")
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    with pytest.raises(LlmConfigError, match="LLM_MAX_TOKENS"):
        LlmSettings.from_env()
    monkeypatch.delenv("LLM_MAX_TOKENS")
    monkeypatch.setenv("LLM_DISABLE_REASONING", "maybe")
    with pytest.raises(LlmConfigError, match="LLM_DISABLE_REASONING"):
        LlmSettings.from_env()
    monkeypatch.setenv("LLM_BASE_URL", "ftp://example.test")
    monkeypatch.delenv("LLM_DISABLE_REASONING")
    with pytest.raises(LlmConfigError, match="LLM_BASE_URL"):
        LlmSettings.from_env()


def test_retry_settings_from_env_and_invalid_values(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("LLM_RETRY_BACKOFF", "0.5")
    settings = LlmSettings.from_env()
    assert settings.max_retries == 4
    assert settings.retry_backoff == 0.5

    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RETRY_BACKOFF", "0")
    settings = LlmSettings.from_env()
    assert settings.max_retries == 0
    assert settings.retry_backoff == 0.0

    monkeypatch.setenv("LLM_MAX_RETRIES", "-1")
    with pytest.raises(LlmConfigError, match="LLM_MAX_RETRIES"):
        LlmSettings.from_env()
    monkeypatch.setenv("LLM_MAX_RETRIES", "nope")
    with pytest.raises(LlmConfigError, match="LLM_MAX_RETRIES"):
        LlmSettings.from_env()

    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_RETRY_BACKOFF", "-0.5")
    with pytest.raises(LlmConfigError, match="LLM_RETRY_BACKOFF"):
        LlmSettings.from_env()
    monkeypatch.setenv("LLM_RETRY_BACKOFF", "nope")
    with pytest.raises(LlmConfigError, match="LLM_RETRY_BACKOFF"):
        LlmSettings.from_env()


def test_success_parses_fenced_json_and_sends_configured_request():
    fenced = "```json\n" + json.dumps(_BRAND_JSON) + "\n```"
    queue = _Queue([(200, _completion(fenced))])
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", _page())
    finally:
        analyzer.close()

    assert profile.category == "blue-light glasses"
    assert profile.keywords_th == ["แว่น"]
    assert len(queue.requests) == 1
    request = queue.requests[0]
    body = _body(request)
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {_KEY}"
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.2
    assert "chat_template_kwargs" not in body
    assert body["messages"][0]["content"].startswith("You extract")
    assert "Sample Page" in body["messages"][1]["content"]
    assert _KEY not in request.content.decode()


def test_none_content_is_retried_once_then_succeeds():
    queue = _Queue(
        [
            (200, _completion(None, reasoning="thinking about the brand", finish="length")),
            (200, _completion(json.dumps(_BRAND_JSON))),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.products == ["blue-light glasses"]
    assert len(queue.requests) == 2


def test_none_content_twice_is_a_final_error():
    queue = _Queue(
        [
            (200, _completion(None, reasoning="still thinking", finish="length")),
            (200, _completion("  ", reasoning="still thinking", finish="length")),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        with pytest.raises(LlmOutputError, match="empty content after one retry") as exc:
            analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert len(queue.requests) == 2
    assert "reasoning_content" in str(exc.value)
    assert _KEY not in str(exc.value)


def test_invalid_json_is_retried_with_the_validation_error():
    queue = _Queue(
        [
            (200, _completion("sure {not json")),
            (200, _completion(json.dumps(_BRAND_JSON))),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.tone == "friendly"
    assert len(queue.requests) == 2
    retry = _body(queue.requests[1])["messages"]
    assert any("failed validation" in message["content"] for message in retry)
    assert any("not a JSON object" in message["content"] for message in retry)


def test_invalid_json_twice_raises_and_redacts_the_key():
    bad = json.dumps({"category": _KEY})
    queue = _Queue([(200, _completion(bad)), (200, _completion(bad))])
    analyzer = _analyzer(queue)
    try:
        with pytest.raises(LlmOutputError, match="structured output failed after one retry") as exc:
            analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert len(queue.requests) == 2
    assert _KEY not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    retry_user = _body(queue.requests[1])["messages"][-1]["content"]
    assert _KEY not in retry_user
    assert "[redacted]" in retry_user


def test_http_error_redacts_the_key_and_is_not_retried():
    queue = _Queue(
        [
            (
                401,
                {"error": {"message": f"bad key {_KEY}", "type": "invalid_request_error"}},
            )
        ]
    )
    analyzer = _analyzer(queue)
    try:
        with pytest.raises(LlmError, match="LLM request failed") as exc:
            analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert len(queue.requests) == 1
    assert _KEY not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__


def test_fake_client_exception_redacts_raw_and_encoded_key():
    secret = "sk-test/llm+key"
    encoded = quote(secret, safe="")
    fake = _FakeCompletions(RuntimeError(f"failed {secret} encoded {encoded}"))
    client = OpenAIChatClient(
        LlmSettings(api_key=secret, base_url="https://llm.test/v1"),
        client=fake,
    )
    with pytest.raises(LlmError, match="LLM request failed") as exc:
        client.complete([{"role": "user", "content": "hi"}])

    assert fake.calls == 1
    assert secret not in str(exc.value)
    assert encoded not in str(exc.value)
    assert str(exc.value).count("[redacted]") >= 2
    assert exc.value.__cause__ is None


def test_transient_500_is_retried_then_succeeds():
    queue = _Queue(
        [
            (500, {"error": {"message": "temporary failure"}}),
            (200, _completion(json.dumps(_BRAND_JSON))),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.category == "blue-light glasses"
    assert len(queue.requests) == 2
    assert analyzer._chat.retry_count == 1


def test_429_and_503_use_doubling_backoff():
    sleeps: list[float] = []
    fake = _ScriptedCompletions(
        [
            _status_error(429),
            _status_error(503),
            _fake_completion(json.dumps(_BRAND_JSON)),
        ]
    )
    client = OpenAIChatClient(
        LlmSettings(
            api_key=_KEY,
            base_url="https://llm.test/v1",
            max_retries=2,
            retry_backoff=2.0,
        ),
        client=fake,
        sleep=sleeps.append,
    )

    result = client.complete([{"role": "user", "content": "hi"}])

    assert json.loads(result)["category"] == "blue-light glasses"
    assert fake.calls == 3
    assert client.retry_count == 2
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize(("header", "expected"), [("7", 7.0), ("100", 30.0)])
def test_retry_after_header_controls_backoff(header, expected):
    sleeps: list[float] = []
    fake = _ScriptedCompletions(
        [
            _status_error(429, headers={"Retry-After": header}),
            _fake_completion("{}"),
        ]
    )
    client = OpenAIChatClient(
        LlmSettings(
            api_key=_KEY,
            base_url="https://llm.test/v1",
            retry_backoff=2.0,
        ),
        client=fake,
        sleep=sleeps.append,
    )

    assert client.complete([{"role": "user", "content": "hi"}]) == "{}"
    assert sleeps == [expected]


def test_timeout_retries_then_raises_after_two_retries():
    class _TimeoutWithKey(openai.APITimeoutError):
        def __str__(self) -> str:
            return f"timeout with {_KEY}"

    request = httpx2.Request("POST", "https://llm.test/v1/chat/completions")
    fake = _ScriptedCompletions(
        [_TimeoutWithKey(request=request) for _ in range(3)]
    )
    sleeps: list[float] = []
    client = OpenAIChatClient(
        LlmSettings(
            api_key=_KEY,
            base_url="https://llm.test/v1",
            max_retries=2,
            retry_backoff=0.0,
        ),
        client=fake,
        sleep=sleeps.append,
    )

    with pytest.raises(LlmError, match="after 2 retries") as exc:
        client.complete([{"role": "user", "content": "hi"}])

    assert fake.calls == 3
    assert client.retry_count == 2
    assert _KEY not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    assert exc.value.__cause__ is None


def test_connection_error_is_retried_then_succeeds():
    request = httpx2.Request("POST", "https://llm.test/v1/chat/completions")
    fake = _ScriptedCompletions(
        [
            openai.APIConnectionError(request=request),
            _fake_completion("{}"),
        ]
    )
    sleeps: list[float] = []
    client = OpenAIChatClient(
        LlmSettings(
            api_key=_KEY,
            base_url="https://llm.test/v1",
            retry_backoff=0.0,
        ),
        client=fake,
        sleep=sleeps.append,
    )

    assert client.complete([{"role": "user", "content": "hi"}]) == "{}"
    assert fake.calls == 2
    assert client.retry_count == 1
    assert sleeps == [0.0]


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_other_client_errors_are_not_retried(status):
    queue = _Queue([(status, {"error": {"message": "no retry"}})])
    analyzer = _analyzer(queue)
    try:
        with pytest.raises(LlmError, match="LLM request failed"):
            analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert len(queue.requests) == 1
    assert analyzer._chat.retry_count == 0


def test_408_is_retried():
    queue = _Queue(
        [
            (408, {"error": {"message": "request timeout"}}),
            (200, _completion(json.dumps(_BRAND_JSON))),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.category == "blue-light glasses"
    assert len(queue.requests) == 2
    assert analyzer._chat.retry_count == 1


def test_zero_retries_raises_after_one_request():
    queue = _Queue([(500, {"error": {"message": "temporary failure"}})])
    analyzer = _analyzer(queue, max_retries=0)
    try:
        with pytest.raises(LlmError, match="LLM request failed") as exc:
            analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert len(queue.requests) == 1
    assert analyzer._chat.retry_count == 0
    assert "after" not in str(exc.value)


def test_retry_logs_a_warning_without_the_key(caplog):
    caplog.set_level(logging.WARNING, logger="app.integrations.llm_live")
    queue = _Queue(
        [
            (500, {"error": {"message": f"temporary failure {_KEY}"}}),
            (200, _completion(json.dumps(_BRAND_JSON))),
        ]
    )
    analyzer = _analyzer(queue)
    try:
        analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    records = [
        record
        for record in caplog.records
        if record.name == "app.integrations.llm_live"
        and record.levelno == logging.WARNING
    ]
    assert records
    messages = "\n".join(record.getMessage() for record in records)
    assert "LLM request failed" in messages
    assert "retry 1 of 2" in messages
    assert _KEY not in messages


@pytest.mark.parametrize(
    ("mode", "expect_marker", "expect_extra"),
    [
        ("system", True, False),
        ("extra_body", False, True),
        ("both", True, True),
    ],
)
def test_reasoning_control_is_provider_specific_and_optional(mode, expect_marker, expect_extra):
    queue = _Queue([(200, _completion(json.dumps(_BRAND_JSON)))])
    analyzer = _analyzer(queue, reasoning=mode, no_think_message="/no_think")
    try:
        analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    body = _body(queue.requests[0])
    if expect_marker:
        assert body["messages"][0] == {"role": "system", "content": "/no_think"}
    else:
        assert body["messages"][0]["content"].startswith("You extract")
    if expect_extra:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert "chat_template_kwargs" not in body


def test_provider_switch_is_base_url_model_and_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)
    queue = _Queue([(200, _completion(json.dumps(_BRAND_JSON)))])
    http = httpx2.Client(transport=httpx2.MockTransport(queue))
    analyzer = LiveBrandAnalyzer(http_client=http)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.category == "blue-light glasses"
    assert analyzer._settings.api_key == _KEY
    assert str(queue.requests[0].url) == "https://api.openai.com/v1/chat/completions"
    assert _body(queue.requests[0])["model"] == "gpt-4o-mini"


def test_generate_queries_drops_previous_and_retries_when_all_repeat():
    prompts: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        user = _body(request)["messages"][-1]["content"]
        prompts.append(user)
        if len(prompts) == 1:
            payload = {"queries": ["Old Query", "old query"]}
        else:
            payload = {"queries": ["แว่น รีวิว", "blue light glasses", "Old Query"]}
        return httpx2.Response(200, json=_completion(json.dumps(payload)))

    analyzer = _analyzer(handler)
    try:
        queries = analyzer.generate_queries(_brand(), ["old query"], 2)
    finally:
        analyzer.close()

    assert queries == ["แว่น รีวิว", "blue light glasses"]
    assert "Round: 2" in prompts[0]
    assert "old query" in prompts[0]
    assert "no new queries" in prompts[1]


def test_generate_queries_caps_an_oversized_batch_at_the_prompt_limit():
    sent = [f"query {i}" for i in range(60)]
    chat = _ScriptedChat([json.dumps({"queries": sent})])
    analyzer = LiveBrandAnalyzer(settings=LlmSettings(api_key=_KEY), chat=chat)

    queries = analyzer.generate_queries(_brand(), [], 1)

    assert queries == sent[:10]


def test_rerank_batches_and_keeps_heuristic_scores_in_the_total():
    kols = [
        _kol("Alpha"),
        _kol("Beta"),
        _kol("Gamma", followers=500, language="en", region="US"),
    ]
    seen: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        user = _body(request)["messages"][-1]["content"]
        handles = re.findall(r"^handle: (.+)$", user, re.M)
        seen.append(handles)
        payload = {
            "scores": [
                {"handle": f"@{handle}", "relevance": 0.8, "reason": f"fits {handle}"}
                for handle in handles
            ]
        }
        return httpx2.Response(200, json=_completion(json.dumps(payload)))

    analyzer = _analyzer(handler, rerank_batch_size=2)
    try:
        ranked = analyzer.rerank(_brand(), kols)
        assert analyzer.rerank(_brand(), []) == []
        state = MatchState(
            request=_request(),
            brand_profile=_brand().model_copy(update={"brand_names": ["Alpha"]}),
            shortlisted=[kols[0]],
            stage="embedding_prefilter",
        )
        update = make_nodes(
            replace(mock_integrations(), llm=analyzer),
            min_candidates=1,
            max_search_rounds=1,
        )["llm_rerank"](state)
    finally:
        analyzer.close()

    assert seen[:2] == [["Alpha", "Beta"], ["Gamma"]]
    assert [item[0].handle for item in ranked] == ["Alpha", "Beta", "Gamma"]
    assert ranked[0][1].relevance == 0.8
    assert ranked[0][2] == "fits Alpha"
    heuristic, _reason = heuristic_score(_brand(), kols[0])
    assert ranked[0][1].engagement == heuristic.engagement
    assert ranked[0][1].audience_fit == heuristic.audience_fit
    assert ranked[0][1].tier_fit == heuristic.tier_fit
    assert ranked[0][1].audience_confidence == heuristic.audience_confidence
    gamma_heuristic, _reason = heuristic_score(_brand(), kols[2])
    assert ranked[2][1].audience_fit == gamma_heuristic.audience_fit == 0.35
    weights = MatchWeights()
    assert (weights.relevance, weights.engagement, weights.audience, weights.tier) == (
        0.4,
        0.25,
        0.2,
        0.15,
    )
    assert weights.apply(ranked[0][1]) == round(
        0.8 * 0.4
        + ranked[0][1].engagement * 0.25
        + ranked[0][1].audience_fit * 0.2
        + ranked[0][1].tier_fit * 0.15,
        4,
    )
    item = update["ranked"][0]
    assert update["stage"] == "reranking"
    assert item.scores.relevance == 0.8
    assert item.total_score == state.request.weights.apply(item.scores)


class _FakeEmbedder:
    """Stand-in for a real embedder: not MockEmbedder, so its scores are evidence."""

    def similarity(self, query, documents, *, titles=None):
        return [0.9] * len(documents)


def _live_rerank_prompt(embedder) -> str:
    kols = [_kol("Alpha")]
    reply = json.dumps(
        {"scores": [{"handle": kol.handle, "relevance": 0.5, "reason": "ok"} for kol in kols]}
    )
    chat = _ScriptedChat([reply])
    analyzer = LiveBrandAnalyzer(settings=LlmSettings(api_key=_KEY), chat=chat)
    state = MatchState(
        request=_request(),
        brand_profile=_brand(),
        shortlisted=kols,
        prefilter_scores=[0.2899],
        stage="embedding_prefilter",
    )
    update = make_nodes(
        replace(mock_integrations(), llm=analyzer, embedder=embedder),
        min_candidates=1,
        max_search_rounds=1,
    )["llm_rerank"](state)
    assert update["stage"] == "reranking", update
    return "\n".join(message["content"] for message in chat.messages[0])


def test_live_rerank_hides_mock_embedder_scores():
    prompt = _live_rerank_prompt(MockEmbedder())
    assert "embedding_similarity" not in prompt
    assert "cosine similarity" not in prompt


def test_live_rerank_sends_real_embedder_scores():
    prompt = _live_rerank_prompt(_FakeEmbedder())
    assert "embedding_similarity: 0.2899" in prompt
    assert "cosine similarity" in prompt


def test_generate_queries_node_keeps_per_round_queries():
    class _Recording(MockBrandAnalyzer):
        def __init__(self) -> None:
            self.seen: tuple[list[str], int] | None = None

        def generate_queries(self, brand, previous_queries, round):
            self.seen = (list(previous_queries), round)
            return ["only new"]

    recorder = _Recording()
    state = MatchState(
        request=_request(),
        brand_profile=_brand(),
        search_queries=["already used"],
        search_rounds=1,
        stage="discovering_kols",
    )
    update = make_nodes(
        replace(mock_integrations(), llm=recorder),
        min_candidates=1,
        max_search_rounds=2,
    )["generate_queries"](state)

    assert recorder.seen == (["already used"], 2)
    assert update["search_queries"] == ["only new"]
    assert update["last_queries"] == ["only new"]
    assert update["stage"] == "generating_queries"


def test_structured_output_failure_stops_the_graph():
    chat = _ScriptedChat([json.dumps({"category": _KEY}), json.dumps({"category": _KEY})])
    analyzer = LiveBrandAnalyzer(settings=LlmSettings(api_key=_KEY), chat=chat)
    graph = build_graph(
        replace(mock_integrations(), llm=analyzer),
        min_candidates=1,
        max_search_rounds=1,
    )
    names: list[str] = []
    final = None
    for mode, payload in graph.stream(
        initial_state(_request()),
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            names.extend(payload)
        else:
            final = payload

    assert final is not None
    assert final["stage"] == "failed"
    # None fields are omitted from the values snapshot.
    assert final.get("brand_profile") is None
    assert final["search_queries"] == []
    assert "generate_queries" not in names
    assert names[-1] == "analyze_brand"
    joined = " ".join(final["errors"])
    assert "analyze_brand: LLM structured output failed after one retry" in joined
    assert _KEY not in joined
    assert "[redacted]" in joined
    assert len(chat.messages) == 2


def test_build_integrations_live_llm_keeps_other_services_mocked(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", _KEY)
    assert isinstance(build_integrations().llm, MockBrandAnalyzer)

    bundle = build_integrations(live_llm=True, llm_api_key="explicit-key")
    try:
        assert isinstance(bundle.llm, LiveBrandAnalyzer)
        assert bundle.llm._settings.api_key == "explicit-key"
        assert isinstance(bundle.web_reader, MockWebReader)
        assert isinstance(bundle.facebook, MockFacebookScraper)
        assert isinstance(bundle.kol_search, MockKolSearch)
        assert isinstance(bundle.embedder, MockEmbedder)
        assert "MOCK" not in (bundle.llm.__class__.__doc__ or "")
    finally:
        bundle.llm.close()

    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-only")
    nvidia = build_integrations(live_llm=True)
    try:
        assert nvidia.llm._settings.api_key == "nv-only"
    finally:
        nvidia.llm.close()

    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(LlmConfigError, match="LLM_API_KEY"):
        build_integrations(live_llm=True, llm_api_key="  ")


def test_help_mentions_live_llm(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["match", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--live-llm" in help_text
    assert "LLM_API_KEY" in help_text


def test_live_llm_without_a_key_exits_2(monkeypatch, capsys):
    _clear_llm_env(monkeypatch)

    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            "https://example.com",
            "--facebook",
            "https://www.facebook.com/example",
            "--live-llm",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "LLM_API_KEY" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_live_llm_flag_wires_the_client_without_calling_it(monkeypatch, capsys):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", _KEY)
    captured: dict = {}

    def spy(integrations=None, **kwargs):
        captured["integrations"] = integrations
        captured["kwargs"] = kwargs

        class _Graph:
            def stream(self, *args, **kwargs):
                return iter([("values", {"stage": "done", "results": [], "errors": []})])

        return _Graph()

    monkeypatch.setattr("app.cli.build_graph", spy)
    exit_code = main(
        [
            "match",
            "--website",
            "https://example.com",
            "--facebook",
            "https://www.facebook.com/example",
            "--top-k",
            "5",
            "--json",
            "--live-llm",
        ]
    )
    captured_io = capsys.readouterr()
    llm = captured["integrations"].llm
    try:
        assert exit_code == 0
        assert isinstance(llm, LiveBrandAnalyzer)
        assert isinstance(captured["integrations"].web_reader, MockWebReader)
        assert isinstance(captured["integrations"].kol_search, MockKolSearch)
        assert llm._settings.api_key == _KEY
        assert llm._settings.base_url == DEFAULT_BASE_URL
        assert llm._settings.model == DEFAULT_MODEL
    finally:
        llm.close()

    assert _KEY not in captured_io.out
    assert _KEY not in captured_io.err


def test_bad_llm_timeout_exits_2_without_the_key(monkeypatch, capsys):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", _KEY)
    monkeypatch.setenv("LLM_TIMEOUT", "nope")

    def fail_build(*args, **kwargs):
        raise AssertionError("graph should not run")

    monkeypatch.setattr("app.cli.build_graph", fail_build)
    exit_code = main(
        [
            "match",
            "--website",
            "https://example.com",
            "--facebook",
            "https://www.facebook.com/example",
            "--live-llm",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "LLM_TIMEOUT" in captured.err
    assert _KEY not in captured.err
    assert _KEY not in captured.out


def test_smoke_script_exits_2_without_a_key(monkeypatch):
    env = os.environ.copy()
    for name in ("LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(_BACKEND / "scripts" / "llm_smoke.py")],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "LLM_API_KEY" in result.stderr
    assert result.stdout == ""


def test_brand_list_fields_given_as_strings_are_split():
    messy = {
        **_BRAND_JSON,
        "hashtags": "#glasses #gaming",
        "keywords_en": "blue light glasses, gaming glasses",
        "products": "clip-on lenses; gaming glasses\nreading glasses",
        "preferred_tiers": "micro",
        "target_audience": {**_BRAND_JSON["target_audience"], "interests": "gaming, study"},
    }
    queue = _Queue([(200, _completion(json.dumps(messy)))])
    analyzer = _analyzer(queue)
    try:
        profile = analyzer.analyze_brand("A shop sells blue-light glasses.", None)
    finally:
        analyzer.close()

    assert profile.hashtags == ["#glasses", "#gaming"]
    assert profile.keywords_en == ["blue light glasses", "gaming glasses"]
    assert profile.products == ["clip-on lenses", "gaming glasses", "reading glasses"]
    assert profile.preferred_tiers == ["micro"]
    assert profile.target_audience.interests == ["gaming", "study"]
    assert len(queue.requests) == 1


def test_analyze_brand_tidies_a_polluted_profile():
    polluted = {
        **_BRAND_JSON,
        "products": [
            "blue-light glasses",
            "free cotton towel from the latest giveaway post",
            "towel",
        ],
        "keywords_th": [
            "แว่นตา",
            (
                "แว่นตากันแสงสีฟ้าสำหรับเกมเมอร์ที่ต้องนั่งหน้าจอคอมพิวเตอร์"
                "เป็นเวลานานทุกวันโดยไม่พักสายตา"
            ),
        ],
        "keywords_en": [
            "blue light glasses",
            "what to look for when buying blue light glasses",
        ],
        "hashtags": ["#Blue Light", "##Glasses", "x" * 40],
        "target_audience": {
            **_BRAND_JSON["target_audience"],
            "age_min": 75,
            "age_max": 18,
            "interests": ["gaming", "too many words in one interest"],
        },
    }
    chat = _ScriptedChat([json.dumps(polluted)])
    analyzer = LiveBrandAnalyzer(settings=LlmSettings(api_key=_KEY), chat=chat)

    profile = analyzer.analyze_brand("A shop sells blue-light glasses.", _page())

    assert profile.products == ["blue-light glasses", "towel"]
    assert profile.keywords_th == ["แว่นตา"]
    assert profile.keywords_en == ["blue light glasses"]
    assert profile.hashtags == ["#BlueLight", "#Glasses"]
    assert profile.target_audience.age_min == 18
    assert profile.target_audience.age_max == 75
    assert profile.target_audience.interests == ["gaming"]
    assert len(chat.messages) == 1
