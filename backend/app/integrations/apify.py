"""Minimal Apify REST client for synchronous actor runs.

The token is read from the ``token`` argument or ``APIFY_TOKEN``. It is sent
only as an ``Authorization`` header and is stripped from every error message.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, NoReturn
from urllib.parse import quote

import httpx

# Official sync endpoint. Actor ids use ``owner~name`` (slash becomes tilde).
# https://docs.apify.com/api/v2/act-run-sync-get-dataset-items-post
_RUN_SYNC_ITEMS = (
    "https://api.apify.com/v2/actors/{actor_id}/run-sync-get-dataset-items"
)

# Connection failures and these statuses are safe to retry. 408 is not: the
# sync HTTP call gave up, but the paid run may still be executing.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class ApifyError(RuntimeError):
    """An Apify run failed or returned a payload we cannot use."""


class ApifyAuthError(ApifyError):
    """Apify rejected ``APIFY_TOKEN`` (HTTP 401 or 403)."""


class ApifyConfigError(ApifyError):
    """Live Apify was requested without ``APIFY_TOKEN``."""


def actor_run_sync_url(actor: str) -> str:
    """URL for ``POST /v2/actors/{owner}~{name}/run-sync-get-dataset-items``."""
    actor_id = actor.strip().replace("/", "~")
    return _RUN_SYNC_ITEMS.format(actor_id=actor_id)


def dig(item: dict[str, Any], *path: str) -> Any:
    """Read a nested field, then a flat ``a.b`` key from the actor overview view."""
    current: Any = item
    for key in path:
        if not isinstance(current, dict) or key not in current:
            current = None
            break
        current = current[key]
    if current is not None:
        return current
    flat = ".".join(path)
    if flat in item:
        return item[flat]
    return None


def as_count(value: object) -> int | None:
    """Coerce a documented count field to a non-negative int."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.replace(",", "").strip())
        except ValueError:
            return None
    else:
        return None
    if number < 0:
        return None
    return number


class ApifyClient:
    """POST an actor's run-sync-get-dataset-items endpoint and return items."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 180.0,
        max_retries: int = 1,
        backoff: float = 1.0,
        max_charge_usd: float = 0.50,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved = token if token is not None else os.environ.get("APIFY_TOKEN")
        resolved = (resolved or "").strip()
        if not resolved:
            raise ApifyConfigError(
                "APIFY_TOKEN is not set. Export it before using live KOL search."
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        if max_charge_usd <= 0:
            raise ValueError("max_charge_usd must be positive")

        self._token = resolved
        self._run_timeout = timeout
        self._max_retries = max_retries
        self._backoff = backoff
        self._max_charge_usd = max_charge_usd
        self._sleeper = sleeper
        # The sync endpoint can sit until ``timeout``. Give the socket a little
        # longer so httpx does not cancel the request first.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout + 30.0, connect=10.0),
        )

    def run_sync_get_items(
        self,
        actor: str,
        run_input: dict[str, Any],
        *,
        max_items: int,
    ) -> list[Any]:
        """Run ``actor`` and return the dataset items (a JSON list)."""
        if max_items < 1:
            raise ValueError("max_items must be at least 1")

        url = actor_run_sync_url(actor)
        params = {
            "timeout": str(int(self._run_timeout)),
            "maxItems": str(max_items),
            "limit": str(max_items),
            "maxTotalChargeUsd": _charge_param(self._max_charge_usd),
            "format": "json",
            "restartOnError": "false",
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        attempts = self._max_retries + 1
        last_transport = "request failed"

        for attempt in range(attempts):
            try:
                response = self._client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=run_input,
                )
            except httpx.TimeoutException as exc:
                # A timed-out POST may already have started a paid run.
                self._fail(f"Apify request timed out calling {actor}", exc)
            except httpx.TransportError as exc:
                last_transport = type(exc).__name__
                if attempt >= attempts - 1:
                    self._fail(
                        f"Apify request failed calling {actor}: {last_transport}",
                        exc,
                    )
                self._sleep(attempt)
                continue

            if response.status_code in (401, 403):
                # Body omitted on purpose: it can echo the token.
                self._fail(
                    f"Apify rejected the token (HTTP {response.status_code})",
                    auth=True,
                )

            if response.status_code in (200, 201):
                return self._parse_items(response)

            if response.status_code in _RETRY_STATUS and attempt < attempts - 1:
                self._sleep(attempt)
                continue

            self._fail(f"Apify run for {actor} failed: {self._detail(response)}")

        self._fail(f"Apify request failed calling {actor}: {last_transport}")

    def _parse_items(self, response: httpx.Response) -> list[Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            self._fail("Apify returned a non-JSON dataset payload", exc)

        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                payload = items
            elif payload.get("error"):
                self._fail(f"Apify run failed: {self._detail(response)}")

        if not isinstance(payload, list):
            self._fail("Apify returned a non-list dataset payload")
        return payload

    def _detail(self, response: httpx.Response) -> str:
        message = ""
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = str(err.get("message") or err.get("type") or "")
            elif isinstance(err, str):
                message = err
        if not message:
            message = response.text[:180]
        message = self._redact(message).replace("\n", " ").strip()
        if len(message) > 180:
            message = message[:177] + "..."
        if message:
            return f"HTTP {response.status_code}: {message}"
        return f"HTTP {response.status_code}"

    def _redact(self, text: str) -> str:
        if not text or not self._token:
            return text
        redacted = text.replace(self._token, "[redacted]")
        encoded = quote(self._token, safe="")
        if encoded and encoded != self._token:
            redacted = redacted.replace(encoded, "[redacted]")
        return redacted

    def _sleep(self, attempt: int) -> None:
        delay = self._backoff * (2**attempt)
        if delay > 0:
            self._sleeper(delay)

    def _fail(
        self,
        message: str,
        exc: BaseException | None = None,
        *,
        auth: bool = False,
    ) -> NoReturn:
        clean = self._redact(message)
        error_type = ApifyAuthError if auth else ApifyError
        if exc is not None and self._token not in f"{exc} {exc!r}":
            raise error_type(clean) from exc
        raise error_type(clean) from None


def _charge_param(amount: float) -> str:
    """Stable query value (``0.5`` rather than ``0.5000000001``)."""
    text = f"{amount:.4f}".rstrip("0").rstrip(".")
    return text or "0"
