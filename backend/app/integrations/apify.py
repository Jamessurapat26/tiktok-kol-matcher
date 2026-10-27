"""Minimal Apify REST client for synchronous actor runs.

The token is read from the ``token`` argument or ``APIFY_TOKEN``. It is sent
only as an ``Authorization`` header and is stripped from every error message.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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

# Conservative estimates. The sync endpoint returns dataset items only, not
# the run usage or cost, so these numbers cannot be exact.
_FIXED_RUN_USD = 0.01
_DEFAULT_ITEM_USD = 0.01
ESTIMATED_USD_PER_ITEM: dict[str, float] = {
    "clockworks/tiktok-scraper": 0.005,
    "clockworks/tiktok-profile-scraper": 0.005,
    "apify/facebook-pages-scraper": 0.01,
    "apify/facebook-posts-scraper": 0.005,
}


class ApifyError(RuntimeError):
    """An Apify run failed or returned a payload we cannot use."""


class ApifyAuthError(ApifyError):
    """Apify rejected ``APIFY_TOKEN`` (HTTP 401 or 403)."""


class ApifyConfigError(ApifyError):
    """Live Apify was requested without ``APIFY_TOKEN``."""


class ApifyBudgetExceeded(ApifyError):
    """A job-level Apify budget was reached before a run could start."""


class ApifyBudget:
    """Thread-safe estimated spend for one job's Apify calls.

    The run-sync-get-dataset-items endpoint returns only dataset items, not the
    run cost, so ``record`` receives an estimate. The budget is soft: ``check``
    runs before each actor call, and the call that crosses the limit is allowed
    to finish. Spend can therefore exceed the limit by at most that last run.
    """

    def __init__(self, limit_usd: float) -> None:
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        self._limit = float(limit_usd)
        self._spent = 0.0
        self._skipped = 0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._limit - self._spent)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._spent >= self._limit

    @property
    def skipped(self) -> int:
        with self._lock:
            return self._skipped

    def check(self, actor: str) -> None:
        """Raise ``ApifyBudgetExceeded`` once the estimated spend is used up."""
        with self._lock:
            if self._spent >= self._limit:
                self._skipped += 1
                raise ApifyBudgetExceeded(
                    f"Apify job budget of ${self._limit:.2f} reached "
                    f"(estimated spend ${self._spent:.2f}); skipping {actor}"
                )

    def record(self, actor: str, usd: float) -> None:
        """Add an estimated charge. ``actor`` labels the call for callers."""
        del actor
        with self._lock:
            self._spent += float(usd)

    def summary(self) -> str:
        with self._lock:
            return (
                f"Apify budget: estimated ${self._spent:.2f} of "
                f"${self._limit:.2f} spent, {self._skipped} calls skipped"
            )


def estimate_run_cost(
    actor: str,
    items_returned: int,
    max_charge_usd: float,
) -> float:
    """Estimate one run's cost from its item count and per-run cap.

    The sync endpoint does not return usage, so this is an estimate: a small
    fixed per-run amount plus a conservative per-item rate. Unknown actors use
    the default per-item rate. The result never exceeds the run's own cap.
    """
    per_item = ESTIMATED_USD_PER_ITEM.get(actor, _DEFAULT_ITEM_USD)
    estimate = _FIXED_RUN_USD + per_item * max(items_returned, 1)
    return min(max_charge_usd, estimate)


def _positive_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


@dataclass(frozen=True)
class ApifyLimits:
    """Per-run caps and the shared job budget for live Apify calls."""

    job_budget_usd: float
    tiktok_search_max_charge_usd: float
    tiktok_profile_max_charge_usd: float
    facebook_max_charge_usd: float

    @classmethod
    def from_env(cls) -> ApifyLimits:
        job_budget = _positive_env("APIFY_JOB_BUDGET_USD", 1.00)
        tiktok_search = _positive_env("APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD", 0.50)
        tiktok_profile = _positive_env("APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD", 0.50)
        facebook = _positive_env("APIFY_FACEBOOK_MAX_CHARGE_USD", 0.50)
        for name, value in (
            ("APIFY_TIKTOK_SEARCH_MAX_CHARGE_USD", tiktok_search),
            ("APIFY_TIKTOK_PROFILE_MAX_CHARGE_USD", tiktok_profile),
        ):
            if value < 0.50:
                raise ValueError(
                    f"{name} must be at least 0.50, the minimum the TikTok "
                    "actors accept"
                )
        return cls(
            job_budget_usd=job_budget,
            tiktok_search_max_charge_usd=tiktok_search,
            tiktok_profile_max_charge_usd=tiktok_profile,
            facebook_max_charge_usd=facebook,
        )

    def budget(self) -> ApifyBudget:
        """Return a new budget for one job or CLI run."""
        return ApifyBudget(self.job_budget_usd)


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
        budget: ApifyBudget | None = None,
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
        self._budget = budget
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
        if self._budget is not None:
            self._budget.check(actor)

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
                self._record_cap(actor)
                self._fail(f"Apify request timed out calling {actor}", exc)
            except httpx.TransportError as exc:
                # Connection errors may never have reached the server, so no
                # charge is recorded until an attempt gets a response.
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
                try:
                    items = self._parse_items(response)
                except ApifyError:
                    # The run happened; keep the estimate conservative.
                    self._record_cap(actor)
                    raise
                self._record_estimate(actor, len(items))
                return items

            if response.status_code in _RETRY_STATUS and attempt < attempts - 1:
                self._sleep(attempt)
                continue

            if response.status_code == 408 or response.status_code >= 500:
                # The POST may have started a paid run before this failure.
                # Other 4xx (400, 402, 404, 422, 429) are refused before a run starts.
                self._record_cap(actor)
            self._fail(f"Apify run for {actor} failed: {self._detail(response)}")

        self._fail(f"Apify request failed calling {actor}: {last_transport}")

    def _record_estimate(self, actor: str, items_returned: int) -> None:
        if self._budget is None:
            return
        self._budget.record(
            actor,
            estimate_run_cost(actor, items_returned, self._max_charge_usd),
        )

    def _record_cap(self, actor: str) -> None:
        if self._budget is not None:
            self._budget.record(actor, self._max_charge_usd)

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
