import logging
from typing import Any

from ..cli import summarize_update
from ..graph import build_graph, initial_state
from ..integrations import DiscoveryLimits, Integrations
from .store import JobStore

_SUMMARY_LIMIT = 160

log = logging.getLogger(__name__)


def _short(text: str) -> str:
    return text if len(text) <= _SUMMARY_LIMIT else text[: _SUMMARY_LIMIT - 3] + "..."


def _is_failure(result: dict[str, Any]) -> bool:
    # _guarded turns an exception into {"errors": [...]} plus stage "failed"
    # for fatal nodes. Scrapers are non-fatal, so their failure has no stage.
    return result.get("stage") == "failed" or set(result) == {"errors"}


def run_job(
    store: JobStore,
    job_id: str,
    integrations: Integrations,
    *,
    enrich_limit: int | None = None,
    discovery_limits: DiscoveryLimits | None = None,
) -> None:
    """Run one job's graph to completion, recording per-node events.

    Never raises: any failure ends the job as ``failed`` whenever the store
    accepts the write, so the event stream always gets its terminal event.
    ``discovery_limits`` comes from the API request so bad env values are a
    request error; a direct call falls back to ``DISCOVERY_*`` env values.
    """
    final: dict[str, Any] = {}
    try:
        job = store.transition(job_id, "running")
        discovery = discovery_limits or DiscoveryLimits.from_env()
        graph = build_graph(
            integrations,
            min_candidates=discovery.min_thai_candidates,
            max_search_rounds=discovery.max_rounds,
            enrich_limit=enrich_limit,
        )
        for mode, payload in graph.stream(
            initial_state(job.request),
            stream_mode=["tasks", "values"],
        ):
            if mode == "values":
                final = payload
                store.progress(
                    job_id,
                    stage=payload["stage"],
                    errors=payload["errors"],
                    brand_profile=payload.get("brand_profile"),
                    rejected=payload.get("rejected") or {},
                    cost_log=payload.get("cost_log") or [],
                )
            elif "input" in payload:
                store.add_event(job_id, node=payload["name"], status="started")
            else:
                result = payload.get("result") or {}
                failed = payload.get("error") is not None or _is_failure(result)
                summary = (
                    (result.get("errors") or [str(payload.get("error"))])[-1]
                    if failed
                    else summarize_update(result, node=payload["name"])
                )
                store.add_event(
                    job_id,
                    node=payload["name"],
                    status="failed" if failed else "finished",
                    summary=_short(summary),
                )

        errors = list(final.get("errors") or [])
        rejected = final.get("rejected") or {}
        cost_log = final.get("cost_log") or []
        if final.get("stage") == "done":
            results = final.get("results") or []
            store.transition(
                job_id,
                "succeeded",
                stage="done",
                errors=errors,
                rejected=rejected,
                cost_log=cost_log,
                results=results,
                summary=f"ranked {len(results)} KOLs",
            )
        else:
            store.transition(
                job_id,
                "failed",
                stage=final.get("stage") or "failed",
                errors=errors,
                rejected=rejected,
                cost_log=cost_log,
                summary=_short(errors[-1] if errors else "pipeline stopped before output"),
            )
    except Exception as exc:  # noqa: BLE001 - a job must never stay queued or running
        errors = [*(final.get("errors") or []), f"pipeline: {exc}"]
        try:
            store.transition(job_id, "failed", errors=errors, summary=_short(errors[-1]))
        except Exception as store_exc:  # noqa: BLE001
            # Only the store's own error is logged: the pipeline error may quote page text.
            log.error("job %s could not be marked failed: %s", job_id, store_exc)
