import json
import logging
import sqlite3
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.runner import run_job
from app.api.settings import LiveFlags, Settings
from app.api.store import JobStore
from app.graph import _NODE_ORDER
from app.integrations import mock_integrations
from app.integrations.web_reader import MockWebReader
from app.models import MatchRequest

_ENV_VARS = (
    "DEMO_MODE",
    "APIFY_TOKEN",
    "LLM_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "JINA_API_KEY",
    "JOBS_DB",
    "CORS_ORIGINS",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBED_MODEL",
    "JOB_WORKERS",
    "MAX_QUEUED_JOBS",
)
_BODY = {
    "website_url": "https://example-brand.co.th",
    "facebook_url": "https://www.facebook.com/examplebrand",
    "top_k": 5,
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_client(tmp_path):
    def make(integrations=None, submit=lambda work: work(), **settings):
        app = create_app(
            Settings(jobs_db=tmp_path / "jobs.db", **settings),
            integrations_factory=lambda flags: integrations or mock_integrations(),
            submit=submit,
        )
        return TestClient(app)

    return make


def read_events(client, job_id, **kwargs):
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events", **kwargs) as response:
        assert response.status_code == 200
        block: dict[str, str] = {}
        for line in response.iter_lines():
            if not line:
                if "data" in block:
                    events.append({**json.loads(block["data"]), "sse_id": block["id"]})
                block = {}
            elif not line.startswith(":"):
                key, _, value = line.partition(": ")
                block[key] = value
    return events


class FailingAnalyzer:
    def analyze_brand(self, website_content, facebook_page):
        raise RuntimeError("model unavailable")


def test_job_runs_to_succeeded_with_top_k_results(make_client):
    with make_client() as client:
        created = client.post("/api/jobs", json=_BODY)
        assert created.status_code == 202
        job_id = created.json()["job_id"]

        job = client.get(f"/api/jobs/{job_id}").json()

    assert job["status"] == "succeeded"
    assert job["stage"] == "done"
    assert len(job["results"]) == 5
    assert job["request"]["top_k"] == 5
    assert job["results"][0]["kol"]["handle"]


@pytest.mark.parametrize(
    "body",
    [
        {**_BODY, "facebook_url": "https://example-brand.co.th/examplebrand"},
        {**_BODY, "facebook_url": "https://facebook.com.example-brand.co.th/x"},
        {**_BODY, "website_url": "http://localhost:8000"},
        {**_BODY, "website_url": "http://127.0.0.1:9"},
        {**_BODY, "website_url": "http://[::1]:9/"},
        {**_BODY, "website_url": "http://10.0.0.1"},
    ],
)
def test_private_or_non_facebook_hosts_are_422(make_client, body):
    with make_client() as client:
        response = client.post("/api/jobs", json=body)
        listed = client.get("/api/jobs").json()

    assert response.status_code == 422
    assert listed == []


@pytest.mark.parametrize(
    "facebook_url",
    [
        "https://www.facebook.com/examplebrand",
        "https://m.facebook.com/examplebrand",
        "https://facebook.com/examplebrand",
        "https://fb.com/examplebrand",
    ],
)
def test_facebook_hosts_are_accepted(make_client, facebook_url):
    with make_client() as client:
        response = client.post("/api/jobs", json={**_BODY, "facebook_url": facebook_url})

    assert response.status_code == 202


def test_full_queue_is_429_until_a_job_finishes(make_client):
    pending = []
    with make_client(submit=pending.append, max_queued_jobs=1) as client:
        first = client.post("/api/jobs", json=_BODY)
        second = client.post("/api/jobs", json=_BODY)
        listed = client.get("/api/jobs").json()
        pending.pop()()
        third = client.post("/api/jobs", json=_BODY)

    assert first.status_code == 202
    assert second.status_code == 429
    assert "MAX_QUEUED_JOBS" in second.json()["detail"]
    assert [job["id"] for job in listed] == [first.json()["job_id"]]
    assert third.status_code == 202


def test_job_limits_come_from_the_environment(monkeypatch):
    assert (Settings().job_workers, Settings().max_queued_jobs) == (2, 20)

    monkeypatch.setenv("JOB_WORKERS", "3")
    monkeypatch.setenv("MAX_QUEUED_JOBS", "5")

    assert (Settings().job_workers, Settings().max_queued_jobs) == (3, 5)


def test_unknown_job_is_404(make_client):
    with make_client() as client:
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.get("/api/jobs/nope/events").status_code == 404


def test_list_jobs_newest_first_with_limit(make_client):
    with make_client() as client:
        ids = [client.post("/api/jobs", json=_BODY).json()["job_id"] for _ in range(3)]

        listed = client.get("/api/jobs", params={"limit": 2}).json()

    assert [job["id"] for job in listed] == ids[::-1][:2]


def test_health_without_env_is_demo(make_client):
    with make_client() as client:
        health = client.get("/api/health").json()

    assert health["status"] == "ok"
    assert health["mode"] == "demo"
    assert not any(health["defaults"].values())
    assert not any(health["credentials"].values())


def test_health_reports_integration_default_models(make_client, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  ")
    with make_client() as client:
        models = client.get("/api/health").json()["models"]

    assert models == {
        "llm_model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "llm_host": "integrate.api.nvidia.com",
        "embed_model": "google/embeddinggemma-2",
    }


def test_health_reports_configured_models_and_only_the_llm_host(make_client, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://user:test-secret@llm.example-brand.co.th/v1?key=k")
    monkeypatch.setenv("LLM_MODEL", "example/model-7b")
    monkeypatch.setenv("EMBED_MODEL", "example/embedder")
    with make_client() as client:
        response = client.get("/api/health")

    assert response.json()["models"] == {
        "llm_model": "example/model-7b",
        "llm_host": "llm.example-brand.co.th",
        "embed_model": "example/embedder",
    }
    assert "test-secret" not in response.text


def test_health_demo_mode_wins_over_token(make_client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("APIFY_TOKEN", "test-token-value")
    with make_client() as client:
        response = client.get("/api/health")

    health = response.json()
    assert health["mode"] == "demo"
    assert not any(health["defaults"].values())
    assert health["credentials"]["APIFY_TOKEN"] is True
    assert "test-token-value" not in response.text


def test_health_live_mode_defaults_follow_credentials(make_client, monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "test-token-value")
    with make_client() as client:
        health = client.get("/api/health").json()

    assert health["mode"] == "live"
    assert health["defaults"]["live_web"] is True
    assert health["defaults"]["live_kols"] is True
    assert health["defaults"]["live_fb"] is True
    assert health["defaults"]["live_llm"] is False


def test_live_mode_rejects_flag_without_credential(make_client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "0")
    with make_client() as client:
        kols = client.post("/api/jobs", json={**_BODY, "live_kols": True})
        llm = client.post("/api/jobs", json={**_BODY, "live_llm": True})
        listed = client.get("/api/jobs").json()

    assert kols.status_code == 400
    assert "APIFY_TOKEN" in kols.json()["detail"]
    assert llm.status_code == 400
    assert "LLM_API_KEY" in llm.json()["detail"]
    assert listed == []


def test_demo_mode_forces_live_flags_off(make_client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("APIFY_TOKEN", "test-token-value")
    with make_client() as client:
        body = {**_BODY, "live_web": True, "live_kols": True, "live_llm": True}
        job_id = client.post("/api/jobs", json=body).json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").json()

    assert job["status"] == "succeeded"
    assert not any(job["flags"].values())


def test_events_stream_every_node_then_done(make_client):
    with make_client() as client:
        job_id = client.post("/api/jobs", json=_BODY).json()["job_id"]
        events = read_events(client, job_id)

    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert [e["sse_id"] for e in events] == [str(e["seq"]) for e in events]
    node_events = events[:-1]
    assert all(e["type"] == "node" for e in node_events)
    assert events[-1]["type"] == "done"
    for index, event in enumerate(node_events):
        if event["status"] == "finished":
            assert any(
                earlier["node"] == event["node"] and earlier["status"] == "started"
                for earlier in node_events[:index]
            )
    assert all(e["status"] in {"started", "finished"} for e in node_events)
    finished_order = list(
        dict.fromkeys(e["node"] for e in node_events if e["status"] == "finished")
    )
    assert set(finished_order[:2]) == {"scrape_website", "scrape_facebook"}
    assert finished_order[2:] == list(_NODE_ORDER[2:])
    discover = next(e for e in node_events if e["node"] == "discover_kols" and e["summary"])
    assert "discovered+" in discover["summary"]


def test_events_reconnect_replays_only_later_events(make_client):
    with make_client() as client:
        job_id = client.post("/api/jobs", json=_BODY).json()["job_id"]
        everything = read_events(client, job_id)
        by_header = read_events(client, job_id, headers={"Last-Event-ID": "5"})
        by_query = read_events(client, job_id, params={"after": 5})

    assert [e["seq"] for e in by_header] == [e["seq"] for e in everything[5:]]
    assert by_query == by_header
    assert by_header[-1]["type"] == "done"


def test_events_reconnect_past_the_terminal_event_ends_at_once(make_client):
    with make_client() as client:
        job_id = client.post("/api/jobs", json=_BODY).json()["job_id"]
        last = read_events(client, job_id)[-1]
        started = time.monotonic()
        replayed = read_events(client, job_id, headers={"Last-Event-ID": last["sse_id"]})
        elapsed = time.monotonic() - started

    assert last["type"] == "done"
    assert replayed == []
    assert elapsed < 2


def test_fatal_node_failure_fails_job(make_client):
    broken = replace(mock_integrations(), llm=FailingAnalyzer())
    with make_client(integrations=broken) as client:
        job_id = client.post("/api/jobs", json=_BODY).json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        events = read_events(client, job_id)

    assert job["status"] == "failed"
    assert job["results"] == []
    assert any("model unavailable" in error for error in job["errors"])
    analyze = [e for e in events if e["node"] == "analyze_brand"]
    assert [e["status"] for e in analyze] == ["started", "failed"]
    assert "model unavailable" in analyze[-1]["summary"]
    assert events[-1]["type"] == "failed"
    assert not any(e["node"] == "generate_queries" for e in events)


class FlakyStore(JobStore):
    """Every status write raises until ``failures`` is used up."""

    def __init__(self, path, failures):
        super().__init__(path)
        self.failures = failures

    def transition(self, *args, **kwargs):
        if self.failures:
            self.failures -= 1
            raise sqlite3.OperationalError("disk I/O error")
        return super().transition(*args, **kwargs)


def test_job_still_ends_failed_when_the_first_store_write_raises(tmp_path):
    store = FlakyStore(tmp_path / "jobs.db", failures=1)
    job = store.create(MatchRequest(**_BODY), flags=LiveFlags())

    run_job(store, job.id, mock_integrations())

    after = store.get(job.id)
    assert after.status == "failed"
    assert any("disk I/O error" in error for error in after.errors)
    assert [e.type for e in store.events_after(job.id, 0)] == ["failed"]


def test_runner_logs_instead_of_raising_when_no_write_succeeds(tmp_path, caplog):
    store = FlakyStore(tmp_path / "jobs.db", failures=99)
    job = store.create(MatchRequest(**_BODY), flags=LiveFlags())

    with caplog.at_level(logging.ERROR):
        run_job(store, job.id, mock_integrations())

    assert store.get(job.id).status == "queued"
    assert job.id in caplog.text
    assert "disk I/O error" in caplog.text


def test_restart_fails_jobs_left_running(make_client, tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    request = MatchRequest(**_BODY)
    running = store.create(request, flags=LiveFlags())
    store.transition(running.id, "running")

    with make_client() as client:
        job = client.get(f"/api/jobs/{running.id}").json()
        events = read_events(client, running.id)

    assert job["status"] == "failed"
    assert "interrupted by restart" in job["errors"]
    assert [e["type"] for e in events] == ["failed"]


class GatedWebReader:
    def __init__(self):
        self.gate = threading.Event()

    def read(self, url):
        assert self.gate.wait(5), "test never opened the gate"
        return MockWebReader().read(url)


def test_events_stream_live_while_job_runs(tmp_path):
    reader = GatedWebReader()
    app = create_app(
        Settings(jobs_db=tmp_path / "jobs.db"),
        integrations_factory=lambda flags: replace(mock_integrations(), web_reader=reader),
    )
    types = []
    with TestClient(app) as client:
        job_id = client.post("/api/jobs", json=_BODY).json()["job_id"]
        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            for line in response.iter_lines():
                if line.startswith("event: "):
                    types.append(line.removeprefix("event: "))
                    if not reader.gate.is_set():
                        job = client.get(f"/api/jobs/{job_id}").json()
                        assert job["status"] == "running", "first event arrived mid-run"
                        reader.gate.set()

    assert types[0] == "node"
    assert types[-1] == "done"
