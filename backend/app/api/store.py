import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..models import BrandProfile, MatchRequest, RankedKol, RemovedAccount
from .settings import LiveFlags

JobStatus = Literal["queued", "running", "succeeded", "failed"]
EventType = Literal["node", "done", "failed"]
NodeStatus = Literal["started", "finished", "failed"]

# The only legal status changes. queued -> failed covers jobs that never
# started (integration setup failed, or the server restarted first).
_NEXT_STATUS: dict[JobStatus, frozenset[JobStatus]] = {
    "queued": frozenset({"running", "failed"}),
    "running": frozenset({"succeeded", "failed"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
}
_TERMINAL_EVENT: dict[JobStatus, EventType] = {"succeeded": "done", "failed": "failed"}
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(_TERMINAL_EVENT)
TERMINAL_EVENTS: frozenset[EventType] = frozenset(_TERMINAL_EVENT.values())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs (created_at);
CREATE TABLE IF NOT EXISTS events (
    job_id TEXT NOT NULL REFERENCES jobs (id),
    seq INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (job_id, seq)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Job(BaseModel):
    id: str
    status: JobStatus
    stage: str
    request: MatchRequest
    flags: LiveFlags
    errors: list[str] = []
    brand_profile: BrandProfile | None = None
    brand_logo_url: str | None = None
    rejected: dict[str, int] = Field(default_factory=dict)
    removed: list[RemovedAccount] = Field(default_factory=list)
    cost_log: list[str] = Field(default_factory=list)
    results: list[RankedKol] = []
    created_at: str
    updated_at: str


LEGACY_JOB_NOTE = (
    "stored by an earlier version; its brand profile and results use an old "
    "format and are not shown"
)


def _parse_job(data: str) -> Job:
    """Load a stored job, downgrading rows written before matching-v2.

    Older rows carry ``request.weights``, an old brand profile, and old result
    rows. They stay listable: the request is kept, the old profile and results
    are dropped, and the job says why.
    """
    try:
        return Job.model_validate_json(data)
    except ValidationError:
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise
    request = raw.get("request")
    if isinstance(request, dict):
        raw["request"] = {
            key: request[key]
            for key in ("website_url", "facebook_url", "top_k")
            if key in request
        }
    raw["brand_profile"] = None
    raw["results"] = []
    errors = raw.get("errors") if isinstance(raw.get("errors"), list) else []
    if LEGACY_JOB_NOTE not in errors:
        errors = [*errors, LEGACY_JOB_NOTE]
    raw["errors"] = errors
    return Job.model_validate(raw)


class Event(BaseModel):
    job_id: str
    seq: int
    type: EventType
    node: str | None = None
    status: NodeStatus | None = None
    summary: str = ""
    timestamp: str


class IllegalTransition(RuntimeError):
    pass


class JobStore:
    """Jobs and their progress events in SQLite.

    Each operation opens its own connection, so any thread can call any method.
    Writes share one lock: event ``seq`` is read-then-insert and must not race.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def create(self, request: MatchRequest, flags: LiveFlags) -> Job:
        now = _now()
        job = Job(
            id=uuid.uuid4().hex,
            status="queued",
            stage="received",
            request=request,
            flags=flags,
            created_at=now,
            updated_at=now,
        )
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, data) VALUES (?, ?, ?, ?)",
                (job.id, job.status, job.created_at, job.model_dump_json()),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _parse_job(row[0]) if row else None

    def recent(self, limit: int) -> list[Job]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT data FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_parse_job(row[0]) for row in rows]

    def progress(
        self,
        job_id: str,
        *,
        stage: str,
        errors: list[str],
        brand_profile: BrandProfile | None = None,
        brand_logo_url: str | None = None,
        rejected: dict[str, int] | None = None,
        cost_log: list[str] | None = None,
    ) -> None:
        """Record a running job's latest stage, errors, and partial output."""
        changes: dict[str, object] = {
            "stage": stage,
            "errors": errors,
            "brand_profile": brand_profile,
            "brand_logo_url": brand_logo_url,
        }
        if rejected is not None:
            changes["rejected"] = rejected
        if cost_log is not None:
            changes["cost_log"] = cost_log
        with self._lock, closing(self._connect()) as conn, conn:
            job = self._load(conn, job_id)
            if job.status != "running":
                raise IllegalTransition(f"job {job_id} is {job.status}, not running")
            self._save(conn, job, **changes)

    def transition(
        self,
        job_id: str,
        status: JobStatus,
        *,
        summary: str = "",
        **changes: object,
    ) -> Job:
        """Move a job to ``status``. A terminal status also appends its terminal event."""
        with self._lock, closing(self._connect()) as conn, conn:
            job = self._load(conn, job_id)
            if status not in _NEXT_STATUS[job.status]:
                raise IllegalTransition(f"job {job_id} cannot go {job.status} -> {status}")
            job = self._save(conn, job, **changes, status=status)
            if status in _TERMINAL_EVENT:
                self._append_event(conn, job_id, type=_TERMINAL_EVENT[status], summary=summary)
        return job

    def add_event(
        self,
        job_id: str,
        *,
        node: str,
        status: NodeStatus,
        summary: str = "",
    ) -> Event:
        with self._lock, closing(self._connect()) as conn, conn:
            return self._append_event(
                conn, job_id, type="node", node=node, status=status, summary=summary
            )

    def events_after(self, job_id: str, seq: int) -> list[Event]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT data FROM events WHERE job_id = ? AND seq > ? ORDER BY seq",
                (job_id, seq),
            ).fetchall()
        return [Event.model_validate_json(row[0]) for row in rows]

    def fail_interrupted(self) -> int:
        """Fail jobs a previous process left queued or running."""
        with closing(self._connect()) as conn:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM jobs WHERE status IN ('queued', 'running')"
                )
            ]
        for job_id in ids:
            job = self.get(job_id)
            self.transition(
                job_id,
                "failed",
                summary="interrupted by restart",
                errors=[*job.errors, "interrupted by restart"],
            )
        return len(ids)

    def _load(self, conn: sqlite3.Connection, job_id: str) -> Job:
        row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return _parse_job(row[0])

    def _save(self, conn: sqlite3.Connection, job: Job, **changes: object) -> Job:
        job = Job.model_validate({**job.model_dump(), **changes, "updated_at": _now()})
        conn.execute(
            "UPDATE jobs SET status = ?, data = ? WHERE id = ?",
            (job.status, job.model_dump_json(), job.id),
        )
        return job

    def _append_event(self, conn: sqlite3.Connection, job_id: str, **fields: object) -> Event:
        (seq,) = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        event = Event(job_id=job_id, seq=seq, timestamp=_now(), **fields)
        conn.execute(
            "INSERT INTO events (job_id, seq, data) VALUES (?, ?, ?)",
            (job_id, seq, event.model_dump_json()),
        )
        return event
