import asyncio
import ipaddress
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from ..integrations import Integrations, build_integrations
from ..integrations.apify import ApifyConfigError
from ..integrations.embeddings import EmbedConfigError
from ..integrations.llm_live import LlmConfigError
from ..integrations.web_reader_live import is_public_address
from ..models import MatchRequest
from .runner import run_job
from .settings import FlagUnavailable, LiveFlags, Mode, Settings
from .store import TERMINAL_EVENTS, TERMINAL_STATUSES, Job, JobStore

# Same as the CLI's --max-enrich default: each enrichment is a paid Apify run.
_MAX_ENRICH = 10
_POLL_SECONDS = 0.25

IntegrationsFactory = Callable[[LiveFlags], Integrations]
Submit = Callable[[Callable[[], None]], object]


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    website_url: HttpUrl
    facebook_url: HttpUrl
    top_k: int = Field(default=10, ge=1, le=200)
    # None means "use the mode's default" (see Settings.resolve_flags).
    live_web: bool | None = None
    live_fb: bool | None = None
    live_kols: bool | None = None
    live_llm: bool | None = None
    live_embed: bool | None = None

    @field_validator("website_url")
    @classmethod
    def _public_website(cls, url: HttpUrl) -> HttpUrl:
        # Literal addresses only; names are resolved and checked by the web reader.
        host = (url.host or "").strip("[]")
        if host == "localhost" or (_is_ip_literal(host) and not is_public_address(host)):
            raise ValueError("website_url must not point at a private or local address")
        return url

    @field_validator("facebook_url")
    @classmethod
    def _facebook_host(cls, url: HttpUrl) -> HttpUrl:
        host = url.host or ""
        if host not in {"facebook.com", "fb.com"} and not host.endswith(".facebook.com"):
            raise ValueError("facebook_url must be a facebook.com or fb.com page")
        return url


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


class JobCreated(BaseModel):
    job_id: str


class Health(BaseModel):
    status: Literal["ok"]
    mode: Mode
    defaults: LiveFlags
    credentials: dict[str, bool]
    models: dict[str, str | None]


def _live_integrations(settings: Settings, flags: LiveFlags) -> Integrations:
    return build_integrations(
        **flags.model_dump(),
        apify_token=(
            settings.apify_token.get_secret_value() if settings.apify_token else None
        ),
        llm_api_key=settings.llm_key(),
    )


def create_app(
    settings: Settings | None = None,
    *,
    integrations_factory: IntegrationsFactory | None = None,
    submit: Submit | None = None,
) -> FastAPI:
    settings = settings or Settings()

    make_integrations = integrations_factory or (
        lambda flags: _live_integrations(settings, flags)
    )
    executor = ThreadPoolExecutor(settings.job_workers, thread_name_prefix="job")
    submit = submit or executor.submit
    # One permit per job from POST until run_job returns, so queued and running count together.
    slots = threading.Semaphore(settings.max_queued_jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = JobStore(settings.jobs_db)
        app.state.store.fail_interrupted()
        yield
        # Queued jobs stay "queued" in the store; the next start fails them as interrupted.
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="TikTok KOL Matcher", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def store(request: Request) -> JobStore:
        return request.app.state.store

    Store = Annotated[JobStore, Depends(store)]

    def existing_job(job_id: str, store: Store) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    @app.get("/api/health")
    def health() -> Health:
        return Health(
            status="ok",
            mode=settings.mode,
            defaults=settings.default_flags(),
            credentials=settings.credentials(),
            models=settings.models(),
        )

    @app.post("/api/jobs", status_code=202)
    def create_job(body: JobCreate, store: Store) -> JobCreated:
        request = MatchRequest(
            website_url=body.website_url,
            facebook_url=body.facebook_url,
            top_k=body.top_k,
        )
        try:
            flags = settings.resolve_flags(body.model_dump(include=set(LiveFlags.model_fields)))
            integrations = make_integrations(flags)
        except (FlagUnavailable, ApifyConfigError, LlmConfigError, EmbedConfigError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if not slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail=f"{settings.max_queued_jobs} jobs are already queued or running "
                "(MAX_QUEUED_JOBS); retry when one finishes",
            )
        try:
            job = store.create(request, flags)
        except BaseException:
            slots.release()
            raise
        enrich_limit = _MAX_ENRICH if flags.live_kols else None

        def work() -> None:
            try:
                run_job(store, job.id, integrations, enrich_limit=enrich_limit)
            finally:
                slots.release()

        submit(work)
        return JobCreated(job_id=job.id)

    @app.get("/api/jobs")
    def list_jobs(store: Store, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[Job]:
        return store.recent(limit)

    @app.get("/api/jobs/{job_id}")
    def get_job(job: Annotated[Job, Depends(existing_job)]) -> Job:
        return job

    @app.get("/api/jobs/{job_id}/events", response_class=EventSourceResponse)
    async def job_events(
        request: Request,
        job: Annotated[Job, Depends(existing_job)],
        store: Store,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        # Replay and live share one path: poll stored events after the cursor.
        cursor = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
        while True:
            # Status is read before the events: a job that finishes between the two
            # reads still delivers its terminal event instead of ending early.
            current = await run_in_threadpool(store.get, job.id)
            finished = current is not None and current.status in TERMINAL_STATUSES
            for event in await run_in_threadpool(store.events_after, job.id, cursor):
                yield ServerSentEvent(id=str(event.seq), event=event.type, data=event)
                cursor = event.seq
                if event.type in TERMINAL_EVENTS:
                    return
            # finished here means the cursor already sits past the terminal event.
            if finished or await request.is_disconnected():
                return
            await asyncio.sleep(_POLL_SECONDS)

    return app


app = create_app()
