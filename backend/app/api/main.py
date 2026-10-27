import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ..integrations import Integrations, build_integrations
from ..integrations.apify import ApifyConfigError
from ..integrations.embeddings import EmbedConfigError
from ..integrations.llm_live import LlmConfigError
from ..models import MatchRequest
from .runner import run_job
from .settings import FlagUnavailable, LiveFlags, Mode, Settings
from .store import TERMINAL_EVENTS, Job, JobStore

# Same as the CLI's --max-enrich default: each enrichment is a paid Apify run.
_MAX_ENRICH = 10
_POLL_SECONDS = 0.25

IntegrationsFactory = Callable[[LiveFlags], Integrations]
Submit = Callable[[Callable[[], None]], None]


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


class JobCreated(BaseModel):
    job_id: str


class Health(BaseModel):
    status: Literal["ok"]
    mode: Mode
    defaults: LiveFlags
    credentials: dict[str, bool]


def _live_integrations(settings: Settings, flags: LiveFlags) -> Integrations:
    return build_integrations(
        **flags.model_dump(),
        apify_token=(
            settings.apify_token.get_secret_value() if settings.apify_token else None
        ),
        llm_api_key=settings.llm_key(),
    )


def _start_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True).start()


def create_app(
    settings: Settings | None = None,
    *,
    integrations_factory: IntegrationsFactory | None = None,
    submit: Submit = _start_thread,
) -> FastAPI:
    settings = settings or Settings()

    make_integrations = integrations_factory or (
        lambda flags: _live_integrations(settings, flags)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = JobStore(settings.jobs_db)
        app.state.store.fail_interrupted()
        yield

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
        job = store.create(request, flags)
        enrich_limit = _MAX_ENRICH if flags.live_kols else None
        submit(lambda: run_job(store, job.id, integrations, enrich_limit=enrich_limit))
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
            for event in await run_in_threadpool(store.events_after, job.id, cursor):
                yield ServerSentEvent(id=str(event.seq), event=event.type, data=event)
                cursor = event.seq
                if event.type in TERMINAL_EVENTS:
                    return
            if await request.is_disconnected():
                return
            await asyncio.sleep(_POLL_SECONDS)

    return app


app = create_app()
