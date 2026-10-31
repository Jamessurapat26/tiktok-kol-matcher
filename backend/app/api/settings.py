import importlib.util
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ..integrations.embeddings_live import DEFAULT_EMBED_MODEL
from ..integrations.llm_live import DEFAULT_BASE_URL, DEFAULT_MODEL

Mode = Literal["demo", "live"]


class LiveFlags(BaseModel):
    """Which integrations a job runs live. Everything else stays on mocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    live_web: bool = False
    live_fb: bool = False
    live_kols: bool = False
    live_llm: bool = False
    live_embed: bool = False
    live_search: bool = False


class FlagUnavailable(ValueError):
    """A live flag was requested but its credential or dependency is missing."""


_LLM_KEY_VARS = "LLM_API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY"


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


class Settings(BaseSettings):
    """Service configuration from the environment.

    The integrations read their own variables when they are built. The model
    fields here read the same names only to report them in /api/health.
    """

    model_config = SettingsConfigDict(extra="ignore")

    demo_mode: bool | None = None
    apify_token: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    nvidia_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    jina_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    embed_model: str | None = None
    jobs_db: Path = Path("data/jobs.db")
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    job_workers: int = Field(default=2, ge=1)
    max_queued_jobs: int = Field(default=20, ge=1)

    @field_validator("demo_mode", "llm_base_url", "llm_model", "embed_model", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def mode(self) -> Mode:
        if self.demo_mode is not None:
            return "demo" if self.demo_mode else "live"
        return "live" if _is_set(self.apify_token) else "demo"

    def llm_key(self) -> str | None:
        for secret in (self.llm_api_key, self.nvidia_api_key, self.openai_api_key):
            if _is_set(secret):
                return secret.get_secret_value().strip()
        return None

    def credentials(self) -> dict[str, bool]:
        return {
            "APIFY_TOKEN": _is_set(self.apify_token),
            "LLM_API_KEY": _is_set(self.llm_api_key),
            "NVIDIA_API_KEY": _is_set(self.nvidia_api_key),
            "OPENAI_API_KEY": _is_set(self.openai_api_key),
            "JINA_API_KEY": _is_set(self.jina_api_key),
        }

    def models(self) -> dict[str, str | None]:
        """What the live LLM and embedder would use. Host only, never the full URL."""
        return {
            "llm_model": self.llm_model or DEFAULT_MODEL,
            "llm_host": urlsplit(self.llm_base_url or DEFAULT_BASE_URL).hostname,
            "embed_model": self.embed_model or DEFAULT_EMBED_MODEL,
        }

    def unavailable(self) -> dict[str, str | None]:
        """Why each live flag cannot run, or None when it can."""
        no_apify = None if _is_set(self.apify_token) else "APIFY_TOKEN is not set"
        no_llm = None if self.llm_key() else f"{_LLM_KEY_VARS} is not set"
        return {
            "live_web": None,
            "live_fb": no_apify,
            "live_kols": no_apify,
            "live_llm": no_llm,
            "live_search": (
                no_llm or (None if _is_set(self.jina_api_key) else "JINA_API_KEY is not set")
            ),
            # find_spec only, so the model is neither imported nor loaded here.
            "live_embed": (
                None
                if importlib.util.find_spec("sentence_transformers") is not None
                else "sentence-transformers is not installed "
                "(run `uv sync --extra embed`)"
            ),
        }

    def default_flags(self) -> LiveFlags:
        if self.mode == "demo":
            return LiveFlags()
        return LiveFlags(**{flag: why is None for flag, why in self.unavailable().items()})

    def resolve_flags(self, requested: dict[str, bool | None]) -> LiveFlags:
        """Effective flags for a job. Omitted (None) flags take the default.

        Demo mode forces every flag off. In live mode an explicit true flag that
        cannot run raises ``FlagUnavailable``.
        """
        if self.mode == "demo":
            return LiveFlags()
        unavailable = self.unavailable()
        problems = [
            f"{flag} is unavailable: {unavailable[flag]}"
            for flag, wanted in requested.items()
            if wanted and unavailable[flag]
        ]
        if problems:
            raise FlagUnavailable("; ".join(problems))
        defaults = self.default_flags()
        return defaults.model_copy(
            update={flag: wanted for flag, wanted in requested.items() if wanted is not None}
        )
