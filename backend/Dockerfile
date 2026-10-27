# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

ARG INSTALL_EMBED=false
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    $([ "$INSTALL_EMBED" = "true" ] && echo --extra embed)

FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY app ./app
# A named volume mounted here starts with this ownership, so the non-root user can write.
RUN mkdir -p /app/data/cache && chown -R app:app /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    KOL_CACHE_DIR=/app/data/cache \
    JOBS_DB=/app/data/jobs.db \
    HF_HOME=/app/data/huggingface
USER app
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
