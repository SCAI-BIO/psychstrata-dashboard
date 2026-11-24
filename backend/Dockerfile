FROM python:3.12-slim AS base

# fallback for if build fails (should not happen)
ARG APP_VERSION=development

COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    APP_VERSION="${APP_VERSION}" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
# PyPGx bundles beagle.22Jul22.46e.jar (GPL-3.0-or-later). This project does not
# use PyPGx phasing/imputation pipelines, so the JAR and the uv cache that holds
# it are removed to avoid redistributing a GPL-licensed binary. See
# ../THIRD_PARTY_LICENSES.md.
RUN uv sync --frozen --no-dev \
    && rm -f .venv/lib/python*/site-packages/pypgx/api/*.jar \
    && uv cache clean

COPY app ./app

EXPOSE 8000

LABEL org.opencontainers.image.title="Psych-STRATA Dashboard backend" \
      org.opencontainers.image.description="FastAPI backend for the Psych-STRATA Dashboard (demo, synthetic data only)" \
      org.opencontainers.image.source="https://github.com/SCAI-BIO/psychstrata-dashboard" \
      org.opencontainers.image.licenses="Apache-2.0"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dev

# Re-running uv sync restores the vendored Beagle JAR; remove it again.
RUN uv sync --frozen \
    && rm -f .venv/lib/python*/site-packages/pypgx/api/*.jar \
    && uv cache clean
COPY tests ./tests

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
