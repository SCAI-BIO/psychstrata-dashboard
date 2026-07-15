# Treatment Resistance Classifier Backend

FastAPI backend for the Treatment Resistance Classifier demo.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- FastAPI application entry point
- Health check endpoint
- Small JSON endpoint for the React frontend
- pytest test coverage
- uv-based dependency management
- Docker image for production-like runs

## Setup

```bash
uv sync
```

## Running Locally

Development server:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the API at `http://localhost:8000`.

Optional Basic Auth (recommended for deployed environments):

```bash
export BACKEND_BASIC_AUTH_USERNAME=dashboard-user
export BACKEND_BASIC_AUTH_PASSWORD=change-me
```

Docker:

```bash
docker build -t psychstrata-dashboard-backend .
docker run \
  -e BACKEND_BASIC_AUTH_USERNAME=dashboard-user \
  -e BACKEND_BASIC_AUTH_PASSWORD=change-me \
  -p 8000:8000 psychstrata-dashboard-backend
```

## REST API

The backend exposes a small JSON API.

Public endpoints:

- `GET /api/health` - basic health check
- `GET /api/auth/status` - whether backend Basic Auth is enabled
- `POST /api/auth/login` - credential check endpoint for the frontend login flow

Protected endpoints (require Basic Auth when configured):

- `GET /api/summary` - demo content for the frontend
- `GET /api/features`
- `POST /api/predict`
- `POST /api/explain`
- `GET /api/tsne`

Example request:

```bash
curl -u dashboard-user:change-me http://localhost:8000/api/summary
```

## Authentication Configuration

Basic Auth is controlled entirely in the backend:

- `BACKEND_BASIC_AUTH_USERNAME`
- `BACKEND_BASIC_AUTH_PASSWORD`

If both variables are omitted (or empty), authentication is disabled.
If only one variable is set, the API returns a misconfiguration error so deployment issues are visible.

## Tests

```bash
uv run pytest
```
