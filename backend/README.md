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

Docker:

```bash
docker build -t psychstrata-dashboard-backend .
docker run -p 8000:8000 psychstrata-dashboard-backend
```

## REST API

The backend exposes a small JSON API:

- `GET /api/health` - basic health check
- `GET /api/summary` - demo content for the frontend

Example request:

```bash
curl http://localhost:8000/api/summary
```

## Tests

```bash
uv run pytest
```
