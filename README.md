# Treatment Resistance Classifier Demo

Interactive dashboard scaffold for predicting treatment resistance in depression using a React frontend and FastAPI backend.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- React + Vite frontend with TypeScript
- FastAPI backend with health and JSON API endpoints
- Environment-based frontend API configuration
- Dockerfiles for frontend and backend services
- Docker Compose for production-like local testing
- Docker Compose dev override for live reload
- App-specific tests and GitHub Actions workflows

## Project Structure

```text
├── backend/          # FastAPI backend, uv dependencies, pytest tests, Dockerfile
├── frontend/         # React frontend, pnpm dependencies, Vitest tests, Dockerfile
├── legacy-dash/      # Previous Dash/Flask dashboard preserved for reference
├── compose.yml       # Production-like local stack
├── compose.dev.yml   # Development override with live reload
└── .github/workflows
```

## Setup

Backend:

```bash
cd backend
uv sync
```

Frontend:

```bash
cd frontend
corepack enable
pnpm install
```

## Running Locally

Backend:

```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 pnpm dev
```

Open `http://localhost:5173`.

Full stack, production-like:

```bash
docker compose up --build
```

Access the frontend at `http://localhost:3000` and the backend at `http://localhost:8000`.

Full stack, development with live reload:

```bash
docker compose -f compose.yml -f compose.dev.yml up --build
```

Access the frontend at `http://localhost:3000`.

## Tests

Backend:

```bash
cd backend
uv run pytest
```

Frontend:

```bash
cd frontend
pnpm test
```

## REST API

The backend exposes a small JSON API:

- `GET /api/health` - basic health check
- `GET /api/summary` - demo content for the frontend

Example request:

```bash
curl http://localhost:8000/api/summary
```

The previous Dash implementation has been moved to `legacy-dash/` so the repository root can act as the monorepo entry point without deleting prior work.
