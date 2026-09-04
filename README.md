# Psych-STRATA Dashboard

Interactive dashboard enabling shared decision making for treatment resistance in depression.

https://psych-strata.eu/

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Project Structure

```text
├── backend/          
├── frontend/         
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

Authentication is configured in the backend. Set the backend env vars below to enable HTTP Basic Auth:

```bash
export BACKEND_BASIC_AUTH_USERNAME=dashboard-user
export BACKEND_BASIC_AUTH_PASSWORD=change-me
```

Open `http://localhost:5173` and sign in with the configured backend credentials (when enabled).

Full stack, production-like:

```bash
docker compose up --build
```

Access the frontend at `http://localhost:3000`. The backend is not published to
the host; the frontend reaches it over the internal Docker network, and nginx
proxies `/api` requests to the backend service.

Full stack, development with live reload:

```bash
docker compose -f compose.yml -f compose.dev.yml up --build
```

Access the frontend at `http://localhost:3000`. As above, the backend is not
exposed on the host — Vite proxies `/api` requests to the backend service
inside the Docker network.

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

Base URL: `http://localhost:8000`

- `GET /api/health` — service health check
- `GET /api/auth/status` — reports whether backend Basic Auth is enabled
- `POST /api/auth/login` — validates provided Basic Auth credentials
- `GET /api/features` — feature schema, defaults, and confidence-level bounds
- `POST /api/patients` — create a patient for the current clinician
- `GET /api/patients` — list patients for the current clinician
- `GET /api/patients/{patient_id}` — read a patient
- `PATCH /api/patients/{patient_id}` — update a patient
- `DELETE /api/patients/{patient_id}` — delete a patient and cascade-delete treatment plans
- `POST /api/patients/{patient_id}/treatment-plans` — create a treatment plan for a patient
- `GET /api/patients/{patient_id}/treatment-plans` — list a patient's treatment plans
- `GET /api/treatment-plans/{treatment_plan_id}` — read a treatment plan
- `PATCH /api/treatment-plans/{treatment_plan_id}` — update a treatment plan
- `DELETE /api/treatment-plans/{treatment_plan_id}` — delete a treatment plan
- `POST /api/treatment-plans/{treatment_plan_id}/predict` — predict from persisted patient and treatment-plan data
- `POST /api/predict` — prediction, SHAP values, top contributors, and selected t-SNE point
- `POST /api/explain` — prediction context plus generated explanation text
- `GET /api/tsne` — population t-SNE coordinates and class labels

Examples:

```bash
curl http://localhost:8000/api/features
curl http://localhost:8000/api/tsne
```

When auth is enabled, call protected endpoints with `-u <username>:<password>`.

Patient and treatment-plan persistence uses plain SQLAlchemy 2.0. Configure `BACKEND_DATABASE_URL` to choose the database; if omitted, the backend uses `sqlite:///./db.sqlite3`. Configured clinical, medication, and treatment values are stored as validated JSON objects in their aggregate tables.
