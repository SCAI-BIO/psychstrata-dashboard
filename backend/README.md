# Treatment Resistance Classifier Backend

FastAPI backend for the Treatment Resistance Classifier demo.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- FastAPI application entry point
- Health check endpoint
- Patient and treatment-plan persistence endpoints
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

Optional database URL:

```bash
export BACKEND_DATABASE_URL=sqlite:///./db.sqlite3
```

If omitted, the backend uses a local SQLite database at `./db.sqlite3`.

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

- `GET /api/features`
- `POST /api/patients`
- `GET /api/patients`
- `GET /api/patients/{patient_id}`
- `PATCH /api/patients/{patient_id}`
- `DELETE /api/patients/{patient_id}`
- `POST /api/patients/{patient_id}/treatment-plans`
- `GET /api/patients/{patient_id}/treatment-plans`
- `GET /api/treatment-plans/{treatment_plan_id}`
- `PATCH /api/treatment-plans/{treatment_plan_id}`
- `DELETE /api/treatment-plans/{treatment_plan_id}`
- `POST /api/treatment-plans/{treatment_plan_id}/predict`
- `POST /api/predict`
- `POST /api/explain`
- `GET /api/tsne`

Example request:

```bash
curl -u dashboard-user:change-me http://localhost:8000/api/patients
```

## Authentication Configuration

Basic Auth is controlled entirely in the backend:

- `BACKEND_CONFIG_FILE` (optional path to a TOML file with shared backend settings)
- `BACKEND_BASIC_AUTH_USERNAME`
- `BACKEND_BASIC_AUTH_PASSWORD`

If both variables are omitted (or empty), authentication is disabled.
If only one variable is set, the API returns a misconfiguration error so deployment issues are visible.

Example `backend.toml`:

```toml
log_level = "INFO"
backend_cors_origins = ["http://localhost:3000"]
backend_basic_auth_username = "dashboard-user"
backend_basic_auth_password = "change-me"
model_artifact_path = "/models/model.pkl"
features_config_path = "/configs/features.json"
backend_database_url = "sqlite:///./db.sqlite3"
```

## Database Configuration

- `BACKEND_DATABASE_URL` (optional SQLAlchemy database URL)

The default is `sqlite:///./db.sqlite3`, which creates a local SQLite file in the backend working directory. Docker Compose uses `sqlite:////data/db.sqlite3` with a named volume so patient and treatment-plan data persist across container restarts.

Plain SQLAlchemy 2.0 maps three entities:

- `Patient` and its one-to-one `PatientClinicalData`
- `TreatmentPlan`, including medication and adherence values

Configured clinical, medication, and adherence values are stored as JSON objects and validated against the feature configuration. Clinical data is fixed when the patient is created; patient updates only change the patient's name. Age is derived from date of birth when a persisted treatment plan is submitted for prediction. The current version stores only the latest assessment and treatment-plan state.

The current version supports one clinician. When Basic Auth is enabled, the configured username is stored as `clinician_id`; when auth is disabled for local development, records use `default-clinician`. Patient deletion cascades through clinical data and treatment plans.

This development-stage schema does not include migrations. If a local `db.sqlite3` was created by the previous SQLModel implementation, recreate that database before starting this version.

## Model Configuration

- `MODEL_ARTIFACT_PATH` (optional path to a pickled model artifact on disk)

If `MODEL_ARTIFACT_PATH` is not set, the backend uses the existing synthetic model.
You can also set it in `BACKEND_CONFIG_FILE`.

## Feature Configuration

- `FEATURES_CONFIG_PATH` (optional path to a grouped JSON feature file)

If `FEATURES_CONFIG_PATH` is not set, the backend uses the built-in feature definitions from `app/defaults/feature_definitions.json`.
You can also set it in `BACKEND_CONFIG_FILE`.

The configuration contains `clinical`, `medications`, `adherence`, and an explicit `model_feature_order`. Each feature uses `dtype` to distinguish numeric and categorical values. The merged feature API remains available to the frontend, while `feature_groups` exposes the domain grouping. The default `sex_at_birth` feature supports Male, Female, and Other; synthetic data uses a 49%/49%/2% distribution.

## Tests

```bash
uv run pytest
```
