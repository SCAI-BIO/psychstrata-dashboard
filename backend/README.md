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
- `GET /api/version` - deployed application release version
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

Configured clinical, medication, and adherence values are stored as JSON objects and validated against the feature configuration. Clinical data other than genetics is fixed when the patient is created; patient updates only change the patient's name, while `PATCH /api/patients/{id}/genetics` updates the genetics object. Age is derived from date of birth when a persisted treatment plan is submitted for prediction. The current version stores only the latest assessment and treatment-plan state.

The current version supports one clinician. When Basic Auth is enabled, the configured username is stored as `clinician_id`; when auth is disabled for local development, records use `default-clinician`. Patient deletion cascades through clinical data and treatment plans.

At startup, the backend idempotently creates two demo patients for the configured clinician, each with three treatment plans: `Max Mustermann` (CYP2C19 `*1/*2`, CYP2D6 `*1/*1`) and `Erika Mustermann` (CYP2C19 `*2/*2`, a poor metabolizer that yields a reduced-dose sertraline recommendation; CYP2D6 `*1/*4`). Demo genetics are derived from the active genetics configuration, and existing valid demo genetics are preserved across restarts. Stable IDs prevent duplicate demo records.

This development-stage schema does not include migrations. If a local `db.sqlite3` was created by the previous SQLModel implementation, recreate that database before starting this version.

## Model Configuration

- `MODEL_ARTIFACT_PATH` (optional path to a pickled model artifact on disk)
- `LLM_PROVIDER_URL` (optional LLM endpoint URL; defaults to FHGenie at `https://fhgenie.fraunhofer.de/v1`)
- `LLM_PROVIDER_MODEL` (optional model name; defaults to `Qwen/Qwen3-Coder-Next-FP8`)
- `LLM_PROVIDER_KEY` (optional provider API key; required for hosted providers such as FHGenie)

If `MODEL_ARTIFACT_PATH` is not set, the backend uses the existing synthetic model.
You can also set it in `BACKEND_CONFIG_FILE`.

The deployed release version is configured with `APP_VERSION`. Release Docker
builds set it from the GitHub release tag; local runs use `development` when it
is not set.

LLM explanations select a client strategy from the configured URL. FHGenie URLs
use the FHGenie Chat Completions client; OpenAI URLs use the OpenAI Responses API
client; localhost URLs use the Ollama client and default to `/api/chat`. The
secret is optional for local Ollama use. `LLM_CLIENT_URL`, `LLM_CLIENT_MODEL`,
`LLM_CLIENT_SECRET`, `OPENAI_API_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`
remain supported as compatibility fallbacks but are deprecated and will be
removed in a future release.

## Feature Configuration

- `FEATURES_CONFIG_PATH` (optional path to a grouped JSON feature file)

If `FEATURES_CONFIG_PATH` is not set, the backend uses the built-in feature definitions from `app/defaults/feature_definitions.json`.
You can also set it in `BACKEND_CONFIG_FILE`.

The configuration contains `clinical`, `medications`, `adherence`, and an explicit `model_feature_order`. Each feature uses `dtype` to distinguish numeric and categorical values. The merged feature API remains available to the frontend, while `feature_groups` exposes the domain grouping. The default `sex_at_birth` feature supports Male, Female, and Other; synthetic data uses a 49%/49%/2% distribution.

## Genetics Configuration

- `GENETICS_CONFIG_PATH` (optional path to a genetics JSON file)

If `GENETICS_CONFIG_PATH` is not set, the backend uses the built-in gene and drug definitions from `app/defaults/genetics_definitions.json`.
You can also set it in `BACKEND_CONFIG_FILE`.

```json
{
  "genes": [
    { "id": "CYP2C19", "label": "CYP2C19", "common_alleles": ["*1", "*2", "*3"] }
  ],
  "drugs": [
    { "id": "sertraline", "label": "Sertraline" }
  ]
}
```

Configured genes drive the intake form, patient genetics validation, and interpretation. Gene ids and common alleles must be supported by pypgx; the default configured drugs are covered by bundled CPIC recommendations. The configuration is validated during application startup, so invalid entries fail fast with a clear error.

Recommendation text is reproduced verbatim from the bundled CPIC snapshot in `app/defaults/cpic_recommendations.json`. Each recommendation includes its CPIC id and classification, and the API response includes the CPIC source URL and retrieval date. CPIC guidance describes recommended starting doses at therapy initiation; it is not advice to adjust an existing dose, and the backend does not call CPIC at runtime. To refresh the snapshot, fetch the recommendations for the configured drugs from the CPIC API and update the snapshot's `recommendations` and `retrieved_at` fields. Drugs from a custom configuration that are absent from the snapshot are reported as having no bundled CPIC recommendation.

### Licensing and attribution

The PGx feature combines two data paths, both documented in [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md):

- CPIC content (CC0 1.0) — the verbatim recommendations in the snapshot, plus CPIC gene/drug levels, guideline links, and PMIDs from `pypgx.load_cpic_table()`. CPIC asks to be cited as `CPIC®. URL [date accessed]`; the API exposes `recommendation_source_url` and `recommendation_retrieved_at` for that purpose. The CPIC API does not publish a dataset version, so the retrieval date identifies the snapshot. CPIC levels and PMIDs come from the CPIC table bundled with the pinned PyPGx version (current as of 2023-08-19); upgrade PyPGx to refresh them. Do not reproduce the CPIC logo, and do not use the acronym in promotional material without permission from NIH.
- PyPGx (MIT) — star-allele validation, activity scores, and metaboliser phenotypes. Keep its copyright notice when redistributing (see `THIRD_PARTY_LICENSES.md`) and cite Lee et al., 2022 when publishing analyses. PyPGx also ships PharmGKB- and PharmVar-derived tables (CC BY-SA 4.0) that are redistributed unmodified; their values are not displayed by this application.

PyPGx's vendored `beagle.22Jul22.46e.jar` is GPL-3.0-or-later and is not used here. `Dockerfile` deletes it, together with the `uv` cache that contains it, from the built image — keep both removals in place to avoid redistributing a GPL binary.

The feature contains no DPWG recommendations and no safety-code.org / PREPARE safety-card code or assets.

## Tests

```bash
uv run pytest
```
