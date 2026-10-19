# Psych-STRATA Dashboard 
<a href="https://doi.org/10.5281/zenodo.22922039"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22922039-blue.svg" alt="DOI"></a>
[![CI](https://github.com/SCAI-BIO/psychstrata-dashboard/actions/workflows/ci.yaml/badge.svg)](https://github.com/SCAI-BIO/psychstrata-dashboard/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/SCAI-BIO/psychstrata-dashboard/branch/main/graph/badge.svg)](https://codecov.io/gh/SCAI-BIO/psychstrata-dashboard)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)



Interactive dashboard enabling shared decision making for treatment resistance in depression.

https://psych-strata.eu/

## Disclaimer 

This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Project Structure

```text
├── backend/          
├── frontend/         
├── e2e/              # Playwright system tests against the compose stack
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
the host: nginx (production) or Vite (development) proxies `/api` requests to
the backend service over the internal Docker network.

Full stack, development with live reload:

```bash
docker compose -f compose.yml -f compose.dev.yml up --build
```

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

System (Playwright end-to-end against the full compose stack):

```bash
cd e2e
pnpm install
npx playwright install --with-deps chromium
pnpm test
```

The system tests expect the stack to be running with Basic Auth enabled, e.g.:

```bash
BACKEND_BASIC_AUTH_USERNAME=e2e-user BACKEND_BASIC_AUTH_PASSWORD=e2e-pass docker compose up --build
E2E_BASE_URL=http://localhost:3000 pnpm test   # from the e2e directory
```

Override credentials with `E2E_AUTH_USERNAME` / `E2E_AUTH_PASSWORD`. The
Playwright HTML report is written to `e2e/playwright-report/`.

In CI (`.github/workflows/ci.yaml`) the system tests run in the same workflow
as the unit tests, but only once both the backend and frontend unit-test
suites are green — the job depends on them via `needs`.

## API

The backend exposes a REST API under `/api` (health, auth, features, patients,
treatment plans, prediction, explainability, t-SNE). See
[backend/README.md](backend/README.md#rest-api) for endpoints, configuration,
and database behavior.

## License & Citation

This project is licensed under the [Apache License 2.0](LICENSE). Redistributions must preserve the
attribution in [NOTICE](NOTICE), as required by the license.

If you use this software in research, please cite it using the metadata in [CITATION.cff](CITATION.cff).
All versions are archived on Zenodo under the concept DOI
[10.5281/zenodo.22922039](https://doi.org/10.5281/zenodo.22922039).

Questions, feedback, and collaboration inquiries are welcome — reach out via the
[Psych-STRATA project website](https://psych-strata.eu/contact-us/).
