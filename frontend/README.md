# Treatment Resistance Classifier Frontend

React frontend for the Treatment Resistance Classifier demo.

**Disclaimer:** This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions.

## Features

- Vite React application with TypeScript
- Environment-based backend API URL
- Basic page that calls the FastAPI backend
- Vitest test coverage
- pnpm-based dependency management
- Docker image served by nginx

## Setup

```bash
corepack enable
pnpm install
```

## Running Locally

Development server:

```bash
VITE_API_BASE_URL=http://localhost:8000 pnpm dev
```

Open `http://localhost:5173`.

Docker:

```bash
docker build \
  --build-arg VITE_API_BASE_URL=http://localhost:8000 \
  -t psychstrata-dashboard-frontend .
docker run -p 3000:80 psychstrata-dashboard-frontend
```

Access at `http://localhost:3000`.

## Configuration

Set `VITE_API_BASE_URL` to the backend origin the browser should call. For local development, use `http://localhost:8000`.

## Tests

```bash
pnpm test
```
