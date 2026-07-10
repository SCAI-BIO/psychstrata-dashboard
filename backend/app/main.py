import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://localhost:5173")


class HealthResponse(BaseModel):
    status: str


class SummaryResponse(BaseModel):
    title: str
    message: str
    disclaimer: str


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS")
    if raw_origins is None:
        return list(DEFAULT_CORS_ORIGINS)

    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="PsychStrata Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/summary", response_model=SummaryResponse)
def summary() -> SummaryResponse:
    return SummaryResponse(
        title="Treatment Resistance Classifier Demo",
        message="The React frontend is connected to the FastAPI backend.",
        disclaimer=(
            "This demo uses synthetic data for illustration purposes only. "
            "It is not a medical device and must not be used for clinical decisions."
        ),
    )
