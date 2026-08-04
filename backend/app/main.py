import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .routes.auth import router as auth_router
from .routes.features import router as features_router
from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .routes.tsne import router as tsne_router
from .security.basic_auth import BASIC_AUTH_PASSWORD_ENV, BASIC_AUTH_USERNAME_ENV
from .security.rate_limit import get_client_ip, limiter


DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://localhost:5173", "http://localhost:5174")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("psychstrata.api")


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS")
    if raw_origins is None:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="PsychStrata Dashboard API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s %s failed after %.1fms",
            get_client_ip(request),
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %s %s %.1fms",
        get_client_ip(request),
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(features_router)
app.include_router(predict_router)
app.include_router(tsne_router)
