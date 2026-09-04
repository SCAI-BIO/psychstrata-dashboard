import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .persistence.database import init_db
from .routes.patient_records import router as patients_router
from .routes.auth import router as auth_router
from .routes.features import router as features_router
from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .routes.tsne import router as tsne_router
from .security.rate_limit import get_client_ip, limiter
from .settings import get_backend_settings

_settings = get_backend_settings()
DEFAULT_CORS_ORIGINS = tuple(_settings.backend_cors_origins)

logging.basicConfig(
    level=_settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("psychstrata.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="PsychStrata Dashboard API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(DEFAULT_CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "PATCH", "POST", "OPTIONS"],
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
app.include_router(patients_router)
app.include_router(predict_router)
app.include_router(tsne_router)
