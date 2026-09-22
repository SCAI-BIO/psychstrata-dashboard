import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .persistence.database import init_db
from .persistence.demo_data import seed_demo_data
from .routes.patients import router as patients_router
from .routes.auth import router as auth_router
from .routes.features import router as features_router
from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .routes.tsne import router as tsne_router
from .routes.version import router as version_router
from .security.rate_limit import get_client_ip, limiter
from .services import errors
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
    seed_demo_data()
    yield


app = FastAPI(title="PsychStrata Dashboard API", version=_settings.app_version, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _service_validation_error_response(
    _request: Request,
    exception: errors.InvalidPatientDataError | errors.MissingModelFeaturesError,
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exception)})


def _service_not_found_error_response(
    _request: Request,
    exception: errors.PatientNotFoundError | errors.TreatmentPlanNotFoundError,
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exception)})


app.add_exception_handler(errors.InvalidPatientDataError, _service_validation_error_response)
app.add_exception_handler(errors.MissingModelFeaturesError, _service_validation_error_response)
app.add_exception_handler(errors.PatientNotFoundError, _service_not_found_error_response)
app.add_exception_handler(errors.TreatmentPlanNotFoundError, _service_not_found_error_response)
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
app.include_router(version_router)
