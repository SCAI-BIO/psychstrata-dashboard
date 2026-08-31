from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..security.rate_limit import limiter

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get("/api/health", response_model=HealthResponse)
@limiter.limit("60/minute")
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok")
