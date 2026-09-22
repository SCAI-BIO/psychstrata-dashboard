from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..security.rate_limit import limiter
from ..settings import get_backend_settings

router = APIRouter()


class VersionResponse(BaseModel):
    version: str


@router.get("/api/version", response_model=VersionResponse)
@limiter.limit("60/minute")
def version(request: Request) -> VersionResponse:
    return VersionResponse(version=get_backend_settings().app_version)
