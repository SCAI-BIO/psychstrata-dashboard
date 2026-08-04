from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from ..security.basic_auth import get_basic_auth_credentials, require_basic_auth
from ..security.rate_limit import limiter

router = APIRouter()


class AuthStatusResponse(BaseModel):
    auth_enabled: bool


@router.get("/api/auth/status", response_model=AuthStatusResponse)
@limiter.limit("60/minute")
def auth_status(request: Request) -> AuthStatusResponse:
    return AuthStatusResponse(auth_enabled=get_basic_auth_credentials() is not None)


@router.post("/api/auth/login", dependencies=[Depends(require_basic_auth)])
@limiter.limit("10/minute")
def auth_login(request: Request) -> dict[str, str]:
    return {"status": "ok"}
