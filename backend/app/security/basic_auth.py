import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..settings import get_backend_settings
from .rate_limit import get_client_ip

logger = logging.getLogger("psychstrata.api")
basic_auth = HTTPBasic(auto_error=False)
DEFAULT_CLINICIAN_ID = "default-clinician"


def get_basic_auth_credentials() -> tuple[str, str] | None:
    settings = get_backend_settings()
    username = settings.backend_basic_auth_username
    password = settings.backend_basic_auth_password
    if username is None and password is None:
        return None
    if username is None or password is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Backend Basic Auth is misconfigured. "
                "Set both BACKEND_BASIC_AUTH_USERNAME and BACKEND_BASIC_AUTH_PASSWORD."
            ),
        )
    return username, password


def _unauthorized_exception() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid credentials.",
    )


def get_current_clinician_id(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials | None, Depends(basic_auth)],
) -> str:
    configured_credentials = get_basic_auth_credentials()
    if configured_credentials is None:
        return DEFAULT_CLINICIAN_ID
    if credentials is None:
        logger.warning("Auth failure (no credentials) from %s", get_client_ip(request))
        raise _unauthorized_exception()

    configured_username, configured_password = configured_credentials
    is_username_valid = secrets.compare_digest(credentials.username, configured_username)
    is_password_valid = secrets.compare_digest(credentials.password, configured_password)
    if not (is_username_valid and is_password_valid):
        logger.warning("Auth failure (invalid credentials) from %s", get_client_ip(request))
        raise _unauthorized_exception()
    return configured_username


def require_basic_auth(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials | None, Depends(basic_auth)],
) -> None:
    get_current_clinician_id(request, credentials)
