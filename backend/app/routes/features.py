from typing import Any

from fastapi import APIRouter, Depends, Request

from ..security.basic_auth import require_basic_auth
from ..security.rate_limit import limiter
from ..services.features import get_features_response

router = APIRouter()


@router.get("/api/features", dependencies=[Depends(require_basic_auth)])
@limiter.limit("60/minute")
def features(request: Request) -> dict[str, Any]:
    return get_features_response()
