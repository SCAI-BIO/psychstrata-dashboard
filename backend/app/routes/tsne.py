from typing import Any

from fastapi import APIRouter, Depends, Request

from ..security.basic_auth import require_basic_auth
from ..security.rate_limit import limiter
from ..services.prediction import get_tsne_response

router = APIRouter()


@router.get("/api/tsne", dependencies=[Depends(require_basic_auth)])
@limiter.limit("30/minute")
def tsne(request: Request) -> dict[str, Any]:
    return get_tsne_response()
