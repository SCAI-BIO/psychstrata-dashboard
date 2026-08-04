from typing import Any

from fastapi import APIRouter, Depends, Request

from ..main import limiter
from ..services.auth import require_basic_auth
from ..services.prediction import get_tsne_response

router = APIRouter()


@router.get("/api/tsne", dependencies=[Depends(require_basic_auth)])
@limiter.limit("30/minute")
def tsne(request: Request) -> dict[str, Any]:
    return get_tsne_response()
