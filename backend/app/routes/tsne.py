from fastapi import APIRouter, Depends, Request

from ..schemas.prediction_schema import TsneResponse
from ..security.basic_auth import require_basic_auth
from ..security.rate_limit import limiter
from ..services.prediction_service import get_tsne_response

router = APIRouter()


@router.get("/api/tsne", dependencies=[Depends(require_basic_auth)])
@limiter.limit("30/minute")
def tsne(request: Request) -> TsneResponse:
    return get_tsne_response()
