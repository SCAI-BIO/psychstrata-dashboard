from fastapi import APIRouter, Depends, Request

from ..schemas.pgx_schema import (
    DiplotypeInput,
    PgxInterpretRequest,
    PgxInterpretResponse,
    PgxOptionsResponse,
)
from ..security.basic_auth import require_basic_auth
from ..security.rate_limit import limiter
from ..services import pgx_service

router = APIRouter()


def _diplotype(diplotype: DiplotypeInput) -> tuple[str, str]:
    return diplotype.allele1.strip(), diplotype.allele2.strip()


@router.get(
    "/api/pgx/options",
    response_model=PgxOptionsResponse,
    dependencies=[Depends(require_basic_auth)],
)
@limiter.limit("120/minute")
def pgx_options(request: Request) -> PgxOptionsResponse:
    return PgxOptionsResponse(**pgx_service.build_options())


@router.post(
    "/api/pgx/interpret",
    response_model=PgxInterpretResponse,
    dependencies=[Depends(require_basic_auth)],
)
@limiter.limit("60/minute")
def interpret_pgx(request: Request, payload: PgxInterpretRequest) -> PgxInterpretResponse:
    diplotypes: dict[str, tuple[str, str]] = {}
    if payload.cyp2c19 is not None:
        diplotypes["CYP2C19"] = _diplotype(payload.cyp2c19)
    if payload.cyp2d6 is not None:
        diplotypes["CYP2D6"] = _diplotype(payload.cyp2d6)

    return PgxInterpretResponse(**pgx_service.interpret(diplotypes, payload.drugs))
