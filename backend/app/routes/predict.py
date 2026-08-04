from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from ..main import limiter
from ..services.auth import require_basic_auth
from ..services.features import parse_prediction_payload
from ..services.prediction import build_explanation_response, build_prediction_response

router = APIRouter()


@router.post("/api/predict", dependencies=[Depends(require_basic_auth)])
@limiter.limit("30/minute")
def predict(request: Request, payload: Any = Body(...)) -> dict[str, Any]:
    try:
        values_dict, confidence_level = parse_prediction_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_prediction_response(values_dict, confidence_level)


@router.post("/api/explain", dependencies=[Depends(require_basic_auth)])
@limiter.limit("10/minute; 100/day")
def explain(request: Request, payload: Any = Body(...)) -> dict[str, Any]:
    try:
        values_dict, confidence_level = parse_prediction_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_explanation_response(values_dict, confidence_level)
