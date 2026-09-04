from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from ..persistence.database import get_session
from ..domain.patient_schemas import PatientCreate, PatientRead, PatientUpdate, TreatmentPlanCreate, TreatmentPlanRead, TreatmentPlanUpdate
from ..persistence.patient_repository import PatientRepository
from ..security.basic_auth import get_current_clinician_id
from ..services.patient_service import PatientService
from ..services.prediction import build_prediction_response

router = APIRouter()


def get_patient_service(
    session: Annotated[Session, Depends(get_session)],
) -> PatientService:
    return PatientService(PatientRepository(session))


@router.post("/api/patients", response_model=PatientRead, status_code=201)
def create_patient(
    payload: PatientCreate,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> PatientRead:
    return service.create_patient(clinician_id, payload)


@router.get("/api/patients", response_model=list[PatientRead])
def list_patients(
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> list[PatientRead]:
    return service.list_patients(clinician_id)


@router.get("/api/patients/{patient_id}", response_model=PatientRead)
def get_patient(
    patient_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> PatientRead:
    return service.get_patient(clinician_id, patient_id)


@router.patch("/api/patients/{patient_id}", response_model=PatientRead)
def update_patient(
    patient_id: str,
    payload: PatientUpdate,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> PatientRead:
    return service.update_patient(clinician_id, patient_id, payload)


@router.delete("/api/patients/{patient_id}", status_code=204)
def delete_patient(
    patient_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> Response:
    service.delete_patient(clinician_id, patient_id)
    return Response(status_code=204)


@router.post("/api/patients/{patient_id}/treatment-plans", response_model=TreatmentPlanRead, status_code=201)
def create_treatment_plan(
    patient_id: str,
    payload: TreatmentPlanCreate,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> TreatmentPlanRead:
    return service.create_treatment_plan(clinician_id, patient_id, payload)


@router.get("/api/patients/{patient_id}/treatment-plans", response_model=list[TreatmentPlanRead])
def list_treatment_plans(
    patient_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> list[TreatmentPlanRead]:
    return service.list_treatment_plans(clinician_id, patient_id)


@router.get("/api/treatment-plans/{treatment_plan_id}", response_model=TreatmentPlanRead)
def get_treatment_plan(
    treatment_plan_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> TreatmentPlanRead:
    return service.get_treatment_plan(clinician_id, treatment_plan_id)


@router.patch("/api/treatment-plans/{treatment_plan_id}", response_model=TreatmentPlanRead)
def update_treatment_plan(
    treatment_plan_id: str,
    payload: TreatmentPlanUpdate,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> TreatmentPlanRead:
    return service.update_treatment_plan(clinician_id, treatment_plan_id, payload)


@router.delete("/api/treatment-plans/{treatment_plan_id}", status_code=204)
def delete_treatment_plan(
    treatment_plan_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
) -> Response:
    service.delete_treatment_plan(clinician_id, treatment_plan_id)
    return Response(status_code=204)


@router.post("/api/treatment-plans/{treatment_plan_id}/predict")
def predict_treatment_plan(
    treatment_plan_id: str,
    clinician_id: Annotated[str, Depends(get_current_clinician_id)],
    service: Annotated[PatientService, Depends(get_patient_service)],
):
    treatment_plan = service.get_treatment_plan(clinician_id, treatment_plan_id)
    return build_prediction_response(service.build_model_features(treatment_plan), 95)
