from datetime import date
from typing import Any

from fastapi import HTTPException

from ..io.feature_loader import get_features_by_id, get_model_feature_order, validate_feature_values
from ..persistence.patient_repository import PatientRepository
from ..domain.patient_records import Patient, TreatmentPlan
from ..domain.patient_schemas import PatientCreate, PatientUpdate, TreatmentPlanCreate, TreatmentPlanUpdate
from ..utils.datetime import age_on_date


def validate_patient_age(date_of_birth: date) -> None:
    age = age_on_date(date_of_birth)
    age_feature = get_features_by_id()["age"]
    if age < age_feature.params["min"] or age > age_feature.params["max"]:
        raise HTTPException(
            status_code=422,
            detail=f"Derived age must be between {age_feature.params['min']} and {age_feature.params['max']}.",
        )


class PatientService:
    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def create_patient(self, clinician_id: str, payload: PatientCreate) -> Patient:
        clinical_data = payload.clinical_data
        validate_patient_age(clinical_data.date_of_birth)
        try:
            clinical_features = validate_feature_values(
                "clinical",
                clinical_data.clinical_features,
                include_defaults=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return self._repository.create_patient(
            clinician_id=clinician_id,
            patient_values=payload.model_dump(exclude={"clinical_data"}),
            clinical_values={
                **clinical_data.model_dump(exclude={"clinical_features"}),
                "clinical_features": clinical_features,
            },
        )

    def list_patients(self, clinician_id: str) -> list[Patient]:
        return self._repository.list_patients(clinician_id)

    def get_patient(self, clinician_id: str, patient_id: str) -> Patient:
        patient = self._repository.get_patient(clinician_id, patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return patient

    def update_patient(self, clinician_id: str, patient_id: str, payload: PatientUpdate) -> Patient:
        patient = self.get_patient(clinician_id, patient_id)
        return self._repository.update_patient(patient, payload.model_dump(exclude_unset=True))

    def delete_patient(self, clinician_id: str, patient_id: str) -> None:
        self._repository.delete_patient(self.get_patient(clinician_id, patient_id))

    def create_treatment_plan(
        self,
        clinician_id: str,
        patient_id: str,
        payload: TreatmentPlanCreate,
    ) -> TreatmentPlan:
        patient = self.get_patient(clinician_id, patient_id)
        try:
            medications = validate_feature_values("medications", payload.medications, include_defaults=True)
            adherence = validate_feature_values(
                "adherence",
                payload.adherence,
                include_defaults=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return self._repository.create_treatment_plan(
            clinician_id=clinician_id,
            patient=patient,
            plan_values={
                **payload.model_dump(exclude={"medications", "adherence"}),
                "medications": medications,
                "adherence": adherence,
            },
        )

    def list_treatment_plans(self, clinician_id: str, patient_id: str) -> list[TreatmentPlan]:
        self.get_patient(clinician_id, patient_id)
        return self._repository.list_treatment_plans(clinician_id, patient_id)

    def get_treatment_plan(self, clinician_id: str, treatment_plan_id: str) -> TreatmentPlan:
        treatment_plan = self._repository.get_treatment_plan(clinician_id, treatment_plan_id)
        if treatment_plan is None:
            raise HTTPException(status_code=404, detail="Treatment plan not found.")
        return treatment_plan

    def update_treatment_plan(
        self,
        clinician_id: str,
        treatment_plan_id: str,
        payload: TreatmentPlanUpdate,
    ) -> TreatmentPlan:
        treatment_plan = self.get_treatment_plan(clinician_id, treatment_plan_id)
        plan_updates = payload.model_dump(exclude={"medications", "adherence"}, exclude_unset=True)
        try:
            if payload.medications is not None:
                plan_updates["medications"] = {
                    **treatment_plan.medications,
                    **validate_feature_values("medications", payload.medications),
                }
            if payload.adherence is not None:
                plan_updates["adherence"] = {
                    **treatment_plan.adherence,
                    **validate_feature_values("adherence", payload.adherence),
                }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return self._repository.update_treatment_plan(treatment_plan, plan_updates)

    def delete_treatment_plan(self, clinician_id: str, treatment_plan_id: str) -> None:
        self._repository.delete_treatment_plan(self.get_treatment_plan(clinician_id, treatment_plan_id))

    def build_model_features(self, treatment_plan: TreatmentPlan) -> dict[str, Any]:
        patient = treatment_plan.patient
        combined = {
            **patient.clinical_data.clinical_features,
            **treatment_plan.medications,
            **treatment_plan.adherence,
            "age": age_on_date(patient.clinical_data.date_of_birth),
        }
        order = get_model_feature_order()
        missing = [feature_id for feature_id in order if feature_id not in combined]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing persisted model features: {', '.join(missing)}.")
        return {feature_id: combined[feature_id] for feature_id in order}
