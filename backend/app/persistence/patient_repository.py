from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..domain.patient_records import Patient, PatientClinicalData, TreatmentPlan
from ..utils.datetime import utc_now


class PatientRepository:
    def __init__(self, session: Session):
        self._session = session

    def create_patient(
        self,
        *,
        clinician_id: str,
        patient_values: dict[str, Any],
        clinical_values: dict[str, Any],
    ) -> Patient:
        patient = Patient(clinician_id=clinician_id, **patient_values)
        patient.clinical_data = PatientClinicalData(**clinical_values)
        self._session.add(patient)
        self._session.commit()
        return patient

    def list_patients(self, clinician_id: str) -> list[Patient]:
        statement = (
            select(Patient)
            .options(selectinload(Patient.clinical_data))
            .where(Patient.clinician_id == clinician_id)
            .order_by(Patient.created_at.desc())
        )
        return list(self._session.scalars(statement))

    def get_patient(self, clinician_id: str, patient_id: str) -> Patient | None:
        statement = (
            select(Patient)
            .options(selectinload(Patient.clinical_data))
            .where(Patient.clinician_id == clinician_id, Patient.id == patient_id)
        )
        return self._session.scalar(statement)

    def update_patient(
        self,
        patient: Patient,
        patient_updates: dict[str, Any],
    ) -> Patient:
        for key, value in patient_updates.items():
            setattr(patient, key, value)
        patient.updated_at = utc_now()
        self._session.commit()
        return patient

    def delete_patient(self, patient: Patient) -> None:
        self._session.delete(patient)
        self._session.commit()

    def create_treatment_plan(
        self,
        *,
        clinician_id: str,
        patient: Patient,
        plan_values: dict[str, Any],
    ) -> TreatmentPlan:
        plan = TreatmentPlan(patient=patient, clinician_id=clinician_id, **plan_values)
        self._session.add(plan)
        self._session.commit()
        return plan

    def list_treatment_plans(self, clinician_id: str, patient_id: str) -> list[TreatmentPlan]:
        statement = (
            select(TreatmentPlan)
            .where(TreatmentPlan.clinician_id == clinician_id, TreatmentPlan.patient_id == patient_id)
            .order_by(TreatmentPlan.created_at.desc())
        )
        return list(self._session.scalars(statement))

    def get_treatment_plan(self, clinician_id: str, treatment_plan_id: str) -> TreatmentPlan | None:
        statement = (
            select(TreatmentPlan)
            .options(
                selectinload(TreatmentPlan.patient).selectinload(Patient.clinical_data),
            )
            .where(
                TreatmentPlan.clinician_id == clinician_id,
                TreatmentPlan.id == treatment_plan_id,
            )
        )
        return self._session.scalar(statement)

    def update_treatment_plan(
        self,
        treatment_plan: TreatmentPlan,
        plan_updates: dict[str, Any],
    ) -> TreatmentPlan:
        for key, value in plan_updates.items():
            setattr(treatment_plan, key, value)
        treatment_plan.updated_at = utc_now()
        self._session.commit()
        return treatment_plan

    def delete_treatment_plan(self, treatment_plan: TreatmentPlan) -> None:
        self._session.delete(treatment_plan)
        self._session.commit()
