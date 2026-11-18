from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .patient_record import Patient, PatientClinicalData, TreatmentPlan
from ..io.feature_loader import validate_feature_values
from ..security.basic_auth import DEFAULT_CLINICIAN_ID
from ..settings import get_backend_settings
from .database import get_session_factory

DEMO_PATIENT_ID = "00000000-0000-4000-8000-000000000001"
DEMO_SECOND_PATIENT_ID = "00000000-0000-4000-8000-000000000002"

DEMO_GENETICS = {
    "available": True,
    "genes": {
        "CYP2C19": {"allele1": "*1", "allele2": "*2"},
        "CYP2D6": {"allele1": "*1", "allele2": "*1"},
    },
}
SECOND_DEMO_GENETICS = {
    "available": True,
    "genes": {
        "CYP2C19": {"allele1": "*2", "allele2": "*2"},
        "CYP2D6": {"allele1": "*1", "allele2": "*4"},
    },
}
SECOND_DEMO_CLINICAL_FEATURES = {
    "sex_at_birth": 1,
    "phq9": 22,
    "duration_months": 18,
    "previous_failures": 3,
    "sleep_severity": 2,
    "substance_use": 0,
    "comorbid_anxiety": 1,
}
DEMO_TREATMENT_PLANS = (
    (
        "00000000-0000-4000-8000-000000000101",
        date(2023, 1, 15),
        date(2023, 6, 30),
        {"sertraline_mg": 100, "quetiapine_mg": 0, "lithium_mg": 0},
        {"adherence_pct": 85, "early_improvement": 1, "side_effects": 1},
    ),
    (
        "00000000-0000-4000-8000-000000000102",
        date(2023, 7, 1),
        date(2024, 3, 31),
        {"sertraline_mg": 200, "quetiapine_mg": 0, "lithium_mg": 0},
        {"adherence_pct": 70, "early_improvement": 0, "side_effects": 2},
    ),
    (
        "00000000-0000-4000-8000-000000000103",
        date(2024, 4, 1),
        None,
        {"sertraline_mg": 200, "quetiapine_mg": 150, "lithium_mg": 800},
        {"adherence_pct": 60, "early_improvement": 0, "side_effects": 2},
    ),
)
SECOND_DEMO_TREATMENT_PLANS = (
    (
        "00000000-0000-4000-8000-000000000201",
        date(2022, 3, 1),
        date(2022, 12, 31),
        {"sertraline_mg": 50, "quetiapine_mg": 0, "lithium_mg": 0},
        {"adherence_pct": 90, "early_improvement": 1, "side_effects": 1},
    ),
    (
        "00000000-0000-4000-8000-000000000202",
        date(2023, 1, 1),
        date(2023, 10, 31),
        {"sertraline_mg": 150, "quetiapine_mg": 0, "lithium_mg": 0},
        {"adherence_pct": 75, "early_improvement": 0, "side_effects": 2},
    ),
    (
        "00000000-0000-4000-8000-000000000203",
        date(2023, 11, 1),
        None,
        {"sertraline_mg": 150, "quetiapine_mg": 150, "lithium_mg": 600},
        {"adherence_pct": 55, "early_improvement": 0, "side_effects": 3},
    ),
)


def _upsert_patient(
    session: Session,
    *,
    patient_id: str,
    clinician_id: str,
    first_name: str,
    last_name: str,
    date_of_birth: date,
    diagnosis: str,
    clinical_features: dict[str, Any],
    genetics: dict[str, Any],
) -> Patient:
    patient = session.get(Patient, patient_id)
    if patient is None:
        patient = Patient(
            id=patient_id,
            clinician_id=clinician_id,
            first_name=first_name,
            last_name=last_name,
            clinical_data=PatientClinicalData(
                date_of_birth=date_of_birth,
                diagnosis=diagnosis,
                clinical_features=clinical_features,
                genetics=genetics,
            ),
        )
        session.add(patient)
    else:
        patient.clinician_id = clinician_id
        if patient.clinical_data is not None:
            patient.clinical_data.genetics = genetics
    return patient


def _upsert_treatment_plans(
    session: Session,
    *,
    patient: Patient,
    clinician_id: str,
    plans: tuple[tuple[str, date, date | None, dict[str, int], dict[str, int]], ...],
) -> None:
    existing_plans = {
        plan.id: plan
        for plan in session.scalars(
            select(TreatmentPlan).where(TreatmentPlan.patient_id == patient.id)
        )
    }
    for plan_id, start_date, end_date, medication_values, adherence_values in plans:
        medications = validate_feature_values("medications", medication_values, include_defaults=True)
        adherence = validate_feature_values("adherence", adherence_values, include_defaults=True)
        plan = existing_plans.get(plan_id)
        if plan is None:
            session.add(
                TreatmentPlan(
                    id=plan_id,
                    patient=patient,
                    clinician_id=clinician_id,
                    start_date=start_date,
                    end_date=end_date,
                    medications=medications,
                    adherence=adherence,
                )
            )
        else:
            plan.clinician_id = clinician_id
            plan.start_date = start_date
            plan.end_date = end_date
            plan.medications = medications
            plan.adherence = adherence


def seed_demo_data() -> None:
    settings = get_backend_settings()
    clinician_id = settings.backend_basic_auth_username or DEFAULT_CLINICIAN_ID
    clinical_defaults = validate_feature_values("clinical", {}, include_defaults=True)

    with get_session_factory()() as session:
        patient = _upsert_patient(
            session,
            patient_id=DEMO_PATIENT_ID,
            clinician_id=clinician_id,
            first_name="Max",
            last_name="Mustermann",
            date_of_birth=date(1980, 5, 15),
            diagnosis="F33.1",
            clinical_features=clinical_defaults,
            genetics=DEMO_GENETICS,
        )
        _upsert_treatment_plans(
            session,
            patient=patient,
            clinician_id=clinician_id,
            plans=DEMO_TREATMENT_PLANS,
        )

        second_patient = _upsert_patient(
            session,
            patient_id=DEMO_SECOND_PATIENT_ID,
            clinician_id=clinician_id,
            first_name="Erika",
            last_name="Mustermann",
            date_of_birth=date(1972, 11, 3),
            diagnosis="F33.2",
            clinical_features=validate_feature_values(
                "clinical",
                SECOND_DEMO_CLINICAL_FEATURES,
                include_defaults=True,
            ),
            genetics=SECOND_DEMO_GENETICS,
        )
        _upsert_treatment_plans(
            session,
            patient=second_patient,
            clinician_id=clinician_id,
            plans=SECOND_DEMO_TREATMENT_PLANS,
        )
        session.commit()
