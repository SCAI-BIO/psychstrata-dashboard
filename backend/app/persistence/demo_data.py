from datetime import date

from sqlalchemy import select

from .patient_record import Patient, PatientClinicalData, TreatmentPlan
from ..io.feature_loader import validate_feature_values
from ..security.basic_auth import DEFAULT_CLINICIAN_ID
from ..settings import get_backend_settings
from .database import get_session_factory

DEMO_PATIENT_ID = "00000000-0000-4000-8000-000000000001"
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


def seed_demo_data() -> None:
    settings = get_backend_settings()
    clinician_id = settings.backend_basic_auth_username or DEFAULT_CLINICIAN_ID
    clinical_features = validate_feature_values("clinical", {}, include_defaults=True)

    with get_session_factory()() as session:
        patient = session.get(Patient, DEMO_PATIENT_ID)
        if patient is None:
            patient = Patient(
                id=DEMO_PATIENT_ID,
                clinician_id=clinician_id,
                first_name="Max",
                last_name="Mustermann",
                clinical_data=PatientClinicalData(
                    date_of_birth=date(1980, 5, 15),
                    diagnosis="F33.1",
                    clinical_features=clinical_features,
                ),
            )
            session.add(patient)
        else:
            patient.clinician_id = clinician_id

        existing_plans = {
            plan.id: plan
            for plan in session.scalars(
                select(TreatmentPlan).where(TreatmentPlan.patient_id == DEMO_PATIENT_ID)
            )
        }
        for plan_id, start_date, end_date, medication_values, adherence_values in DEMO_TREATMENT_PLANS:
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
        session.commit()
