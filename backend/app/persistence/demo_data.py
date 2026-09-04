from datetime import date

from sqlalchemy import select

from ..domain.patient_records import Patient, PatientClinicalData, TreatmentPlan
from ..io.feature_loader import validate_feature_values
from ..security.basic_auth import DEFAULT_CLINICIAN_ID
from ..settings import get_backend_settings
from .database import get_session_factory

DEMO_PATIENT_ID = "00000000-0000-4000-8000-000000000001"
DEMO_TREATMENT_PLANS = (
    ("00000000-0000-4000-8000-000000000101", date(2023, 1, 15), date(2023, 6, 30)),
    ("00000000-0000-4000-8000-000000000102", date(2023, 7, 1), date(2024, 3, 31)),
    ("00000000-0000-4000-8000-000000000103", date(2024, 4, 1), None),
)
DEMO_TREATMENT_PLAN_IDS = {plan_id for plan_id, _, _ in DEMO_TREATMENT_PLANS}


def seed_demo_data() -> None:
    settings = get_backend_settings()
    clinician_id = settings.backend_basic_auth_username or DEFAULT_CLINICIAN_ID
    clinical_features = validate_feature_values("clinical", {}, include_defaults=True)
    medications = validate_feature_values("medications", {}, include_defaults=True)
    adherence = validate_feature_values("adherence", {}, include_defaults=True)

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

        existing_plan_ids = set(
            session.scalars(
                select(TreatmentPlan.id).where(TreatmentPlan.patient_id == DEMO_PATIENT_ID)
            )
        )
        for plan_id, start_date, end_date in DEMO_TREATMENT_PLANS:
            if plan_id in existing_plan_ids:
                continue
            session.add(
                TreatmentPlan(
                    id=plan_id,
                    patient=patient,
                    clinician_id=clinician_id,
                    start_date=start_date,
                    end_date=end_date,
                    medications=dict(medications),
                    adherence=dict(adherence),
                )
            )
        for treatment_plan in patient.treatment_plans:
            if treatment_plan.id in DEMO_TREATMENT_PLAN_IDS:
                treatment_plan.clinician_id = clinician_id
        session.commit()
