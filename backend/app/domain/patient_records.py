from datetime import date, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Date, DateTime, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..persistence.database import Base
from ..utils.datetime import utc_now


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    clinician_id: Mapped[str] = mapped_column(String, index=True)
    first_name: Mapped[str] = mapped_column(String)
    last_name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    clinical_data: Mapped["PatientClinicalData"] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        uselist=False,
    )
    treatment_plans: Mapped[list["TreatmentPlan"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class PatientClinicalData(Base):
    __tablename__ = "patient_clinical_data"

    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        primary_key=True,
    )
    date_of_birth: Mapped[date] = mapped_column(Date)
    diagnosis: Mapped[str] = mapped_column(String)
    clinical_features: Mapped[dict[str, Any]] = mapped_column(JSON)
    genetics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    proteomics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    patient: Mapped[Patient] = relationship(back_populates="clinical_data")


class TreatmentPlan(Base):
    __tablename__ = "treatment_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        index=True,
    )
    clinician_id: Mapped[str] = mapped_column(String, index=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    medications: Mapped[dict[str, Any]] = mapped_column(JSON)
    adherence: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    patient: Mapped[Patient] = relationship(back_populates="treatment_plans")
