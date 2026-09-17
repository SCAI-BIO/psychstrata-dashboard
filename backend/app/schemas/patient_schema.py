from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClinicalDataCreate(BaseModel):
    date_of_birth: date
    diagnosis: str = Field(min_length=1)
    clinical_features: dict[str, Any] = Field(default_factory=dict)
    genetics: dict[str, Any] = Field(default_factory=dict)
    proteomics: dict[str, Any] = Field(default_factory=dict)


class ClinicalDataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date_of_birth: date
    diagnosis: str
    clinical_features: dict[str, Any]
    genetics: dict[str, Any]
    proteomics: dict[str, Any]


class PatientCreate(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    clinical_data: ClinicalDataCreate


class PatientUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_name: str | None = Field(default=None, min_length=1)
    last_name: str | None = Field(default=None, min_length=1)


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    clinician_id: str
    first_name: str
    last_name: str
    clinical_data: ClinicalDataRead
    created_at: datetime
    updated_at: datetime


class TreatmentPlanCreate(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    medications: dict[str, Any] = Field(default_factory=dict)
    adherence: dict[str, Any] = Field(default_factory=dict)


class TreatmentPlanUpdate(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    medications: dict[str, Any] | None = None
    adherence: dict[str, Any] | None = None


class TreatmentPlanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    clinician_id: str
    start_date: date | None
    end_date: date | None
    medications: dict[str, Any]
    adherence: dict[str, Any]
    created_at: datetime
    updated_at: datetime
