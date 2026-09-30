import type { FeatureSchema, PatientCreatePayload, PatientRecord, TreatmentPlanRecord } from "../api";
import { createDefaultPatient, type Patient } from "../domain/patient";

/** Ids the backend derives itself (age comes from date_of_birth) and rejects if sent. */
const DERIVED_FEATURE_IDS = new Set(["age"]);

/** Clinical-category values only; medications/adherence belong to treatment plans. */
export function clinicalFeaturesFor(patient: Patient, features: FeatureSchema[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const feature of features) {
    if (feature.category !== "clinical" || DERIVED_FEATURE_IDS.has(feature.id)) continue;
    const value = patient.clinical[feature.id];
    if (value !== undefined) out[feature.id] = value;
  }
  return out;
}

export function patientToCreatePayload(patient: Patient, features: FeatureSchema[]): PatientCreatePayload {
  const d = patient.demographics;
  return {
    first_name: (d.firstName ?? "").trim(),
    last_name: (d.lastName ?? "").trim(),
    clinical_data: {
      date_of_birth: d.dob ?? "",
      diagnosis: (d.diagnosis ?? "").trim(),
      clinical_features: clinicalFeaturesFor(patient, features),
      genetics: { ...patient.genetics },
      proteomics: { ...patient.proteomics }
    }
  };
}

const GENDER_BY_SEX_AT_BIRTH: Record<number, string> = { 0: "Male", 1: "Female", 2: "Other" };

/** Inverse of patientToCreatePayload: rebuild the dashboard Patient from saved records. */
export function recordToPatient(
  record: PatientRecord,
  plan: TreatmentPlanRecord | null,
  features: FeatureSchema[]
): Patient {
  const defaults = Object.fromEntries(features.map((f) => [f.id, f.default]));
  const base = createDefaultPatient(defaults);
  const data = record.clinical_data;

  const clinical: Record<string, number> = {
    ...defaults,
    ...data.clinical_features,
    ...(plan?.medications ?? {}),
    ...(plan?.adherence ?? {})
  };
  const onMedication = ["sertraline_mg", "quetiapine_mg", "lithium_mg"].some((id) => (clinical[id] ?? 0) > 0);

  return {
    ...base,
    demographics: {
      firstName: record.first_name,
      lastName: record.last_name,
      dob: data.date_of_birth,
      gender: GENDER_BY_SEX_AT_BIRTH[data.clinical_features.sex_at_birth] ?? null,
      diagnosis: data.diagnosis
    },
    clinical,
    onMedication,
    episodeDurationMonths: clinical.duration_months ?? null,
    genetics: { available: Boolean(data.genetics.available) },
    proteomics: { available: Boolean(data.proteomics.available) }
  };
}