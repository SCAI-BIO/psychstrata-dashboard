import type { FeatureSchema, PatientCreatePayload } from "../api";
import type { Patient } from "../domain/patient";

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