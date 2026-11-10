import type {
  FeatureSchema,
  PatientCreatePayload,
  PatientRecord,
  TreatmentPlanCreatePayload,
  TreatmentPlanRecord
} from "../api";
import { createDefaultPatient, type Diplotype, type Patient, type PatientGenetics } from "../domain/patient";

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

/**
 * Build a new treatment plan from the patient's current regimen: the
 * medications + adherence values the model knows about. `start_date` is left
 * out so the backend defaults it to today (closing the previous open plan).
 */
export function treatmentPlanFromCurrentRegimen(
  patient: Patient,
  features: FeatureSchema[]
): TreatmentPlanCreatePayload {
  const medications: Record<string, number> = {};
  const adherence: Record<string, number> = {};
  for (const feature of features) {
    if (DERIVED_FEATURE_IDS.has(feature.id)) continue;
    const value = patient.clinical[feature.id];
    if (value === undefined) continue;
    if (feature.category === "medications") medications[feature.id] = value;
    else if (feature.category === "adherence") adherence[feature.id] = value;
  }
  return { medications, adherence };
}

const GENDER_BY_SEX_AT_BIRTH: Record<number, string> = { 0: "Male", 1: "Female", 2: "Other" };

function readDiplotype(value: unknown): Diplotype | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const { allele1, allele2 } = value as Record<string, unknown>;
  if (typeof allele1 !== "string" || !allele1 || typeof allele2 !== "string" || !allele2) {
    return undefined;
  }
  return { allele1, allele2 };
}

/** Restore saved genetics, keeping only well-formed diplotypes. */
function geneticsFromRecord(raw: Record<string, unknown>): PatientGenetics {
  const genetics: PatientGenetics = { available: Boolean(raw.available) };
  const cyp2c19 = readDiplotype(raw.cyp2c19);
  const cyp2d6 = readDiplotype(raw.cyp2d6);
  if (cyp2c19) genetics.cyp2c19 = cyp2c19;
  if (cyp2d6) genetics.cyp2d6 = cyp2d6;
  return genetics;
}

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
    genetics: geneticsFromRecord(data.genetics),
    proteomics: { available: Boolean(data.proteomics.available) }
  };
}