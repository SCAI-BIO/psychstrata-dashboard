/**
 * Patient domain model — the single canonical representation of a patient.
 *
 * This module is framework-agnostic (no React, no state): just the shape,
 * pure `with*` transforms, selectors, and the projection to the model feature
 * vector. The runtime instance is owned by the container hook and shared via
 * PatientContext; every view/selector derives from it.
 */

export interface PatientDemographics {
  name: string | null;
  patientId: string | null;
  dob: string | null;
  gender: string | null;
  diagnosis: string | null;
}

export interface PatientGenetics {
  available: boolean;
  cyp2d6?: string;
  slc6a4?: string;
  bdnf?: string;
}

export interface PatientProteomics {
  available: boolean;
  crp?: number;
  il6?: number;
  tnfAlpha?: number;
}

export interface Patient {
  demographics: PatientDemographics;
  /** Model feature vector — the source of truth for prediction inputs. */
  clinical: Record<string, number>;
  /** Numeric clinical fields captured but NOT part of the backend schema. */
  extras: Record<string, number>;
  onMedication: boolean | null;
  episodeDurationMonths: number | null;
  sleepDisturbance: string | null;
  substanceUse: string | null;
  genetics: PatientGenetics;
  proteomics: PatientProteomics;
}

/** Presentation-only default identity (mirrors the mockups). */
const DEFAULT_DEMOGRAPHICS: PatientDemographics = {
  name: null,
  patientId: null,
  dob: null,
  gender: null,
  diagnosis: null
};

/** Build a fresh patient, seeding the feature vector with backend defaults. */
export function createDefaultPatient(clinicalDefaults: Record<string, number> = {}): Patient {
  return {
    demographics: { ...DEFAULT_DEMOGRAPHICS },
    clinical: { ...clinicalDefaults },
    extras: {},
    onMedication: true,
    episodeDurationMonths: null,
    sleepDisturbance: null,
    substanceUse: null,
    genetics: { available: true },
    proteomics: { available: true }
  };
}

// ── Projection to the model ────────────────────────────────────────────────

/** The feature payload sent to the prediction API (known features only). */
export function patientToFeatures(patient: Patient): Record<string, number> {
  return { ...patient.clinical };
}

// ── Pure transforms ─────────────────────────────────────────────────────────

export function withClinical(patient: Patient, id: string, value: number): Patient {
  return { ...patient, clinical: { ...patient.clinical, [id]: value } };
}

export function withClinicalMerged(patient: Patient, patch: Record<string, number>): Patient {
  return { ...patient, clinical: { ...patient.clinical, ...patch } };
}

export function withExtra(patient: Patient, id: string, value: number): Patient {
  return { ...patient, extras: { ...patient.extras, [id]: value } };
}

export function withDemographics(patient: Patient, patch: Partial<PatientDemographics>): Patient {
  return { ...patient, demographics: { ...patient.demographics, ...patch } };
}

export function withGenetics(patient: Patient, patch: Partial<PatientGenetics>): Patient {
  return { ...patient, genetics: { ...patient.genetics, ...patch } };
}

export function withProteomics(patient: Patient, patch: Partial<PatientProteomics>): Patient {
  return { ...patient, proteomics: { ...patient.proteomics, ...patch } };
}

export type PatientProfilePatch = Partial<
  Pick<Patient, "onMedication" | "episodeDurationMonths" | "sleepDisturbance" | "substanceUse">
>;

export function withProfile(patient: Patient, patch: PatientProfilePatch): Patient {
  return { ...patient, ...patch };
}

// ── Selectors ─────────────────────────────────────────────────────────────

export function firstName(patient: Patient): string {
  const name = patient.demographics.name ?? "";
  return name.split(" ")[0] || name;
}

export function displayName(patient: Patient): string {
  return patient.demographics.name ?? "";
}

export function isGeneticsAvailable(patient: Patient): boolean {
  return patient.genetics.available;
}

export function isProteomicsAvailable(patient: Patient): boolean {
  return patient.proteomics.available;
}

/** Number of active agents inferred from the medication feature values. */
export function activeMedicationCount(patient: Patient): number {
  return ["sertraline_mg", "lithium_mg", "quetiapine_mg"].filter((id) => (patient.clinical[id] ?? 0) > 0).length;
}

/** Sanity Checks for Intake */
function featureValue(patient: Patient, id: string): number | undefined {
  return patient.clinical[id] ?? patient.extras[id];
}

export function isStepComplete(step: number, patient: Patient): boolean {
  switch (step) {
    case 0: {
      const d = patient.demographics;
      console.log("Demographics: %o", d);
      return [d.name, d.dob, d.gender, d.diagnosis].every((v) => (v ?? "").trim() !== "");
    }
    case 1:
      return ["phq9", "sertraline_mg", "adherence_pct"].every((id) => Number.isFinite(featureValue(patient, id)));
    default:
      return true;
  }
}

export function clonePatient(patient: Patient): Patient {
  return {
    ...patient,
    demographics: { ...patient.demographics },
    clinical: { ...patient.clinical },
    extras: { ...patient.extras },
    genetics: { ...patient.genetics },
    proteomics: { ...patient.proteomics }
  };
}