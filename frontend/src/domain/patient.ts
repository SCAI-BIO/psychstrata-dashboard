/**
 * Patient domain model — the single canonical representation of a patient.
 *
 * This module is framework-agnostic (no React, no state): just the shape,
 * pure `with*` transforms, selectors, and the projection to the model feature
 * vector. The runtime instance is owned by the container hook and shared via
 * PatientContext; every view/selector derives from it.
 */

export interface PatientDemographics {
  firstName: string | null;
  lastName: string | null;
  dob: string | null;
  gender: string | null;
  diagnosis: string | null;
}

export interface Diplotype {
  allele1: string;
  allele2: string;
}

export interface PatientGenetics {
  available: boolean;
  /** Diplotypes keyed by configured gene id (e.g. CYP2C19). */
  genes: Record<string, Diplotype>;
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
  firstName: null,
  lastName: null,
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
    genetics: { available: false, genes: {} },
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

// ── Selectors ─────────────────────────────────────────────────────────────

export type PatientProfilePatch = Partial<
  Pick<Patient, "onMedication" | "episodeDurationMonths" | "sleepDisturbance" | "substanceUse">
>;

export function withProfile(patient: Patient, patch: PatientProfilePatch): Patient {
  return { ...patient, ...patch };
}

export function firstName(patient: Patient): string {
  return (patient.demographics.firstName ?? "").trim();
}

export function displayName(patient: Patient): string {
  return [patient.demographics.firstName, patient.demographics.lastName]
    .map((part) => (part ?? "").trim())
    .filter(Boolean)
    .join(" ");
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

export function isDiplotypeComplete(diplotype: Diplotype | undefined): boolean {
  return Boolean(diplotype?.allele1.trim() && diplotype?.allele2.trim());
}

export function isStepComplete(
  step: number,
  patient: Patient,
  configuredGenes: string[] = []
): boolean {
  switch (step) {
    case 0: {
      const d = patient.demographics;
      return [d.firstName, d.lastName, d.dob, d.gender, d.diagnosis].every((v) => (v ?? "").trim() !== "");
    }
    case 1:
      return ["phq9", "sertraline_mg", "adherence_pct"].every((id) => Number.isFinite(featureValue(patient, id)));
    case 2: {
      if (!patient.genetics.available) return true;
      const genes = configuredGenes.length > 0 ? configuredGenes : Object.keys(patient.genetics.genes);
      return genes.length > 0 && genes.every((gene) => isDiplotypeComplete(patient.genetics.genes[gene]));
    }
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
    genetics: {
      ...patient.genetics,
      genes: Object.fromEntries(
        Object.entries(patient.genetics.genes).map(([gene, diplotype]) => [gene, { ...diplotype }])
      )
    },
    proteomics: { ...patient.proteomics }
  };
}