import { describe, expect, it } from "vitest";

import type { FeatureSchema } from "../api";
import { createDefaultPatient, withClinical, withDemographics, type Patient } from "../domain/patient";
import {
  clinicalFeaturesFor,
  patientToCreatePayload,
  recordToPatient,
  treatmentPlanFromCurrentRegimen
} from "./patientPersistence";

const FEATURES: FeatureSchema[] = [
  { id: "age", label: "Age", dtype: "numeric", category: "clinical", default: 40, min: 18, max: 80, step: 1 },
  { id: "sex_at_birth", label: "Sex at birth", dtype: "categorical", category: "clinical", default: 0, options: [{ value: 0, label: "Male" }, { value: 1, label: "Female" }] },
  { id: "phq9", label: "PHQ-9", dtype: "numeric", category: "clinical", default: 18, min: 0, max: 27, step: 1 },
  { id: "sertraline_mg", label: "Sertraline", dtype: "numeric", category: "medications", default: 100, min: 0, max: 200, step: 5 },
  { id: "lithium_mg", label: "Lithium", dtype: "numeric", category: "medications", default: 0, min: 0, max: 1200, step: 100 },
  { id: "adherence_pct", label: "Adherence", dtype: "numeric", category: "adherence", default: 80, min: 0, max: 100, step: 1 }
];

function basePatient(): Patient {
  const p = createDefaultPatient({ age: 40, sex_at_birth: 0, phq9: 18, sertraline_mg: 100, lithium_mg: 0, adherence_pct: 80 });
  return withDemographics(p, {
    firstName: " John ",
    lastName: " Doe ",
    dob: "1978-05-12",
    gender: "Male",
    diagnosis: " F33.1 "
  });
}

describe("clinicalFeaturesFor", () => {
  it("keeps only clinical-category features and drops derived ids", () => {
    const patient = withClinical(basePatient(), "phq9", 20);
    const out = clinicalFeaturesFor(patient, FEATURES);
    expect(out).toEqual({ sex_at_birth: 0, phq9: 20 });
    expect(out).not.toHaveProperty("age");
    expect(out).not.toHaveProperty("sertraline_mg");
  });

  it("skips features with no value on the patient", () => {
    const patient = createDefaultPatient({ sex_at_birth: 1 });
    expect(clinicalFeaturesFor(patient, FEATURES)).toEqual({ sex_at_birth: 1 });
  });
});

describe("patientToCreatePayload", () => {
  it("trims demographics and shapes clinical_data", () => {
    const patient = withClinical(basePatient(), "phq9", 21);
    const payload = patientToCreatePayload(patient, FEATURES);

    expect(payload.first_name).toBe("John");
    expect(payload.last_name).toBe("Doe");
    expect(payload.clinical_data.date_of_birth).toBe("1978-05-12");
    expect(payload.clinical_data.diagnosis).toBe("F33.1");
    expect(payload.clinical_data.clinical_features).toEqual({ sex_at_birth: 0, phq9: 21 });
  });

  it("falls back to empty strings for missing demographics", () => {
    const payload = patientToCreatePayload(createDefaultPatient(), FEATURES);
    expect(payload.first_name).toBe("");
    expect(payload.last_name).toBe("");
    expect(payload.clinical_data.date_of_birth).toBe("");
    expect(payload.clinical_data.diagnosis).toBe("");
  });

  it("copies genetics and proteomics", () => {
    const patient = basePatient();
    patient.genetics = { available: true, cyp2d6: "poor" };
    patient.proteomics = { available: true, crp: 3.2 };
    const payload = patientToCreatePayload(patient, FEATURES);
    expect(payload.clinical_data.genetics).toEqual({ available: true, cyp2d6: "poor" });
    expect(payload.clinical_data.proteomics).toEqual({ available: true, crp: 3.2 });
  });
});

describe("treatmentPlanFromCurrentRegimen", () => {
  it("splits medication and adherence features, ignoring clinical + derived", () => {
    const patient = withClinical(basePatient(), "lithium_mg", 300);
    const plan = treatmentPlanFromCurrentRegimen(patient, FEATURES);
    expect(plan).toEqual({
      medications: { sertraline_mg: 100, lithium_mg: 300 },
      adherence: { adherence_pct: 80 }
    });
  });
});

describe("recordToPatient", () => {
  const record = {
    id: "patient-1",
    clinician_id: "clinician-1",
    first_name: "Jane",
    last_name: "Roe",
    clinical_data: {
      date_of_birth: "1980-01-01",
      diagnosis: "F33.1",
      clinical_features: { sex_at_birth: 1, phq9: 22, duration_months: 10 },
      genetics: { available: true },
      proteomics: { available: false }
    },
    created_at: "2026-09-01T10:00:00Z",
    updated_at: "2026-09-01T10:00:00Z"
  };

  it("rebuilds demographics, gender and flags", () => {
    const patient = recordToPatient(record, null, FEATURES);
    expect(patient.demographics.firstName).toBe("Jane");
    expect(patient.demographics.lastName).toBe("Roe");
    expect(patient.demographics.dob).toBe("1980-01-01");
    expect(patient.demographics.gender).toBe("Female");
    expect(patient.demographics.diagnosis).toBe("F33.1");
    expect(patient.genetics).toEqual({ available: true });
    expect(patient.proteomics).toEqual({ available: false });
  });

  it("merges defaults, saved features and plan medications/adherence", () => {
    const plan = {
      id: "plan-1",
      patient_id: "patient-1",
      clinician_id: "clinician-1",
      start_date: "2026-10-01",
      end_date: null,
      medications: { sertraline_mg: 150, lithium_mg: 0 },
      adherence: { adherence_pct: 90 },
      created_at: "2026-10-01T10:00:00Z",
      updated_at: "2026-10-01T10:00:00Z"
    };
    const patient = recordToPatient(record, plan, FEATURES);
    expect(patient.clinical.sertraline_mg).toBe(150);
    expect(patient.clinical.adherence_pct).toBe(90);
    expect(patient.clinical.phq9).toBe(22);
    expect(patient.episodeDurationMonths).toBe(10);
    expect(patient.onMedication).toBe(true);
  });

  it("reports onMedication false when no agent is dosed", () => {
    const noMeds = FEATURES.map((f) =>
      f.category === "medications" ? { ...f, default: 0 } : f
    );
    const patient = recordToPatient(record, null, noMeds);
    expect(patient.onMedication).toBe(false);
  });

  it("maps unknown sex_at_birth values to a null gender", () => {
    const p = recordToPatient(
      { ...record, clinical_data: { ...record.clinical_data, clinical_features: { sex_at_birth: 9 } } },
      null,
      FEATURES
    );
    expect(p.demographics.gender).toBeNull();
  });
});
