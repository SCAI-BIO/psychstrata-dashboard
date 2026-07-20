import { describe, expect, it } from "vitest";

import {
  activeMedicationCount,
  clonePatient,
  createDefaultPatient,
  firstName,
  isStepComplete,
  patientToFeatures,
  withClinical,
  withClinicalMerged,
  withDemographics,
  withExtra,
  withProfile
} from "./patient";

describe("createDefaultPatient", () => {
  it("seeds the clinical vector with the provided defaults", () => {
    const p = createDefaultPatient({ phq9: 12, adherence_pct: 80 });
    expect(p.clinical).toEqual({ phq9: 12, adherence_pct: 80 });
    expect(p.extras).toEqual({});
    expect(p.onMedication).toBe(true);
  });
});

describe("pure transforms are immutable", () => {
  it("withClinical does not mutate the source patient", () => {
    const p = createDefaultPatient({ phq9: 10 });
    const next = withClinical(p, "phq9", 20);
    expect(next.clinical.phq9).toBe(20);
    expect(p.clinical.phq9).toBe(10);
    expect(next).not.toBe(p);
  });

  it("withClinicalMerged merges a patch over existing values", () => {
    const p = createDefaultPatient({ phq9: 10, adherence_pct: 80 });
    const next = withClinicalMerged(p, { adherence_pct: 60, sertraline_mg: 100 });
    expect(next.clinical).toEqual({ phq9: 10, adherence_pct: 60, sertraline_mg: 100 });
  });

  it("withExtra stores values outside the model vector", () => {
    const p = createDefaultPatient();
    const next = withExtra(p, "custom_score", 3);
    expect(next.extras.custom_score).toBe(3);
    expect(patientToFeatures(next)).not.toHaveProperty("custom_score");
  });

  it("withDemographics and withProfile patch nested/flat fields", () => {
    const p = createDefaultPatient();
    const named = withDemographics(p, { name: "Jane Roe" });
    expect(named.demographics.name).toBe("Jane Roe");
    const profiled = withProfile(p, { onMedication: false });
    expect(profiled.onMedication).toBe(false);
  });
});

describe("patientToFeatures", () => {
  it("projects only the clinical vector (not extras)", () => {
    let p = createDefaultPatient({ phq9: 12 });
    p = withExtra(p, "note", 1);
    expect(patientToFeatures(p)).toEqual({ phq9: 12 });
  });
});

describe("selectors", () => {
  it("firstName returns the first token of the name", () => {
    const p = withDemographics(createDefaultPatient(), { name: "John Doe" });
    expect(firstName(p)).toBe("John");
  });

  it("firstName is empty when no name is set", () => {
    expect(firstName(createDefaultPatient())).toBe("");
  });

  it("activeMedicationCount counts dosed agents", () => {
    const p = createDefaultPatient({ sertraline_mg: 100, lithium_mg: 0, quetiapine_mg: 25 });
    expect(activeMedicationCount(p)).toBe(2);
  });
});

describe("isStepComplete", () => {
  it("requires all key demographics for step 0", () => {
    const incomplete = withDemographics(createDefaultPatient(), { name: "John Doe" });
    expect(isStepComplete(0, incomplete)).toBe(false);

    const complete = withDemographics(createDefaultPatient(), {
      name: "John Doe",
      dob: "1978-05-12",
      gender: "Male",
      diagnosis: "F33.1"
    });
    expect(isStepComplete(0, complete)).toBe(true);
  });

  it("requires the core clinical fields for step 1", () => {
    const incomplete = createDefaultPatient({ phq9: 12 });
    expect(isStepComplete(1, incomplete)).toBe(false);

    const complete = createDefaultPatient({ phq9: 12, sertraline_mg: 100, adherence_pct: 80 });
    expect(isStepComplete(1, complete)).toBe(true);
  });

  it("treats later steps as always complete", () => {
    expect(isStepComplete(2, createDefaultPatient())).toBe(true);
  });
});

describe("clonePatient", () => {
  it("produces an independent deep-ish copy of the nested records", () => {
    const p = createDefaultPatient({ phq9: 12 });
    const copy = clonePatient(p);
    copy.clinical.phq9 = 99;
    expect(p.clinical.phq9).toBe(12);
    expect(copy.demographics).not.toBe(p.demographics);
  });
});
