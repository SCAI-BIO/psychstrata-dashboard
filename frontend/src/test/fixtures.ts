import { vi } from "vitest";

import type {
  FeaturesResponse,
  PredictionResponse,
  TsneResponse
} from "../api";
import type { PatientApi } from "../context/PatientContext";
import {
  createDefaultPatient,
  patientToFeatures,
  type Patient
} from "../domain/patient";
import type { DashboardApi } from "../hooks/useDashboard";
import type { LoadState, ReadyState } from "../types";

/**
 * Shared test fixtures. These mirror the real backend payload shapes so the
 * unit/component tests exercise the same types the app runs against.
 */

export const featuresPayload: FeaturesResponse = {
  features: [
    { id: "phq9", label: "PHQ-9", kind: "numeric", default: 12, params: { min: 0, max: 27, step: 1 }, min: 0, max: 27, step: 1 },
    { id: "adherence_pct", label: "Adherence (%)", kind: "numeric", default: 80, params: { min: 0, max: 100, step: 1 }, min: 0, max: 100, step: 1 },
    { id: "sertraline_mg", label: "Sertraline", kind: "numeric", default: 100, params: { min: 0, max: 200, step: 5 }, min: 0, max: 200, step: 5 },
    { id: "lithium_mg", label: "Lithium", kind: "numeric", default: 0, params: { min: 0, max: 1200, step: 100 }, min: 0, max: 1200, step: 100 },
    { id: "quetiapine_mg", label: "Quetiapine", kind: "numeric", default: 0, params: { min: 0, max: 300, step: 25 }, min: 0, max: 300, step: 25 }
  ],
  defaults: { phq9: 12, adherence_pct: 80, sertraline_mg: 100, lithium_mg: 0, quetiapine_mg: 0 },
  model_feature_order: ["phq9", "adherence_pct", "sertraline_mg", "lithium_mg", "quetiapine_mg"],
  confidence_level: { default: 95, min: 80, max: 99, step: 1 }
};

export const tsneResponse: TsneResponse = {
  points: [
    { x: 0, y: 0, class_value: 0, class_label: "Responsive" },
    { x: 1, y: 1, class_value: 1, class_label: "Resistant" }
  ]
};

export const predictionResponse: PredictionResponse = {
  features: { phq9: 12, adherence_pct: 80, sertraline_mg: 100, lithium_mg: 0, quetiapine_mg: 0 },
  prediction: {
    probability_resistance: 0.68,
    predicted_class: "Resistant",
    conformal_prediction: {
      confidence_level: 95,
      alpha: 0.05,
      label: "Resistant",
      included_classes: ["Resistant"]
    }
  },
  shap_values: [
    {
      feature_id: "phq9",
      feature_label: "PHQ-9",
      selected_value: 12,
      selected_value_label: "12",
      shap_value: 0.3,
      abs_shap_value: 0.3,
      direction: "raises"
    }
  ],
  top_contributors: {
    positive: [{ feature_id: "phq9", feature_label: "PHQ-9", selected_value: "12", shap_value: 0.3, direction: "raises" }],
    negative: [{ feature_id: "adherence_pct", feature_label: "Adherence", selected_value: "80%", shap_value: -0.2, direction: "lowers" }]
  },
  tsne: { selected: { x: 0.2, y: 0.3 } },
  disclaimer: "Synthetic data for demonstration only."
};

export const explainResponse = {
  features: predictionResponse.features,
  prediction: predictionResponse.prediction,
  top_contributors: predictionResponse.top_contributors,
  explanation: "Model explanation text."
};

/** A patient whose demographics + clinical fields satisfy the intake wizard. */
export function makeCompletePatient(overrides: Partial<Patient> = {}): Patient {
  const base = createDefaultPatient(featuresPayload.defaults);
  return {
    ...base,
    demographics: {
      name: "John Doe",
      patientId: "98421",
      dob: "1978-05-12",
      gender: "Male",
      diagnosis: "F33.1 — Major depressive disorder, recurrent, moderate"
    },
    episodeDurationMonths: 8,
    sleepDisturbance: "Moderate",
    substanceUse: "None",
    ...overrides
  };
}

export function makeReadyState(overrides: Partial<ReadyState> = {}): ReadyState {
  return {
    status: "ready",
    features: featuresPayload.features,
    confidenceBounds: { min: 80, max: 99, step: 1 },
    confidenceLevel: 95,
    prediction: null,
    explanation: "",
    tsne: tsneResponse,
    isSubmitting: false,
    isSummaryRefreshing: false,
    error: null,
    ...overrides
  };
}

/** Build a PatientApi backed by a fixed patient (writers are spies). */
export function makePatientApi(patient: Patient = makeCompletePatient()): PatientApi {
  return {
    patient,
    featureValues: patientToFeatures(patient),
    readFeature: (id: string) => patient.clinical[id] ?? patient.extras[id],
    writeFeature: vi.fn(),
    resetPatient: vi.fn(),
    updateDemographics: vi.fn(),
    updateGenetics: vi.fn(),
    updateProteomics: vi.fn(),
    updateProfile: vi.fn()
  };
}

/**
 * Build a DashboardApi with sensible defaults; pass overrides for the fields a
 * given test cares about. Function fields default to vi.fn() spies.
 */
export function makeDashboard(overrides: Partial<DashboardApi> = {}): DashboardApi {
  const patient = overrides.patient ?? makeCompletePatient();
  const state: LoadState = overrides.state ?? makeReadyState();
  const dashboard: DashboardApi = {
    authRequired: false,
    isAuthenticated: true,
    loginError: null,
    isLoginSubmitting: false,
    attemptLogin: vi.fn(async () => {}),
    signOut: vi.fn(),
    route: "intake",
    navigate: vi.fn(),
    state,
    patient,
    patientApi: makePatientApi(patient),
    featureValues: patientToFeatures(patient),
    setFeatureValue: vi.fn(),
    setConfidenceLevel: vi.fn(),
    runPrediction: vi.fn(async () => {}),
    refreshExplanation: vi.fn(async () => {}),
    clinicianSimValues: { sertraline_mg: 0, lithium_mg: 0, quetiapine_mg: 0, adherence_pct: 75 },
    setClinicianSimValues: vi.fn(),
    clinicianImpactPct: 0,
    isClinicianImpactSubmitting: false,
    clinicianImpactError: null,
    isClinicianApplySubmitting: false,
    clinicianApplyError: null,
    clinicianAppliedImpactPct: 0,
    runClinicianImpactPrediction: vi.fn(async () => {}),
    applyClinicianSimulationToPage: vi.fn(async () => {}),
    runScenarioImpact: vi.fn(async () => {}),
    applyScenario: vi.fn(async () => {}),
    ...overrides
  };
  return dashboard;
}
