import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setBasicAuthHeader } from "../api";
import type { PatientRecord } from "../api";
import {
  explainResponse,
  featuresPayload,
  makeCompletePatient,
  predictionResponse,
  treatmentPlanRecord,
  tsneResponse
} from "../test/fixtures";
import { useDashboard } from "./useDashboard";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const patientRecord: PatientRecord = {
  id: "patient-42",
  clinician_id: "clinician-1",
  first_name: "Ada",
  last_name: "Lovelace",
  clinical_data: {
    date_of_birth: "1970-01-01",
    diagnosis: "F33.1 — Recurrent",
    clinical_features: { phq9: 10, sex_at_birth: 1, duration_months: 6 },
    genetics: { available: true },
    proteomics: { available: false }
  },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z"
};

/** A permissive fetch mock covering every endpoint the dashboard touches. */
function makeFetchMock() {
  return vi.fn(async (url: string) => {
    if (url.endsWith("/api/auth/status")) return json({ auth_enabled: false });
    if (url.endsWith("/api/features")) return json(featuresPayload);
    if (url.endsWith("/api/tsne")) return json(tsneResponse);
    if (url.endsWith("/api/predict")) return json(predictionResponse);
    if (url.endsWith("/api/explain")) return json(explainResponse);
    if (url.endsWith("/api/patients") ) return json(patientRecord, 201);
    if (url.endsWith("/treatment-plans")) return json(treatmentPlanRecord, 201);
    throw new Error(`Unexpected fetch URL: ${url}`);
  });
}

async function readyDashboard(fetchMock: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("fetch", fetchMock);
  const rendered = renderHook(() => useDashboard());
  await waitFor(() => expect(rendered.result.current.state.status).toBe("ready"));
  return rendered;
}

describe("useDashboard actions", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    window.history.pushState({}, "", "/");
    setBasicAuthHeader(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setBasicAuthHeader(null);
  });

  it("writes known features to clinical and unknown ones to extras", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);

    act(() => result.current.setFeatureValue("phq9", 20));
    expect(result.current.patient.clinical.phq9).toBe(20);

    act(() => result.current.setFeatureValue("unknown_marker", 3));
    expect(result.current.patient.extras.unknown_marker).toBe(3);
    expect(result.current.patientApi.readFeature("unknown_marker")).toBe(3);
    expect(result.current.patientApi.readFeature("phq9")).toBe(20);
  });

  it("updates the confidence level only when ready", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    act(() => result.current.setConfidenceLevel(90));
    expect(result.current.state.status === "ready" && result.current.state.confidenceLevel).toBe(90);
  });

  it("navigates and syncs the route on popstate", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);

    act(() => result.current.navigate("scientist"));
    expect(result.current.route).toBe("scientist");
    expect(window.location.pathname).toBe("/results/scientist");

    act(() => {
      window.history.pushState({}, "", "/intake");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current.route).toBe("intake");
  });

  it("persists an intake, dedupes repeat saves, and surfaces failures", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);

    let saved: Awaited<ReturnType<typeof result.current.persistIntake>> = null;
    await act(async () => {
      saved = await result.current.persistIntake();
    });
    expect((saved as PatientRecord | null)?.id).toBe("patient-42");
    expect(result.current.currentPatientId).toBe("patient-42");
    const createCalls = fetchMock.mock.calls.filter((c) => (c[0] as string).endsWith("/api/patients"));
    expect(createCalls).toHaveLength(1);

    // Second identical save reuses the cached record without hitting the API again.
    await act(async () => {
      saved = await result.current.persistIntake();
    });
    expect((saved as PatientRecord | null)?.id).toBe("patient-42");
    expect(fetchMock.mock.calls.filter((c) => (c[0] as string).endsWith("/api/patients"))).toHaveLength(1);
  });

  it("records an intake error on the ready state", async () => {
    const fetchMock = makeFetchMock();
    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/auth/status")) return json({ auth_enabled: false });
      if (url.endsWith("/api/features")) return json(featuresPayload);
      if (url.endsWith("/api/tsne")) return json(tsneResponse);
      if (url.endsWith("/api/patients")) return json({ detail: "nope" }, 500);
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    const { result } = await readyDashboard(fetchMock);

    await act(async () => {
      await result.current.persistIntake();
    });
    expect(result.current.state.status === "ready" && result.current.state.error).toMatch(/nope/);
    expect(result.current.currentPatientId).toBeNull();
  });

  it("opens a saved patient's results and navigates to the clinician view", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);

    await act(async () => {
      await result.current.openPatientResults(patientRecord, treatmentPlanRecord);
    });

    expect(result.current.patient.demographics.firstName).toBe("Ada");
    expect(result.current.currentPatientId).toBe("patient-42");
    expect(result.current.state.status === "ready" && result.current.state.prediction).not.toBeNull();
    expect(result.current.route).toBe("clinician");
  });

  it("refreshes the explanation and handles refresh failures", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);

    await act(async () => {
      await result.current.runPrediction("patient");
    });

    await act(async () => {
      await result.current.refreshExplanation();
    });
    expect(result.current.state.status === "ready" && result.current.state.explanation).toBe(
      explainResponse.explanation
    );

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/explain")) return json({ detail: "llm down" }, 500);
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    await act(async () => {
      await result.current.refreshExplanation();
    });
    expect(result.current.state.status === "ready" && result.current.state.error).toMatch(/llm down/);
  });

  it("computes a clinician impact from a simulation", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.runPrediction("clinician");
    });

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/predict")) {
        return json({
          ...predictionResponse,
          prediction: { ...predictionResponse.prediction, probability_resistance: 0.4 }
        });
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    act(() => result.current.setClinicianSimValues({ sertraline_mg: 200, lithium_mg: 0, quetiapine_mg: 0, adherence_pct: 90 }));
    await act(async () => {
      await result.current.runClinicianImpactPrediction();
    });
    expect(result.current.clinicianImpactPct).toBeCloseTo(28, 5);
    expect(result.current.isClinicianImpactSubmitting).toBe(false);
  });

  it("records a simulation error", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.runPrediction("clinician");
    });

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/predict")) return json({ detail: "sim boom" }, 500);
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    await act(async () => {
      await result.current.runClinicianImpactPrediction();
    });
    expect(result.current.clinicianImpactError).toMatch(/sim boom/);
  });

  it("runs a scenario impact and applies a scenario to the page", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.runPrediction("clinician");
    });

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/predict")) {
        return json({
          ...predictionResponse,
          features: { ...predictionResponse.features, sertraline_mg: 200 },
          prediction: { ...predictionResponse.prediction, probability_resistance: 0.5 }
        });
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    await act(async () => {
      await result.current.runScenarioImpact({ ...predictionResponse.features, sertraline_mg: 200 });
    });
    expect(result.current.clinicianImpactPct).toBeCloseTo(18, 5);

    const scenario = makeCompletePatient();
    scenario.clinical.sertraline_mg = 200;
    await act(async () => {
      await result.current.applyScenario(scenario);
    });
    expect(result.current.clinicianAppliedImpactPct).toBeCloseTo(18, 5);
    expect(result.current.patient.clinical.sertraline_mg).toBe(200);
    expect(result.current.state.status === "ready" && result.current.state.prediction).not.toBeNull();
  });

  it("applies the raw clinician simulation to the page", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.runPrediction("clinician");
    });

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/api/predict")) {
        return json({ ...predictionResponse, features: { ...predictionResponse.features, lithium_mg: 300 } });
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });

    act(() => result.current.setClinicianSimValues({ sertraline_mg: 100, lithium_mg: 300, quetiapine_mg: 0, adherence_pct: 80 }));
    await act(async () => {
      await result.current.applyClinicianSimulationToPage();
    });
    expect(result.current.patient.clinical.lithium_mg).toBe(300);
  });

  it("saves the current regimen as a treatment plan", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.persistIntake();
    });

    let plan: Awaited<ReturnType<typeof result.current.saveCurrentRegimenAsPlan>> = null;
    await act(async () => {
      plan = await result.current.saveCurrentRegimenAsPlan();
    });
    expect((plan as { id: string } | null)?.id).toBe("plan-1");
    expect(result.current.lastSavedPlan?.id).toBe("plan-1");
    expect(result.current.isPlanSaving).toBe(false);
  });

  it("returns null when saving a plan without a patient", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    let plan: Awaited<ReturnType<typeof result.current.saveCurrentRegimenAsPlan>> = "x" as never;
    await act(async () => {
      plan = await result.current.saveCurrentRegimenAsPlan();
    });
    expect(plan).toBeNull();
  });

  it("records a plan save error", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.persistIntake();
    });

    fetchMock.mockImplementation(async (url: string) => {
      if (url.endsWith("/treatment-plans")) return json({ detail: "plan boom" }, 500);
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    await act(async () => {
      await result.current.saveCurrentRegimenAsPlan();
    });
    expect(result.current.planSaveError).toMatch(/plan boom/);
    expect(result.current.lastSavedPlan).toBeNull();
  });

  it("resets the patient and clears results", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    await act(async () => {
      await result.current.runPrediction("patient");
    });
    expect(result.current.state.status === "ready" && result.current.state.prediction).not.toBeNull();

    act(() => result.current.patientApi.resetPatient());
    expect(result.current.patient.demographics.firstName).toBeNull();
    expect(result.current.state.status === "ready" && result.current.state.prediction).toBeNull();
  });

  it("signs out and redirects to the patient list", async () => {
    const fetchMock = makeFetchMock();
    const { result } = await readyDashboard(fetchMock);
    act(() => result.current.navigate("scientist"));

    act(() => result.current.signOut());
    expect(result.current.route).toBe("patients");
    expect(result.current.state.status).toBe("loading");
    expect(window.location.pathname).toBe("/");
  });
});
