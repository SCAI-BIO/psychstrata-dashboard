import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { makeDashboard, treatmentPlanRecord } from "../test/fixtures";
import { PatientListView } from "./PatientListView";

const h = vi.hoisted(() => {
  const base = {
    id: "plan-x",
    patient_id: "patient-1",
    clinician_id: "clinician-1",
    start_date: "2026-10-01",
    end_date: null,
    medications: { sertraline_mg: 150, lithium_mg: 0, quetiapine_mg: 0 },
    adherence: { adherence_pct: 90 },
    created_at: "2026-10-01T10:00:00Z",
    updated_at: "2026-10-01T10:00:00Z"
  };
  return {
    patientsResult: { current: {} as Record<string, unknown> },
    plansResult: { current: {} as Record<string, unknown> },
    patientsReload: vi.fn(),
    plansReload: vi.fn(),
    fetchTreatmentPlans: vi.fn(),
    patient: {
      id: "patient-1",
      clinician_id: "clinician-1",
      first_name: "John",
      last_name: "Doe",
      clinical_data: {
        date_of_birth: "1978-05-12",
        diagnosis: "F33.1 — Recurrent depressive disorder",
        clinical_features: {},
        genetics: {},
        proteomics: {}
      },
      created_at: "2026-09-01T10:00:00Z",
      updated_at: "2026-09-01T10:00:00Z"
    },
    planA: { ...base, id: "plan-a", start_date: "2026-01-01", created_at: "2026-01-01T10:00:00Z" },
    planB: { ...base, id: "plan-b", start_date: "2026-02-01", created_at: "2026-02-01T10:00:00Z" }
  };
});

vi.mock("../hooks/usePatients", () => ({
  usePatients: () => ({ ...h.patientsResult.current, reload: h.patientsReload })
}));

vi.mock("../hooks/useTreatmentPlans", () => ({
  useTreatmentPlans: () => ({ ...h.plansResult.current, reload: h.plansReload })
}));

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  fetchTreatmentPlans: h.fetchTreatmentPlans
}));

const readyPatients = { state: { status: "ready", patients: [h.patient] }, isRefreshing: false };
const readyPlans = { state: { status: "ready", plans: [h.planB, h.planA] } };

beforeEach(() => {
  h.patientsResult.current = readyPatients;
  h.plansResult.current = readyPlans;
  h.patientsReload.mockClear();
  h.plansReload.mockClear();
  h.fetchTreatmentPlans.mockReset();
});

function renderView(dashboard = makeDashboard()) {
  render(<PatientListView dashboard={dashboard} />);
  return dashboard;
}

describe("PatientListView", () => {
  it("lists plans oldest first", () => {
    renderView();
    const first = screen.getByText(/Jan 1, 2026 → Ongoing/);
    const second = screen.getByText(/Feb 1, 2026 → Ongoing/);
    expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("marks the newest plan as Latest", () => {
    renderView();
    const latest = screen.getByText("Latest");
    expect(latest.closest("li")).toHaveTextContent(/Feb 1, 2026/);
  });

  it("renders the diagnosis code and description", () => {
    renderView();
    expect(screen.getByText("F33.1")).toBeInTheDocument();
    expect(screen.getByText("Recurrent depressive disorder")).toBeInTheDocument();
    expect(screen.getByText("1 patient in the database.")).toBeInTheDocument();
  });

  it("shows a loading notice while the dashboard is loading", () => {
    renderView(makeDashboard({ state: { status: "loading" } }));
    expect(screen.getByText("Loading clinical model configuration…")).toBeInTheDocument();
  });

  it("shows a backend error notice", () => {
    renderView(makeDashboard({ state: { status: "error", message: "connection refused" } }));
    expect(screen.getByText("Backend unavailable.")).toBeInTheDocument();
    expect(screen.getByText(/connection refused/)).toBeInTheDocument();
  });

  it("renders a sign-out button when auth is required", () => {
    const signOut = vi.fn();
    renderView(makeDashboard({ authRequired: true, signOut }));
    fireEvent.click(screen.getByRole("button", { name: /Sign out/ }));
    expect(signOut).toHaveBeenCalledTimes(1);
  });

  it("navigates to intake from Add New Patient", () => {
    const navigate = vi.fn();
    renderView(makeDashboard({ navigate }));
    fireEvent.click(screen.getByRole("button", { name: /Add New Patient/ }));
    expect(navigate).toHaveBeenCalledWith("intake");
  });

  it("refreshes the patient list", () => {
    renderView();
    fireEvent.click(screen.getAllByRole("button", { name: /Refresh/ })[0]);
    expect(h.patientsReload).toHaveBeenCalledTimes(1);
  });

  it("shows loading and empty notices for the patient list", () => {
    h.patientsResult.current = { state: { status: "loading" }, isRefreshing: false };
    renderView();
    expect(screen.getByText("Loading patients…")).toBeInTheDocument();

    h.patientsResult.current = { state: { status: "ready", patients: [] }, isRefreshing: false };
    renderView();
    expect(screen.getByText(/No patients yet/)).toBeInTheDocument();
  });

  it("shows an error notice when patients fail to load", () => {
    h.patientsResult.current = { state: { status: "error", message: "nope" }, isRefreshing: false };
    renderView();
    expect(screen.getByText("Couldn't load patients.")).toBeInTheDocument();
    expect(screen.getByText(/nope/)).toBeInTheDocument();
  });

  it("shows loading and empty notices for treatment plans", () => {
    h.plansResult.current = { state: { status: "loading" } };
    renderView();
    expect(screen.getByText("Loading treatment plans…")).toBeInTheDocument();

    h.plansResult.current = { state: { status: "ready", plans: [] } };
    renderView();
    expect(screen.getByText(/No treatment plans for this patient yet/)).toBeInTheDocument();
  });

  it("opens the latest plan when View Results is clicked", async () => {
    h.fetchTreatmentPlans.mockResolvedValue([h.planB, h.planA]);
    const openPatientResults = vi.fn(async () => {});
    renderView(makeDashboard({ openPatientResults }));
    fireEvent.click(screen.getByRole("button", { name: /View Results/ }));
    await waitFor(() => expect(openPatientResults).toHaveBeenCalledWith(h.patient, h.planB));
    expect(h.fetchTreatmentPlans).toHaveBeenCalledWith("patient-1");
  });

  it("surfaces an error when opening results fails", async () => {
    h.fetchTreatmentPlans.mockRejectedValue(new Error("latest plan boom"));
    renderView();
    fireEvent.click(screen.getByRole("button", { name: /View Results/ }));
    expect(await screen.findByText("latest plan boom")).toBeInTheDocument();
  });

  it("opens a specific plan when a plan row is clicked", async () => {
    const openPatientResults = vi.fn(async () => {});
    renderView(makeDashboard({ openPatientResults }));
    fireEvent.click(screen.getByRole("button", { name: /Jan 1, 2026 → Ongoing/ }));
    await waitFor(() => expect(openPatientResults).toHaveBeenCalledWith(h.patient, h.planA));
  });

  it("collapses the row details when the row is clicked again", () => {
    renderView();
    expect(screen.getByText("Treatment plans")).toBeInTheDocument();
    fireEvent.click(screen.getByText("John Doe"));
    expect(screen.queryByText("Treatment plans")).not.toBeInTheDocument();
  });

  it("reloads treatment plans from the details panel", () => {
    renderView();
    fireEvent.click(screen.getAllByRole("button", { name: /Refresh/ })[1]);
    expect(h.plansReload).toHaveBeenCalledTimes(1);
  });

  it("shows a treatment-plan error message", () => {
    h.plansResult.current = { state: { status: "error", message: "plans boom" } };
    renderView();
    expect(screen.getByText("plans boom")).toBeInTheDocument();
  });

  it("falls back to the default medication text when a plan has no active medication", () => {
    h.plansResult.current = {
      state: {
        status: "ready",
        plans: [{ ...treatmentPlanRecord, medications: { sertraline_mg: 0, lithium_mg: 0, quetiapine_mg: 0 } }]
      }
    };
    renderView();
    expect(screen.getByText(/No active medication/)).toBeInTheDocument();
  });
});
