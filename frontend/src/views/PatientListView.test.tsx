import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeDashboard } from "../test/fixtures";
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
    patient: {
      id: "patient-1",
      clinician_id: "clinician-1",
      first_name: "John",
      last_name: "Doe",
      clinical_data: {
        date_of_birth: "1978-05-12",
        diagnosis: "F33.1",
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
  usePatients: () => ({
    state: { status: "ready", patients: [h.patient] },
    isRefreshing: false,
    reload: vi.fn()
  })
}));

// Deliberately unsorted: the view must sort chronologically.
vi.mock("../hooks/useTreatmentPlans", () => ({
  useTreatmentPlans: () => ({
    state: { status: "ready", plans: [h.planB, h.planA] },
    reload: vi.fn()
  })
}));

describe("PatientListView", () => {
  function renderView() {
    const dashboard = makeDashboard();
    render(<PatientListView dashboard={dashboard} />);
    return { dashboard };
  }

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
});
