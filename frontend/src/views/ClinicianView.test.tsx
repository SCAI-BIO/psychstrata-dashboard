import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeDashboard, makeReadyState, predictionResponse, treatmentPlanRecord } from "../test/fixtures";
import { ClinicianView } from "./ClinicianView";

// Isolate ClinicianView from the heavy chart + slider subtrees.
const sim = vi.hoisted(() => ({ props: {} as Record<string, unknown> }));

vi.mock("../components/SimPanel", () => ({
  SimControls: (props: Record<string, unknown>) => {
    sim.props = props;
    return (
      <button type="button" data-testid="sim-controls" onClick={() => (props.onChange as (p: object) => void)({ sertraline_mg: 250 })}>
        sim
      </button>
    );
  },
  PredictedImpact: ({ impactPct, error }: { impactPct: number; error?: string | null }) => (
    <div data-testid="predicted-impact" data-impact={impactPct} data-error={error ?? ""} />
  )
}));
vi.mock("../components/charts/ShapChart", () => ({ ShapChart: () => <div data-testid="shap-chart" /> }));
vi.mock("../components/charts/TimeToEventChart", () => ({
  TimeToEventChart: ({ simulatedProbability }: { simulatedProbability?: number | null }) => (
    <div data-testid="tte-chart" data-sim={simulatedProbability ?? "none"} />
  )
}));
vi.mock("../components/cards/InsightsCard", () => ({ InsightsCard: () => <div data-testid="insights-card" /> }));

describe("ClinicianView", () => {
  function renderClinicianView(dashboard = makeDashboard()) {
    render(<ClinicianView dashboard={dashboard} ready={makeReadyState()} prediction={predictionResponse} />);
    return dashboard;
  }

  it("renders a save button next to Calculate", () => {
    renderClinicianView();
    expect(screen.getByRole("button", { name: "Calculate" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save as new treatment plan" })).toBeInTheDocument();
  });

  it("applies the simulation to the page when Calculate is clicked", () => {
    const applyScenario = vi.fn();
    const dashboard = makeDashboard({ applyScenario });
    renderClinicianView(dashboard);
    fireEvent.click(screen.getByRole("button", { name: "Calculate" }));
    expect(applyScenario).toHaveBeenCalledTimes(1);
    expect(applyScenario.mock.calls[0][0]).toMatchObject({ clinical: expect.any(Object) });
  });

  it("disables Calculate while a simulation is running", () => {
    renderClinicianView(makeDashboard({ isClinicianImpactSubmitting: true }));
    expect(screen.getByRole("button", { name: "Calculating…" })).toBeDisabled();
  });

  it("forwards sim slider changes through the scenario draft", () => {
    renderClinicianView();
    fireEvent.click(screen.getByTestId("sim-controls"));
    const values = (sim.props.values as { sertraline_mg: number });
    expect(values.sertraline_mg).toBe(250);
  });

  it("renders the predicted impact and error", () => {
    renderClinicianView(makeDashboard({ clinicianAppliedImpactPct: 12.5, clinicianApplyError: "apply boom" }));
    const impact = screen.getByTestId("predicted-impact");
    expect(impact).toHaveAttribute("data-impact", "12.5");
    expect(impact).toHaveAttribute("data-error", "apply boom");
  });

  it("passes a simulated probability to the time-to-event chart only when impact is non-zero", () => {
    renderClinicianView(makeDashboard({ clinicianImpactPct: 0 }));
    expect(screen.getByTestId("tte-chart")).toHaveAttribute("data-sim", "none");
  });

  it("passes a reduced simulated probability when impact is non-zero", () => {
    renderClinicianView(makeDashboard({ clinicianImpactPct: 10 }));
    expect(screen.getByTestId("tte-chart").getAttribute("data-sim")).toMatch(/^0\.58/);
  });

  it("calls saveCurrentRegimenAsPlan when the save button is clicked", () => {
    const saveCurrentRegimenAsPlan = vi.fn(async () => treatmentPlanRecord);
    const dashboard = makeDashboard({ currentPatientId: "patient-1", saveCurrentRegimenAsPlan });
    renderClinicianView(dashboard);
    fireEvent.click(screen.getByRole("button", { name: "Save as new treatment plan" }));
    expect(saveCurrentRegimenAsPlan).toHaveBeenCalledTimes(1);
  });

  it("disables the save button while a plan is being saved", () => {
    const dashboard = makeDashboard({ isPlanSaving: true });
    renderClinicianView(dashboard);
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
  });

  it("shows an inline confirmation after a plan is saved", () => {
    const dashboard = makeDashboard({ lastSavedPlan: treatmentPlanRecord });
    renderClinicianView(dashboard);
    expect(screen.getByText("Saved as a new treatment plan (starts today).")).toBeInTheDocument();
  });

  it("shows an inline error when saving fails", () => {
    const dashboard = makeDashboard({ planSaveError: "Backend API error (500): boom" });
    renderClinicianView(dashboard);
    expect(screen.getByText("Backend API error (500): boom")).toBeInTheDocument();
  });
});
