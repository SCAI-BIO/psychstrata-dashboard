import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeDashboard, makeReadyState, predictionResponse, treatmentPlanRecord } from "../test/fixtures";
import { ClinicianView } from "./ClinicianView";

// Isolate ClinicianView from the heavy chart + slider subtrees.
vi.mock("../components/SimPanel", () => ({
  SimControls: () => <div data-testid="sim-controls" />,
  PredictedImpact: () => <div data-testid="predicted-impact" />
}));
vi.mock("../components/charts/ShapChart", () => ({ ShapChart: () => <div data-testid="shap-chart" /> }));
vi.mock("../components/charts/TimeToEventChart", () => ({ TimeToEventChart: () => <div data-testid="tte-chart" /> }));
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
