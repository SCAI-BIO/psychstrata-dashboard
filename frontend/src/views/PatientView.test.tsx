import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PatientProvider } from "../context/PatientContext";
import { makeDashboard, makePatientApi, makeCompletePatient, makeReadyState, predictionResponse } from "../test/fixtures";
import type { DashboardApi } from "../hooks/useDashboard";
import { PatientView } from "./PatientView";

// Isolate PatientView from the heavy chart + slider subtrees.
vi.mock("../components/SimPanel", () => ({ SimControls: () => <div data-testid="sim-controls" /> }));
vi.mock("../components/charts/TimeToEventChart", () => ({ TimeToEventChart: () => <div data-testid="tte-chart" /> }));

describe("PatientView", () => {
  function renderPatientView(overrides: Partial<DashboardApi> = {}) {
    const patient = makeCompletePatient();
    const patientApi = makePatientApi(patient);
    const dashboard = makeDashboard({ patient, patientApi, ...overrides });
    render(
      <PatientProvider value={patientApi}>
        <PatientView dashboard={dashboard} ready={makeReadyState()} prediction={predictionResponse} />
      </PatientProvider>
    );
    return dashboard;
  }

  it("greets the patient by first name", () => {
    renderPatientView();
    expect(screen.getByText("Welcome, John.")).toBeInTheDocument();
  });

  it("renders the estimated treatment response as a percentage", () => {
    renderPatientView();
    // risk 0.68 with zero simulated impact ⇒ 32.0% response.
    expect(screen.getByText("Estimated Treatment Response")).toBeInTheDocument();
    expect(screen.getByText("32.0%")).toBeInTheDocument();
  });

  it("shows a response badge matching the response band", () => {
    renderPatientView();
    expect(screen.getByText("Low Response")).toBeInTheDocument();
  });

  it("renders the strengths and action-item cards", () => {
    renderPatientView();
    expect(screen.getByText("Your Strengths")).toBeInTheDocument();
    expect(screen.getByText("Action Items")).toBeInTheDocument();
  });

  it("calls resetSimulation when Reset is clicked", () => {
    const resetSimulation = vi.fn();
    renderPatientView({ resetSimulation, canResetSimulation: true });
    fireEvent.click(screen.getByRole("button", { name: "Reset to patient" }));
    expect(resetSimulation).toHaveBeenCalledTimes(1);
  });

  it("disables Reset until a baseline exists", () => {
    renderPatientView({ canResetSimulation: false });
    expect(screen.getByRole("button", { name: "Reset to patient" })).toBeDisabled();
  });

  it("disables Reset while a scenario impact is being calculated", () => {
    renderPatientView({ canResetSimulation: true, isClinicianImpactSubmitting: true });
    expect(screen.getByRole("button", { name: "Reset to patient" })).toBeDisabled();
  });
});
