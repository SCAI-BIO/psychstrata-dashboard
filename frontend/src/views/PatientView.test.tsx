import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PatientProvider } from "../context/PatientContext";
import { makeDashboard, makePatientApi, makeCompletePatient, makeReadyState, predictionResponse } from "../test/fixtures";
import { PatientView } from "./PatientView";

// Isolate PatientView from the heavy chart + slider subtrees.
vi.mock("../components/SimPanel", () => ({ SimControls: () => <div data-testid="sim-controls" /> }));
vi.mock("../components/charts/TimeToEventChart", () => ({ TimeToEventChart: () => <div data-testid="tte-chart" /> }));

describe("PatientView", () => {
  function renderPatientView() {
    const patient = makeCompletePatient();
    const patientApi = makePatientApi(patient);
    const dashboard = makeDashboard({ patient, patientApi });
    render(
      <PatientProvider value={patientApi}>
        <PatientView dashboard={dashboard} ready={makeReadyState()} prediction={predictionResponse} />
      </PatientProvider>
    );
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
});
