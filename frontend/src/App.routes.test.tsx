import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DashboardApi } from "./hooks/useDashboard";
import { makeDashboard, makeReadyState, predictionResponse } from "./test/fixtures";

const { dashboardRef } = vi.hoisted(() => ({ dashboardRef: { current: null as DashboardApi | null } }));

vi.mock("./hooks/useDashboard", () => ({
  useDashboard: () => dashboardRef.current
}));

vi.mock("./components/VersionBadge", () => ({ VersionBadge: () => <span>v-test</span> }));
vi.mock("./views/LoginView", () => ({ LoginView: () => <div data-testid="login-view" /> }));
vi.mock("./views/IntakeView", () => ({ IntakeView: () => <div data-testid="intake-view" /> }));
vi.mock("./views/PatientListView", () => ({ PatientListView: () => <div data-testid="patient-list-view" /> }));
vi.mock("./views/PatientView", () => ({ PatientView: () => <div data-testid="patient-view" /> }));
vi.mock("./views/ClinicianView", () => ({ ClinicianView: () => <div data-testid="clinician-view" /> }));
vi.mock("./views/ScientistView", () => ({ ScientistView: () => <div data-testid="scientist-view" /> }));

import App from "./App";

function renderWith(overrides: Partial<DashboardApi>) {
  dashboardRef.current = makeDashboard(overrides);
  return render(<App />);
}

describe("App routing", () => {
  afterEach(() => {
    dashboardRef.current = null;
  });

  it("shows the login view when auth is required and the user is not signed in", () => {
    renderWith({ authRequired: true, isAuthenticated: false });
    expect(screen.getByTestId("login-view")).toBeInTheDocument();
  });

  it("routes to the intake wizard", () => {
    renderWith({ route: "intake" });
    expect(screen.getByTestId("intake-view")).toBeInTheDocument();
  });

  it("routes to the patient list", () => {
    renderWith({ route: "patients" });
    expect(screen.getByTestId("patient-list-view")).toBeInTheDocument();
  });

  it("shows a loading message while the result data is loading", () => {
    renderWith({ route: "clinician", state: { status: "loading" } });
    expect(screen.getByText("Loading clinical model configuration…")).toBeInTheDocument();
  });

  it("shows the backend-unavailable message on error", () => {
    renderWith({ route: "clinician", state: { status: "error", message: "connection refused" } });
    expect(screen.getByText("Backend unavailable.")).toBeInTheDocument();
    expect(screen.getByText(/connection refused/)).toBeInTheDocument();
  });

  it("prompts for intake when there is no prediction yet", () => {
    renderWith({ route: "clinician", state: makeReadyState({ prediction: null }) });
    expect(screen.getByText(/No results yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Go to intake" })).toBeInTheDocument();
  });

  it("renders the patient result view", () => {
    renderWith({ route: "patient", state: makeReadyState({ prediction: predictionResponse }) });
    expect(screen.getByTestId("patient-view")).toBeInTheDocument();
  });

  it("renders the clinician result view", () => {
    renderWith({ route: "clinician", state: makeReadyState({ prediction: predictionResponse }) });
    expect(screen.getByTestId("clinician-view")).toBeInTheDocument();
  });

  it("renders the scientist result view", () => {
    renderWith({ route: "scientist", state: makeReadyState({ prediction: predictionResponse }) });
    expect(screen.getByTestId("scientist-view")).toBeInTheDocument();
  });
});
