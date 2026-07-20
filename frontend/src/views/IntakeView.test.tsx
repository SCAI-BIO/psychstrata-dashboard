import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PatientProvider } from "../context/PatientContext";
import { createDefaultPatient } from "../domain/patient";
import { featuresPayload, makeDashboard, makeReadyState } from "../test/fixtures";
import { IntakeView } from "./IntakeView";

function renderIntake(dashboard = makeDashboard()) {
  return render(
    <PatientProvider value={dashboard.patientApi}>
      <IntakeView dashboard={dashboard} />
    </PatientProvider>
  );
}

describe("IntakeView", () => {
  it("shows a loading message while the model config is loading", () => {
    renderIntake(makeDashboard({ state: { status: "loading" } }));
    expect(screen.getByText(/Loading clinical model configuration/i)).toBeInTheDocument();
  });

  it("shows a backend error state", () => {
    renderIntake(makeDashboard({ state: { status: "error", message: "boom" } }));
    expect(screen.getByText(/Backend unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it("renders the first wizard step when ready", () => {
    renderIntake(makeDashboard({ state: makeReadyState() }));
    expect(screen.getByRole("heading", { name: "New Patient" })).toBeInTheDocument();
    expect(screen.getByText("Core Demographics")).toBeInTheDocument();
  });

  it("enables 'Next Step' only once the step's required fields are complete", () => {
    const { unmount } = renderIntake(
      makeDashboard({ state: makeReadyState(), patient: createDefaultPatient(featuresPayload.defaults) })
    );
    expect(screen.getByRole("button", { name: /Next Step/ })).toBeDisabled();
    unmount();

    // makeDashboard()'s default patient has complete demographics.
    renderIntake(makeDashboard({ state: makeReadyState() }));
    expect(screen.getByRole("button", { name: /Next Step/ })).not.toBeDisabled();
  });

  it("shows the sign-out control only when auth is required", () => {
    const { unmount } = renderIntake(makeDashboard({ state: makeReadyState(), authRequired: false }));
    expect(screen.queryByRole("button", { name: /Sign out/i })).not.toBeInTheDocument();
    unmount();

    renderIntake(makeDashboard({ state: makeReadyState(), authRequired: true }));
    expect(screen.getByRole("button", { name: /Sign out/i })).toBeInTheDocument();
  });
});
