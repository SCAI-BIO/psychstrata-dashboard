import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PatientProvider } from "../context/PatientContext";
import { createDefaultPatient } from "../domain/patient";
import type { PatientRead } from "../api";
import { featuresPayload, makeCompletePatient, makeDashboard, makeReadyState } from "../test/fixtures";
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

  it("advances through the wizard and reports progress", () => {
    renderIntake(makeDashboard({ state: makeReadyState() }));

    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    expect(screen.getByText("Clinical Information")).toBeInTheDocument();
    expect(screen.getByText("33% Complete")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    expect(screen.getByText("Genomic Data")).toBeInTheDocument();
    expect(screen.getByText("Proteomic Data")).toBeInTheDocument();
    expect(screen.getByText("66% Complete")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Back/ }));
    expect(screen.getByText("Clinical Information")).toBeInTheDocument();
  });

  it("routes gender selection through writeFeature", () => {
    const dashboard = makeDashboard({ state: makeReadyState() });
    renderIntake(dashboard);
    fireEvent.change(screen.getByLabelText(/Gender at Birth/), { target: { value: "Female" } });
    expect(dashboard.patientApi.updateDemographics).toHaveBeenCalledWith({ gender: "Female" });
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("sex_at_birth", 1);
  });

  it("edits clinical fields and toggles medication on the clinical step", () => {
    const dashboard = makeDashboard({ state: makeReadyState(), patient: makeCompletePatient({ onMedication: true }) });
    renderIntake(dashboard);
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));

    fireEvent.change(screen.getByLabelText(/PHQ-9 Score/), { target: { value: "20" } });
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("phq9", 20);

    fireEvent.click(screen.getAllByRole("radio")[0]);
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("comorbid_anxiety", 1);

    const toggle = screen.getByRole("switch");
    fireEvent.click(toggle);
    expect(dashboard.patientApi.updateProfile).toHaveBeenCalledWith({ onMedication: false });
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("sertraline_mg", 0);
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("quetiapine_mg", 0);
    expect(dashboard.patientApi.writeFeature).toHaveBeenCalledWith("lithium_mg", 0);
  });

  it("toggles genomics and proteomics availability on the last step", () => {
    const dashboard = makeDashboard({
      state: makeReadyState(),
      patient: makeCompletePatient({ genetics: { available: false }, proteomics: { available: false } })
    });
    renderIntake(dashboard);
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));

    const toggles = screen.getAllByRole("switch");
    fireEvent.click(toggles[0]);
    expect(dashboard.patientApi.updateGenetics).toHaveBeenCalledWith({ available: true });
    fireEvent.click(toggles[1]);
    expect(dashboard.patientApi.updateProteomics).toHaveBeenCalledWith({ available: true });
  });

  it("submits the intake and runs the prediction", async () => {
    const persistIntake = vi.fn(async () => ({ id: "patient-1" }) as unknown as PatientRead);
    const runPrediction = vi.fn(async () => {});
    const dashboard = makeDashboard({ state: makeReadyState(), persistIntake, runPrediction });
    renderIntake(dashboard);

    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Calculate Risk/ }));

    await vi.waitFor(() => expect(runPrediction).toHaveBeenCalledWith("clinician"));
    expect(persistIntake).toHaveBeenCalledTimes(1);
  });

  it("does not run a prediction when intake is not persisted", async () => {
    const persistIntake = vi.fn(async () => null);
    const runPrediction = vi.fn(async () => {});
    const dashboard = makeDashboard({ state: makeReadyState(), persistIntake, runPrediction });
    renderIntake(dashboard);

    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Calculate Risk/ }));

    await vi.waitFor(() => expect(persistIntake).toHaveBeenCalled());
    expect(runPrediction).not.toHaveBeenCalled();
  });

  it("shows a submit error and disables the calculate button while submitting", () => {
    renderIntake(makeDashboard({ state: makeReadyState({ isSubmitting: true, error: "backend exploded" }) }));
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next Step/ }));
    expect(screen.getByText("backend exploded")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Calculating…/ })).toBeDisabled();
  });
});
