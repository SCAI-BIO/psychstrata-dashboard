import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makePatientApi } from "../test/fixtures";
import { PatientProvider, usePatient } from "./PatientContext";

describe("PatientContext", () => {
  it("throws when usePatient is used outside a provider", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => usePatient())).toThrow(/within a PatientProvider/);
    spy.mockRestore();
  });

  it("returns the provided PatientApi", () => {
    const api = makePatientApi();

    const { result } = renderHook(() => usePatient(), {
      wrapper: ({ children }) => <PatientProvider value={api}>{children}</PatientProvider>
    });

    expect(result.current).toBe(api);
    expect(result.current.patient).toBe(api.patient);
    expect(result.current.readFeature("phq9")).toBe(api.patient.clinical.phq9);
  });
});
