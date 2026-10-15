import { renderHook, waitFor, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useTreatmentPlans } from "./useTreatmentPlans";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, fetchTreatmentPlans: vi.fn() };
});

import { fetchTreatmentPlans } from "../api";

const mockedFetch = vi.mocked(fetchTreatmentPlans);

const plan = {
  id: "plan-1",
  patient_id: "patient-1",
  clinician_id: "clinician-1",
  start_date: "2026-10-01",
  end_date: null,
  medications: { sertraline_mg: 100, lithium_mg: 0, quetiapine_mg: 0 },
  adherence: { adherence_pct: 80 },
  created_at: "2026-10-01T10:00:00Z",
  updated_at: "2026-10-01T10:00:00Z"
};

describe("useTreatmentPlans", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("starts loading and resolves to ready", async () => {
    mockedFetch.mockResolvedValue([plan]);

    const { result } = renderHook(() => useTreatmentPlans("patient-1"));

    expect(result.current.state).toEqual({ status: "loading" });
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state).toEqual({ status: "ready", plans: [plan] });
    expect(mockedFetch).toHaveBeenCalledWith("patient-1");
  });

  it("surfaces Error rejection messages", async () => {
    mockedFetch.mockRejectedValue(new Error("kaboom"));

    const { result } = renderHook(() => useTreatmentPlans("patient-1"));

    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", message: "kaboom" });
  });

  it("falls back for non-Error rejections", async () => {
    mockedFetch.mockRejectedValue(42);

    const { result } = renderHook(() => useTreatmentPlans("patient-1"));

    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", message: "Unable to load treatment plans." });
  });

  it("re-fetches when patientId changes", async () => {
    mockedFetch.mockResolvedValue([plan]);

    const { result, rerender } = renderHook(({ id }) => useTreatmentPlans(id), {
      initialProps: { id: "patient-1" }
    });
    await waitFor(() => expect(result.current.state.status).toBe("ready"));

    rerender({ id: "patient-2" });

    await waitFor(() => expect(mockedFetch).toHaveBeenCalledWith("patient-2"));
  });

  it("reload re-fetches", async () => {
    mockedFetch.mockResolvedValue([plan]);

    const { result } = renderHook(() => useTreatmentPlans("patient-1"));
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(mockedFetch).toHaveBeenCalledTimes(1);

    act(() => result.current.reload());

    await waitFor(() => expect(mockedFetch).toHaveBeenCalledTimes(2));
  });
});
