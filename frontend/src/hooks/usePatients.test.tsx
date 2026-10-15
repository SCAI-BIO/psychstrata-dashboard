import { renderHook, waitFor, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { usePatients } from "./usePatients";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, fetchPatients: vi.fn() };
});

import { fetchPatients, type PatientRecord } from "../api";

const mockedFetchPatients = vi.mocked(fetchPatients);

const patient = {
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
};

describe("usePatients", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("starts in the loading state", () => {
    mockedFetchPatients.mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => usePatients());

    expect(result.current.state).toEqual({ status: "loading" });
    expect(result.current.isRefreshing).toBe(true);
  });

  it("transitions to ready with the fetched patients", async () => {
    mockedFetchPatients.mockResolvedValue([patient]);

    const { result } = renderHook(() => usePatients());

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state).toEqual({ status: "ready", patients: [patient] });
    expect(result.current.isRefreshing).toBe(false);
  });

  it("surfaces the error message from an Error rejection", async () => {
    mockedFetchPatients.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => usePatients());

    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", message: "boom" });
  });

  it("falls back to a default message for non-Error rejections", async () => {
    mockedFetchPatients.mockRejectedValue("nope");

    const { result } = renderHook(() => usePatients());

    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", message: "Unable to load patients." });
  });

  it("reload re-fetches", async () => {
    mockedFetchPatients.mockResolvedValue([patient]);

    const { result } = renderHook(() => usePatients());
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(mockedFetchPatients).toHaveBeenCalledTimes(1);

    act(() => result.current.reload());

    await waitFor(() => expect(mockedFetchPatients).toHaveBeenCalledTimes(2));
  });

  it("does not update state after unmount", async () => {
    let resolve: (value: PatientRecord[]) => void = () => {};
    mockedFetchPatients.mockReturnValue(new Promise<PatientRecord[]>((r) => (resolve = r)));

    const { unmount } = renderHook(() => usePatients());
    unmount();

    await act(async () => {
      resolve([patient]);
    });

    expect(mockedFetchPatients).toHaveBeenCalledTimes(1);
  });
});
