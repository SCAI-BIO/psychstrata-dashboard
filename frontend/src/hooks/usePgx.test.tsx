import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PgxInterpretResponse } from "../api";
import { pgxInterpretResponse, pgxOptionsResponse } from "../test/fixtures";
import { usePgxInterpretation, usePgxOptions } from "./usePgx";

vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ...actual, fetchPgxOptions: vi.fn(), fetchPgxInterpretation: vi.fn() };
});

import { fetchPgxInterpretation, fetchPgxOptions } from "../api";

const mockedFetchOptions = vi.mocked(fetchPgxOptions);
const mockedFetchInterpretation = vi.mocked(fetchPgxInterpretation);

describe("usePgxOptions", () => {
  afterEach(() => vi.clearAllMocks());

  it("starts loading and exposes the fetched options", async () => {
    mockedFetchOptions.mockResolvedValue(pgxOptionsResponse);

    const { result } = renderHook(() => usePgxOptions());
    expect(result.current.state).toEqual({ status: "loading" });

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.state).toEqual({ status: "ready", options: pgxOptionsResponse });
  });

  it("surfaces loading errors", async () => {
    mockedFetchOptions.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => usePgxOptions());

    await waitFor(() => expect(result.current.state.status).toBe("error"));
    expect(result.current.state).toEqual({ status: "error", message: "boom" });
  });

  it("reload refetches the catalogue", async () => {
    mockedFetchOptions.mockResolvedValue(pgxOptionsResponse);

    const { result } = renderHook(() => usePgxOptions());
    await waitFor(() => expect(result.current.state.status).toBe("ready"));

    act(() => result.current.reload());

    await waitFor(() => expect(mockedFetchOptions).toHaveBeenCalledTimes(2));
  });
});

describe("usePgxInterpretation", () => {
  afterEach(() => vi.clearAllMocks());

  const wildType = { cyp2c19: { allele1: "*1", allele2: "*1" }, cyp2d6: { allele1: "*1", allele2: "*1" } };

  it("is idle when genetics is unavailable", () => {
    const { result } = renderHook(() => usePgxInterpretation({ available: false, ...wildType }));
    expect(result.current).toEqual({ status: "idle" });
    expect(mockedFetchInterpretation).not.toHaveBeenCalled();
  });

  it("is idle when no diplotype is present", () => {
    const { result } = renderHook(() => usePgxInterpretation({ available: true }));
    expect(result.current).toEqual({ status: "idle" });
    expect(mockedFetchInterpretation).not.toHaveBeenCalled();
  });

  it("interprets the provided diplotypes", async () => {
    mockedFetchInterpretation.mockResolvedValue(pgxInterpretResponse);

    const { result } = renderHook(() => usePgxInterpretation({ available: true, ...wildType }));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current).toEqual({ status: "ready", result: pgxInterpretResponse });
    expect(mockedFetchInterpretation).toHaveBeenCalledWith(wildType);
  });

  it("sends only the genes that have a diplotype", async () => {
    mockedFetchInterpretation.mockResolvedValue(pgxInterpretResponse);

    const { result } = renderHook(() =>
      usePgxInterpretation({ available: true, cyp2c19: { allele1: "*1", allele2: "*2" } })
    );

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(mockedFetchInterpretation).toHaveBeenCalledWith({ cyp2c19: { allele1: "*1", allele2: "*2" } });
  });

  it("refetches when an allele changes", async () => {
    mockedFetchInterpretation.mockResolvedValue(pgxInterpretResponse);

    const { result, rerender } = renderHook(
      ({ genetics }: { genetics: Parameters<typeof usePgxInterpretation>[0] }) =>
        usePgxInterpretation(genetics),
      { initialProps: { genetics: { available: true, ...wildType } } }
    );
    await waitFor(() => expect(result.current.status).toBe("ready"));

    const changed = { available: true, ...wildType, cyp2c19: { allele1: "*1", allele2: "*2" } };
    rerender({ genetics: changed });

    await waitFor(() => expect(mockedFetchInterpretation).toHaveBeenCalledTimes(2));
    expect(mockedFetchInterpretation).toHaveBeenLastCalledWith({
      cyp2c19: { allele1: "*1", allele2: "*2" },
      cyp2d6: { allele1: "*1", allele2: "*1" }
    });
  });

  it("surfaces interpretation errors", async () => {
    mockedFetchInterpretation.mockRejectedValue(new Error("invalid allele"));

    const { result } = renderHook(() => usePgxInterpretation({ available: true, ...wildType }));

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current).toEqual({ status: "error", message: "invalid allele" });
  });

  it("does not update state after unmount", async () => {
    let resolve: (value: PgxInterpretResponse) => void = () => {};
    mockedFetchInterpretation.mockReturnValue(new Promise((r) => (resolve = r)));

    const { unmount } = renderHook(() => usePgxInterpretation({ available: true, ...wildType }));
    unmount();

    await act(async () => {
      resolve(pgxInterpretResponse);
    });

    expect(mockedFetchInterpretation).toHaveBeenCalledTimes(1);
  });
});
