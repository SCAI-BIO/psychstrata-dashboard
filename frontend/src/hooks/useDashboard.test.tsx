import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setBasicAuthHeader } from "../api";
import { AUTH_HEADER_STORAGE_KEY } from "../constants";
import { explainResponse, featuresPayload, predictionResponse, tsneResponse } from "../test/fixtures";
import { useDashboard } from "./useDashboard";

const USERNAME = "test-user";
const PASSWORD = "test-password";
const EXPECTED_HEADER = `Basic ${btoa(`${USERNAME}:${PASSWORD}`)}`;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Build a fetch mock whose backend either requires Basic auth or does not. */
function makeFetchMock(authEnabled: boolean) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const auth = new Headers(init?.headers).get("Authorization");
    if (url.endsWith("/api/auth/status")) return json({ auth_enabled: authEnabled });
    if (url.endsWith("/api/auth/login")) {
      return auth === EXPECTED_HEADER ? new Response(null, { status: 200 }) : json({ detail: "Invalid credentials." }, 401);
    }
    // Every data endpoint is protected when auth is enabled.
    if (authEnabled && auth !== EXPECTED_HEADER) return json({ detail: "Invalid credentials." }, 401);
    if (url.endsWith("/api/features")) return json(featuresPayload);
    if (url.endsWith("/api/tsne")) return json(tsneResponse);
    if (url.endsWith("/api/predict")) return json(predictionResponse);
    if (url.endsWith("/api/explain")) return json(explainResponse);
    throw new Error(`Unexpected fetch URL: ${url}`);
  });
}

function calledUrls(fetchMock: ReturnType<typeof vi.fn>): string[] {
  return fetchMock.mock.calls.map((c) => c[0] as string);
}

describe("useDashboard", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    window.sessionStorage.clear();
    window.history.pushState({}, "", "/");
    setBasicAuthHeader(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setBasicAuthHeader(null);
  });

  it("loads data immediately when the backend reports auth is disabled", async () => {
    fetchMock = makeFetchMock(false);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.authRequired).toBe(false);
    expect(result.current.isAuthenticated).toBe(true);
    expect(calledUrls(fetchMock)).toEqual(expect.arrayContaining(["/api/features", "/api/tsne"]));
  });

  it("holds data loading until login when the backend requires auth", async () => {
    fetchMock = makeFetchMock(true);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());

    await waitFor(() => expect(result.current.authRequired).toBe(true));
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.state.status).toBe("loading");
    // Gate holds: no protected data request should have fired yet.
    expect(calledUrls(fetchMock)).not.toContain("/api/features");
  });

  it("propagates the auth header to data requests after a successful login", async () => {
    fetchMock = makeFetchMock(true);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());
    await waitFor(() => expect(result.current.authRequired).toBe(true));

    await act(async () => {
      await result.current.attemptLogin(USERNAME, PASSWORD);
    });

    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(result.current.isAuthenticated).toBe(true);

    // Regression guard: the features request must carry the Basic auth header,
    // otherwise the backend 401s and the dashboard shows "Backend unavailable".
    const featuresCall = fetchMock.mock.calls.find((c) => (c[0] as string).endsWith("/api/features"));
    expect(featuresCall).toBeDefined();
    expect(new Headers((featuresCall?.[1] as RequestInit)?.headers).get("Authorization")).toBe(EXPECTED_HEADER);
  });

  it("surfaces a login error and stays gated on wrong credentials", async () => {
    fetchMock = makeFetchMock(true);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());
    await waitFor(() => expect(result.current.authRequired).toBe(true));

    await act(async () => {
      await result.current.attemptLogin(USERNAME, "wrong-password");
    });

    expect(result.current.loginError).toMatch(/Invalid credentials/);
    expect(result.current.isAuthenticated).toBe(false);
    expect(calledUrls(fetchMock)).not.toContain("/api/features");
  });

  it("runs a prediction and navigates to the requested result route", async () => {
    fetchMock = makeFetchMock(false);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());
    await waitFor(() => expect(result.current.state.status).toBe("ready"));

    await act(async () => {
      await result.current.runPrediction("patient");
    });

    await waitFor(() => {
      expect(result.current.state.status === "ready" && result.current.state.prediction !== null).toBe(true);
    });
    expect(result.current.route).toBe("patient");
    expect(calledUrls(fetchMock)).toContain("/api/predict");
  });

  it("clears auth and resets to loading on sign out", async () => {
    fetchMock = makeFetchMock(true);
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useDashboard());
    await waitFor(() => expect(result.current.authRequired).toBe(true));
    await act(async () => {
      await result.current.attemptLogin(USERNAME, PASSWORD);
    });
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    // The header was persisted on login...
    expect(window.sessionStorage.getItem(AUTH_HEADER_STORAGE_KEY)).toBe(EXPECTED_HEADER);

    act(() => result.current.signOut());

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.state.status).toBe("loading");
    // ...and cleared on sign out.
    expect(window.sessionStorage.getItem(AUTH_HEADER_STORAGE_KEY)).toBeNull();
  });
});
