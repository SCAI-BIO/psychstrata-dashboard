import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { setBasicAuthHeader } from "./api";
import { explainResponse, featuresPayload, predictionResponse, tsneResponse } from "./test/fixtures";

const USERNAME = "dashboard-user";
const PASSWORD = "test-password";
const EXPECTED_HEADER = `Basic ${btoa(`${USERNAME}:${PASSWORD}`)}`;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Backend stub: `authEnabled` toggles whether the API demands Basic auth. */
function installFetchMock(authEnabled: boolean) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const auth = new Headers(init?.headers).get("Authorization");
    if (url.endsWith("/api/auth/status")) return json({ auth_enabled: authEnabled });
    if (url.endsWith("/api/auth/login")) {
      return auth === EXPECTED_HEADER ? new Response(null, { status: 200 }) : json({ detail: "Invalid credentials." }, 401);
    }
    if (authEnabled && auth !== EXPECTED_HEADER) return json({ detail: "Invalid credentials." }, 401);
    if (url.endsWith("/api/features")) return json(featuresPayload);
    if (url.endsWith("/api/tsne")) return json(tsneResponse);
    if (url.endsWith("/api/predict")) return json(predictionResponse);
    if (url.endsWith("/api/explain")) return json(explainResponse);
    throw new Error(`Unexpected fetch URL: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    window.history.pushState({}, "", "/");
    setBasicAuthHeader(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setBasicAuthHeader(null);
  });

  it("goes straight to the intake wizard when the backend has auth disabled", async () => {
    installFetchMock(false);
    render(<App />);

    expect(await screen.findByRole("heading", { name: "New Patient" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Dashboard login" })).not.toBeInTheDocument();
  });

  it("gates on login when auth is enabled, then loads the app after a correct sign-in", async () => {
    const fetchMock = installFetchMock(true);
    render(<App />);

    // 1. The login sheet appears; no data has loaded yet.
    expect(await screen.findByRole("heading", { name: "Dashboard login" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.some((c) => (c[0] as string).endsWith("/api/features"))).toBe(false);

    // 2. A wrong password is rejected with the backend's message.
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: USERNAME } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Backend API error (401): Invalid credentials.")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Dashboard login" })).toBeInTheDocument();

    // 3. The correct password logs in and the intake wizard loads.
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: PASSWORD } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByRole("heading", { name: "New Patient" })).toBeInTheDocument();

    // The data load that followed login carried the Basic auth header.
    await waitFor(() => {
      const featuresCall = fetchMock.mock.calls.find((c) => (c[0] as string).endsWith("/api/features"));
      expect(featuresCall).toBeDefined();
      expect(new Headers((featuresCall?.[1] as RequestInit)?.headers).get("Authorization")).toBe(EXPECTED_HEADER);
    });
  });

  it("does not trigger the browser's native basic-auth dialog (no data 401 before login)", async () => {
    const fetchMock = installFetchMock(true);
    render(<App />);
    await screen.findByRole("heading", { name: "Dashboard login" });

    // Before authenticating, only the public status endpoint should have been hit;
    // a 401 from a protected endpoint here is what previously popped the native dialog.
    const preLoginUrls = fetchMock.mock.calls.map((c) => c[0] as string);
    expect(preLoginUrls).toContain("/api/auth/status");
    expect(preLoginUrls).not.toContain("/api/features");
    expect(preLoginUrls).not.toContain("/api/tsne");
  });
});
