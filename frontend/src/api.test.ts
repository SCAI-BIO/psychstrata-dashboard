import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  buildBasicAuthHeader,
  fetchFeatures,
  fetchPredict,
  setBasicAuthHeader,
  verifyBasicAuth
} from "./api";
import { featuresPayload, predictionResponse } from "./test/fixtures";

type FetchArgs = [string, RequestInit | undefined];

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

function authHeaderOf(init: RequestInit | undefined): string | null {
  return new Headers(init?.headers).get("Authorization");
}

describe("buildBasicAuthHeader", () => {
  it("base64-encodes username:password", () => {
    expect(buildBasicAuthHeader("test-user", "test-password")).toBe(`Basic ${btoa("test-user:test-password")}`);
  });
});

describe("auth header propagation", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    setBasicAuthHeader(null);
    fetchMock = vi.fn(async () => jsonResponse(featuresPayload));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setBasicAuthHeader(null);
  });

  it("omits Authorization when no header is set", async () => {
    await fetchFeatures();
    const [, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(authHeaderOf(init)).toBeNull();
  });

  it("attaches the stored header to subsequent requests", async () => {
    const header = buildBasicAuthHeader("test-user", "test-password");
    setBasicAuthHeader(header);
    await fetchFeatures();
    const [, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(authHeaderOf(init)).toBe(header);
  });

  it("stops attaching the header once cleared", async () => {
    setBasicAuthHeader(buildBasicAuthHeader("test-user", "test-password"));
    setBasicAuthHeader(null);
    await fetchFeatures();
    const [, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(authHeaderOf(init)).toBeNull();
  });

  it("verifyBasicAuth sends its explicit header even with none stored", async () => {
    const header = buildBasicAuthHeader("test-user", "test-password");
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 200 }));
    await verifyBasicAuth(header);
    const [url, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/auth/login");
    expect(init?.method).toBe("POST");
    expect(authHeaderOf(init)).toBe(header);
  });

  it("fetchPredict posts JSON with a content-type header", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(predictionResponse));
    await fetchPredict({ features: { phq9: 12 }, confidence_level: 95 });
    const [url, init] = fetchMock.mock.calls[0] as FetchArgs;
    expect(url).toBe("/api/predict");
    expect(init?.method).toBe("POST");
    expect(new Headers(init?.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(init?.body as string)).toEqual({ features: { phq9: 12 }, confidence_level: 95 });
  });
});

describe("error handling", () => {
  beforeEach(() => setBasicAuthHeader(null));
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces the backend detail on a generic error status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "Invalid credentials." }, 401)));
    await expect(fetchFeatures()).rejects.toThrow("Backend API error (401): Invalid credentials.");
  });

  it("passes through the detail verbatim for 503 (demo budget) responses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "Daily demo budget reached." }, 503)));
    await expect(fetchFeatures()).rejects.toThrow("Daily demo budget reached.");
  });

  it("falls back to a status-only message when no detail is present", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchFeatures()).rejects.toThrow("Backend API request failed with status 500");
  });
});
