import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const featuresPayload = {
  features: [
    { id: "phq9", label: "PHQ-9", kind: "numeric", default: 12, min: 0, max: 27, step: 1, params: { min: 0, max: 27, step: 1 } },
    {
      id: "adherence_pct",
      label: "Adherence (%)",
      kind: "numeric",
      default: 80,
      min: 0,
      max: 100,
      step: 1,
      params: { min: 0, max: 100, step: 1 }
    },
    { id: "sertraline_mg", label: "Sertraline", kind: "numeric", default: 100, min: 0, max: 200, step: 5, params: { min: 0, max: 200, step: 5 } },
    { id: "lithium_mg", label: "Lithium", kind: "numeric", default: 0, min: 0, max: 1200, step: 100, params: { min: 0, max: 1200, step: 100 } },
    { id: "quetiapine_mg", label: "Quetiapine", kind: "numeric", default: 0, min: 0, max: 300, step: 25, params: { min: 0, max: 300, step: 25 } }
  ],
  defaults: {
    phq9: 12,
    adherence_pct: 80,
    sertraline_mg: 100,
    lithium_mg: 0,
    quetiapine_mg: 0
  },
  model_feature_order: ["phq9", "adherence_pct", "sertraline_mg", "lithium_mg", "quetiapine_mg"],
  confidence_level: { default: 95, min: 80, max: 99, step: 1 }
};

describe("App", () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  const expectedAuthHeader = "Basic dGVzdC11c2VyOnRlc3QtcGFzc3dvcmQ=";

  const getHeaderValue = (headers: HeadersInit | undefined, key: string): string | null => {
    if (!headers) {
      return null;
    }
    if (headers instanceof Headers) {
      return headers.get(key);
    }
    if (Array.isArray(headers)) {
      const match = headers.find(([headerName]) => headerName.toLowerCase() === key.toLowerCase());
      return match ? match[1] : null;
    }
    const headerValue = headers[key as keyof typeof headers];
    return typeof headerValue === "string" ? headerValue : null;
  };

  beforeEach(() => {
    window.sessionStorage.clear();
    window.history.pushState({}, "", "/");

    fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/api/auth/status")) {
        return new Response(JSON.stringify({ auth_enabled: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        });
      }
      if (url.endsWith("/api/auth/login")) {
        expect(init?.method).toBe("POST");
        const authorizationHeader = getHeaderValue(init?.headers, "Authorization");
        if (authorizationHeader !== expectedAuthHeader) {
          return new Response(JSON.stringify({ detail: "Invalid credentials." }), {
            status: 401,
            headers: { "Content-Type": "application/json" }
          });
        }
        return new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        });
      }
      if (url.endsWith("/api/features")) {
        expect(getHeaderValue(init?.headers, "Authorization")).toBe(expectedAuthHeader);
        return new Response(JSON.stringify(featuresPayload), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        });
      }
      if (url.endsWith("/api/tsne")) {
        expect(getHeaderValue(init?.headers, "Authorization")).toBe(expectedAuthHeader);
        return new Response(
          JSON.stringify({
            points: [
              { x: 0, y: 0, class_value: 0, class_label: "Responsive" },
              { x: 1, y: 1, class_value: 1, class_label: "Resistant" }
            ]
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith("/api/predict")) {
        expect(init?.method).toBe("POST");
        expect(getHeaderValue(init?.headers, "Authorization")).toBe(expectedAuthHeader);
        return new Response(
          JSON.stringify({
            features: featuresPayload.defaults,
            prediction: {
              probability_resistance: 0.684,
              predicted_class: "Resistant",
              conformal_prediction: {
                confidence_level: 95,
                alpha: 0.05,
                label: "Resistant",
                included_classes: ["Resistant"]
              }
            },
            shap_values: [
              {
                feature_id: "phq9",
                feature_label: "PHQ-9",
                selected_value: 12,
                selected_value_label: "12",
                shap_value: 0.3,
                abs_shap_value: 0.3,
                direction: "raises"
              }
            ],
            top_contributors: {
              positive: [{ feature_id: "phq9", feature_label: "PHQ-9", selected_value: "12", shap_value: 0.3, direction: "raises" }],
              negative: [{ feature_id: "adherence_pct", feature_label: "Adherence", selected_value: "80%", shap_value: -0.2, direction: "lowers" }]
            },
            tsne: { selected: { x: 0.2, y: 0.3 } },
            disclaimer: "synthetic"
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.endsWith("/api/explain")) {
        expect(getHeaderValue(init?.headers, "Authorization")).toBe(expectedAuthHeader);
        return new Response(
          JSON.stringify({
            features: featuresPayload.defaults,
            prediction: {
              probability_resistance: 0.684,
              predicted_class: "Resistant",
              conformal_prediction: {
                confidence_level: 95,
                alpha: 0.05,
                label: "Resistant",
                included_classes: ["Resistant"]
              }
            },
            top_contributors: { positive: [], negative: [] },
            explanation: "Model explanation fallback."
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    window.sessionStorage.clear();
  });

  it("requires login first, then loads intake defaults and navigates to patient result view", async () => {
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Dashboard login" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/auth/status", expect.any(Object));

    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "test-user" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Backend API error (401): Invalid credentials.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "test-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Patient feature intake")).toBeInTheDocument();
    expect(screen.getByText("PHQ-9")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Generate results" }));
    expect(await screen.findByText("Welcome, John.")).toBeInTheDocument();
    expect(screen.getByText("ESTIMATED TREATMENT RESPONSE")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/predict", expect.any(Object));
  });
});
