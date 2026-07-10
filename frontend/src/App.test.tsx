import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const summary = {
  title: "Treatment Resistance Classifier Demo",
  message: "The React frontend is connected to the FastAPI backend.",
  disclaimer:
    "This demo uses synthetic data for illustration purposes only. It is not a medical device and must not be used for clinical decisions."
};

describe("App", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(summary), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders content loaded from the backend API", async () => {
    render(<App />);

    expect(screen.getByText("Loading backend response...")).toBeInTheDocument();
    expect(await screen.findByText(summary.message)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/summary");
  });
});
