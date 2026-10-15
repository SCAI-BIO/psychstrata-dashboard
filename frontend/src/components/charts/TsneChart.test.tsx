import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { tsneResponse } from "../../test/fixtures";
import { TsneChart } from "./TsneChart";

vi.mock("plotly.js-dist-min", () => ({ default: { newPlot: vi.fn(), react: vi.fn() } }));
vi.mock("react-plotly.js/factory", () => ({
  default: () => () => <div data-testid="plotly-plot" />
}));

describe("TsneChart", () => {
  it("shows the Suspense fallback before the lazy Plot resolves", () => {
    render(<TsneChart tsne={tsneResponse} selected={{ x: 0.2, y: 0.3 }} />);
    expect(screen.getByText("Loading chart…")).toBeInTheDocument();
  });

  it("renders the plot and the legend once loaded", async () => {
    render(<TsneChart tsne={tsneResponse} selected={{ x: 0.2, y: 0.3 }} />);
    expect(await screen.findByTestId("plotly-plot")).toBeInTheDocument();
    expect(screen.getByText("Current Patient")).toBeInTheDocument();
    expect(screen.getByText("Responsive Patient")).toBeInTheDocument();
    expect(screen.getByText("Resistant Patient")).toBeInTheDocument();
  });
});
