import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makeDashboard, makeReadyState, predictionResponse } from "../test/fixtures";
import { ScientistView } from "./ScientistView";

vi.mock("../components/ModelExplorer", () => ({ ModelExplorer: () => <div data-testid="model-explorer" /> }));
vi.mock("../components/charts/TimeToEventChart", () => ({ TimeToEventChart: () => <div data-testid="tte-chart" /> }));
vi.mock("../components/charts/TsneChart", () => ({ TsneChart: () => <div data-testid="tsne-chart" /> }));
vi.mock("../components/cards/InsightsCard", () => ({ InsightsCard: () => <div data-testid="insights-card" /> }));

describe("ScientistView", () => {
  it("renders the explorer, charts and insights card", () => {
    render(<ScientistView dashboard={makeDashboard()} ready={makeReadyState()} prediction={predictionResponse} />);
    expect(screen.getByTestId("model-explorer")).toBeInTheDocument();
    expect(screen.getByTestId("tte-chart")).toBeInTheDocument();
    expect(screen.getByTestId("tsne-chart")).toBeInTheDocument();
    expect(screen.getByTestId("insights-card")).toBeInTheDocument();
    expect(screen.getByText("t-SNE Population Map")).toBeInTheDocument();
  });

  it("shows a fallback when t-SNE data is unavailable", () => {
    render(
      <ScientistView
        dashboard={makeDashboard()}
        ready={makeReadyState({ tsne: null })}
        prediction={predictionResponse}
      />
    );
    expect(screen.getByText("t-SNE data unavailable.")).toBeInTheDocument();
    expect(screen.queryByTestId("tsne-chart")).not.toBeInTheDocument();
  });
});
