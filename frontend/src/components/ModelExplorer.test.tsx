import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PredictionResponse } from "../api";
import { fetchPredict } from "../api";
import { featuresPayload, predictionResponse } from "../test/fixtures";
import { ModelExplorer } from "./ModelExplorer";

vi.mock("../api", () => ({ fetchPredict: vi.fn() }));

const baselineFeatures = featuresPayload.defaults;

type ExploreOverrides = Partial<Omit<PredictionResponse, "prediction">> & {
  prediction?: Partial<PredictionResponse["prediction"]>;
};

function explorePrediction(overrides: ExploreOverrides = {}) {
  return {
    ...predictionResponse,
    ...overrides,
    prediction: {
      ...predictionResponse.prediction,
      ...(overrides.prediction ?? {})
    }
  };
}

function renderExplorer() {
  return render(
    <ModelExplorer
      features={featuresPayload.features}
      baselineFeatures={baselineFeatures}
      baselinePrediction={predictionResponse}
      confidenceLevel={95}
      dateOfBirth="1978-05-12"
    />
  );
}

/** Fire the debounce timer and flush the resulting fetch promise microtask. */
async function advanceDebounce(ms = 500) {
  await act(async () => {
    vi.advanceTimersByTime(ms);
  });
}

describe("ModelExplorer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the baseline prediction and feature groups on mount", () => {
    renderExplorer();

    expect(screen.getByText("Model Explorer")).toBeInTheDocument();
    expect(screen.getByText("68.0%")).toBeInTheDocument();
    expect(screen.getByText("High Risk")).toBeInTheDocument();
    expect(screen.getByText("Same risk as patient baseline")).toBeInTheDocument();
    expect(screen.getByText("Predicted Outcome")).toBeInTheDocument();
    expect(screen.getByText("Feature Importance (SHAP)")).toBeInTheDocument();

    expect(screen.getByLabelText("PHQ-9")).toBeInTheDocument();
    expect(screen.getByLabelText("Sertraline")).toBeInTheDocument();
    expect(screen.getByLabelText("Adherence (%)")).toBeInTheDocument();

    expect(fetchPredict).not.toHaveBeenCalled();
  });

  it("recomputes the prediction after a debounced edit", async () => {
    vi.mocked(fetchPredict).mockResolvedValue(
      explorePrediction({ prediction: { probability_resistance: 0.4, predicted_class: "Responsive" } })
    );
    renderExplorer();

    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "5" } });

    expect(screen.getByText("Updating prediction…")).toBeInTheDocument();
    expect(fetchPredict).not.toHaveBeenCalled();

    await advanceDebounce();

    expect(fetchPredict).toHaveBeenCalledTimes(1);
    expect(fetchPredict).toHaveBeenCalledWith(
      expect.objectContaining({
        features: expect.objectContaining({ phq9: 5 }),
        confidence_level: 95,
        date_of_birth: "1978-05-12"
      })
    );
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    expect(screen.getByText("Lower Risk")).toBeInTheDocument();
    expect(screen.getByText("▼ 28.0 pts lower risk vs patient baseline")).toBeInTheDocument();
  });

  it("collapses rapid edits into a single debounced request", async () => {
    vi.mocked(fetchPredict).mockResolvedValue(
      explorePrediction({ prediction: { probability_resistance: 0.5 } })
    );
    renderExplorer();

    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "10" } });
    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "8" } });
    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "6" } });

    await advanceDebounce(250);
    expect(fetchPredict).not.toHaveBeenCalled();

    await advanceDebounce(250);
    expect(fetchPredict).toHaveBeenCalledTimes(1);
  });

  it("resets back to the patient baseline", async () => {
    vi.mocked(fetchPredict).mockResolvedValue(
      explorePrediction({ prediction: { probability_resistance: 0.4 } })
    );
    renderExplorer();

    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "5" } });
    await advanceDebounce();
    expect(screen.getByText("40.0%")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset to patient" }));

    expect(screen.getByText("68.0%")).toBeInTheDocument();
    expect(screen.getByText("Same risk as patient baseline")).toBeInTheDocument();
    expect(fetchPredict).toHaveBeenCalledTimes(1);
  });

  it("surfaces prediction errors", async () => {
    vi.mocked(fetchPredict).mockRejectedValue(new Error("model unavailable"));
    renderExplorer();

    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "5" } });
    await advanceDebounce();

    expect(screen.getByText("model unavailable")).toBeInTheDocument();
  });

  it("renders the model certainty slider and commits on release", () => {
    const onChange = vi.fn();
    const onCommit = vi.fn();
    render(
      <ModelExplorer
        features={featuresPayload.features}
        baselineFeatures={baselineFeatures}
        baselinePrediction={predictionResponse}
        confidenceLevel={95}
        confidence={{ value: 95, min: 50, max: 99, step: 1, onChange, onCommit }}
      />
    );

    const slider = screen.getByLabelText("Model Certainty — Confidence Interval (%)");
    expect(slider).toHaveValue("95");
    fireEvent.change(slider, { target: { value: "90" } });
    expect(onChange).toHaveBeenCalledWith(90);
    fireEvent.mouseUp(slider);
    expect(onCommit).toHaveBeenCalled();
  });
});
