import { describe, expect, it } from "vitest";

import { predictionResponse } from "../test/fixtures";
import { deriveClinical } from "./derive";

describe("deriveClinical", () => {
  const featureValues = { phq9: 18, adherence_pct: 55, sertraline_mg: 100, lithium_mg: 300, quetiapine_mg: 0 };

  it("derives risk/response from the prediction", () => {
    const d = deriveClinical(featureValues, predictionResponse);
    expect(d.riskProbability).toBe(0.68);
    expect(d.responseProbability).toBeCloseTo(0.32, 5);
    expect(d.isHighRisk).toBe(true);
  });

  it("counts active agents and total medication load", () => {
    const d = deriveClinical(featureValues, predictionResponse);
    expect(d.activeAgents).toBe(2); // sertraline + lithium
    expect(d.medicationLoad).toBe(400);
  });

  it("buckets adherence and reads PHQ score", () => {
    const d = deriveClinical(featureValues, predictionResponse);
    expect(d.adherenceBucket).toBe("Low");
    expect(d.phqScore).toBe(18);
  });

  it("maps top contributors to strengths and action items", () => {
    const d = deriveClinical(featureValues, predictionResponse);
    expect(d.strengths).toBe(predictionResponse.top_contributors.negative);
    expect(d.actionItems).toBe(predictionResponse.top_contributors.positive);
  });

  it("defaults missing feature values to 0", () => {
    const d = deriveClinical({}, predictionResponse);
    expect(d.sertralineDose).toBe(0);
    expect(d.activeAgents).toBe(0);
    expect(d.medicationLoad).toBe(0);
  });
});
