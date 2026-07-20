import type { PredictionResponse } from "../api";
import { adherenceLabel } from "./format";

/**
 * Clinical values derived from the current prediction + feature vector, shared
 * by every result view so the maths lives in exactly one place. The feature
 * vector is the patient model's projection (see domain/patient.ts).
 */
export function deriveClinical(featureValues: Record<string, number>, prediction: PredictionResponse) {
  const riskProbability = prediction.prediction.probability_resistance;
  const isHighRisk = prediction.prediction.predicted_class === "Resistant";

  const sertralineDose = featureValues.sertraline_mg ?? 0;
  const lithiumDose = featureValues.lithium_mg ?? 0;
  const quetiapineDose = featureValues.quetiapine_mg ?? 0;
  const adherencePct = featureValues.adherence_pct ?? 0;
  const phqScore = featureValues.phq9 ?? 0;

  const activeAgents = [sertralineDose, lithiumDose, quetiapineDose].filter((d) => d > 0).length;
  const medicationLoad = sertralineDose + lithiumDose + quetiapineDose;

  return {
    featureValues,
    riskProbability,
    isHighRisk,
    responseProbability: 1 - riskProbability,
    sertralineDose,
    lithiumDose,
    quetiapineDose,
    adherencePct,
    adherenceBucket: adherenceLabel(adherencePct),
    phqScore,
    activeAgents,
    medicationLoad,
    strengths: prediction.top_contributors.negative ?? [],
    actionItems: prediction.top_contributors.positive ?? [],
    selectedTsne: prediction.tsne.selected
  };
}

export type DerivedClinical = ReturnType<typeof deriveClinical>;
