import type { FeatureSchema, PredictionResponse, TsneResponse } from "./api";

/** The four top-level screens of the app. */
export type Route = "intake" | "patient" | "clinician" | "scientist";

/** Result routes only (everything except the intake wizard). */
export type ResultRoute = Exclude<Route, "intake">;

/** The narrowed, data-loaded variant of {@link LoadState}. */
export type ReadyState = Extract<LoadState, { status: "ready" }>;

/**
 * Async lifecycle of the dashboard's backend-driven data.
 * `ready` carries the full working set shared across every result view.
 */
export type LoadState =
  | { status: "loading" }
  | {
      status: "ready";
      features: FeatureSchema[];
      confidenceBounds: { min: number; max: number; step: number };
      confidenceLevel: number;
      prediction: PredictionResponse | null;
      explanation: string;
      tsne: TsneResponse | null;
      isSubmitting: boolean;
      isSummaryRefreshing: boolean;
      error: string | null;
    }
  | { status: "error"; message: string };

/** The subset of model features exposed as quick-adjust sliders in the sim panels. */
export interface SimValues {
  sertraline_mg: number;
  lithium_mg: number;
  quetiapine_mg: number;
  adherence_pct: number;
}

/**
 * DeepHit-shaped survival point. Field names are intentionally stable so the
 * future DeepHit model can drop straight into the existing chart. Do not rename.
 */
export interface SurvivalPoint {
  week: number; // discrete time horizon (DeepHit time bin)
  cif: number; // patient F(t): cumulative probability of TR by `week`
  cifLower: number; // 95% CI lower bound
  cifUpper: number; // 95% CI upper bound
  population: number; // STAR*D / UKBB Kaplan-Meier reference CIF
}
