import type { SurvivalPoint } from "../types";

// ── Clinical & Model Rationale ─────────────────────────────────────────────
//
// 1. CLINICAL: MDD non-response accumulation is CONCAVE (decelerating).
//    Most failures occur in Trials 1–2, with diminishing new failures each
//    subsequent trial. Source: STAR*D (Rush et al., 2006, Am J Psychiatry).
//
// 2. FUTURE MODEL: A DeepHit survival model trained on UKBB data will replace
//    the current RF classifier. DeepHit outputs a per-patient CIF:
//      F(t) = P(treatment resistance by week t)
//    at discrete time horizons, plus bootstrap 95% CI bands.
//
// 3. MIGRATION PLAN: The chart already accepts data in DeepHit output format
//    (SurvivalPoint[]). When DeepHit is integrated, only buildMockDeepHitCurve()
//    needs to be replaced — the chart is untouched.
//
// 4. CURRENT MOCK: The single RF classifier probability p is scaled onto the
//    STAR*D population CIF shape to produce a clinically plausible curve.
// ──────────────────────────────────────────────────────────────────────────

/** STAR*D population CIF reference — cumulative non-response at each trial endpoint. */
const STAERD_POPULATION_CIF: { week: number; cif: number }[] = [
  { week: 0, cif: 0.0 },
  { week: 8, cif: 0.38 }, // end of Trial 1
  { week: 16, cif: 0.51 }, // end of Trial 2
  { week: 24, cif: 0.61 }, // end of Trial 3
  { week: 36, cif: 0.67 } // TRD threshold (≥4 failed trials)
];

/** CI half-widths widen over time — mirrors the survival model uncertainty fan. */
const MOCK_CI_HALF_WIDTH = [0.0, 0.04, 0.07, 0.09, 0.11];

const STAERD_TERMINAL_CIF = 0.67; // population median at Week 36

/**
 * Builds a mock CIF curve in DeepHit output format. Scales the STAR*D
 * population shape so the patient curve terminates at the RF classifier
 * probability at Week 36.
 *
 * MIGRATION: replace this function body with DeepHit output mapping. Keep the
 * SurvivalPoint[] return type and field names unchanged.
 */
export function buildMockDeepHitCurve(riskProbability: number): SurvivalPoint[] {
  const scale = riskProbability / STAERD_TERMINAL_CIF;
  return STAERD_POPULATION_CIF.map((pt, i) => {
    const cif = parseFloat(Math.min(1.0, pt.cif * scale).toFixed(3));
    const half = MOCK_CI_HALF_WIDTH[i];
    return {
      week: pt.week,
      cif,
      cifLower: parseFloat(Math.max(0, cif - half).toFixed(3)),
      cifUpper: parseFloat(Math.min(1, cif + half).toFixed(3)),
      population: pt.cif
    };
  });
}

/** Human-readable trial milestone labels keyed by week. */
export const TRIAL_LABELS: Record<number, string> = {
  8: "Trial 1",
  16: "Trial 2",
  24: "Trial 3",
  36: "TRD"
};

// ── Month-based non-response curve (mockup chart) ──────────────────────────
// The dashboard's time-to-event chart plots cumulative probability of
// non-response over a 0–15 month horizon. Each plan's curve is a concave
// (decelerating) logistic scaled so its 15-month endpoint equals the model's
// resistance probability for that plan.

export const CURVE_MONTHS = [0, 3, 6, 9, 12, 15] as const;

/** Month at which the "Clinical Review Point" marker is placed. */
export const CLINICAL_REVIEW_MONTH = 9;

export interface MonthlyPoint {
  month: number;
  /** Current care plan CIF at this month (0–1), or null if not shown. */
  current: number | null;
  /** Simulated care plan CIF at this month (0–1), or null if not shown. */
  simulated: number | null;
}

const CURVE_TIME_CONSTANT = 6; // months; controls concavity
const CURVE_NORMALISER = 1 - Math.exp(-15 / CURVE_TIME_CONSTANT);

function cifAtMonth(month: number, terminalProbability: number): number {
  const shape = (1 - Math.exp(-month / CURVE_TIME_CONSTANT)) / CURVE_NORMALISER;
  return parseFloat((terminalProbability * shape).toFixed(4));
}

/**
 * Builds the monthly non-response curve(s). Pass `simulatedProbability` to
 * overlay a second "simulated care plan" line; omit it for a single-line chart.
 */
export function buildMonthlyNonResponseCurve(
  currentProbability: number,
  simulatedProbability?: number | null
): MonthlyPoint[] {
  return CURVE_MONTHS.map((month) => ({
    month,
    current: cifAtMonth(month, currentProbability),
    simulated:
      simulatedProbability === undefined || simulatedProbability === null
        ? null
        : cifAtMonth(month, simulatedProbability)
  }));
}
