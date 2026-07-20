import { describe, expect, it } from "vitest";

import {
  CLINICAL_REVIEW_MONTH,
  CURVE_MONTHS,
  buildMockDeepHitCurve,
  buildMonthlyNonResponseCurve
} from "./survival";

describe("buildMockDeepHitCurve", () => {
  it("starts at zero and terminates at the model probability (week 36)", () => {
    const curve = buildMockDeepHitCurve(0.5);
    expect(curve[0].week).toBe(0);
    expect(curve[0].cif).toBe(0);
    const terminal = curve[curve.length - 1];
    expect(terminal.week).toBe(36);
    expect(terminal.cif).toBeCloseTo(0.5, 2);
  });

  it("produces a monotonically non-decreasing CIF", () => {
    const curve = buildMockDeepHitCurve(0.6);
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].cif).toBeGreaterThanOrEqual(curve[i - 1].cif);
    }
  });

  it("keeps the confidence band ordered and clamped to [0, 1]", () => {
    const curve = buildMockDeepHitCurve(0.9);
    for (const point of curve) {
      expect(point.cifLower).toBeLessThanOrEqual(point.cif);
      expect(point.cifUpper).toBeGreaterThanOrEqual(point.cif);
      expect(point.cifLower).toBeGreaterThanOrEqual(0);
      expect(point.cifUpper).toBeLessThanOrEqual(1);
    }
  });

  it("never exceeds a CIF of 1 even for extreme risk", () => {
    const curve = buildMockDeepHitCurve(1);
    for (const point of curve) {
      expect(point.cif).toBeLessThanOrEqual(1);
    }
  });
});

describe("buildMonthlyNonResponseCurve", () => {
  it("emits one point per curve month with the current plan only", () => {
    const curve = buildMonthlyNonResponseCurve(0.5);
    expect(curve.map((p) => p.month)).toEqual([...CURVE_MONTHS]);
    expect(curve.every((p) => p.simulated === null)).toBe(true);
  });

  it("adds a simulated line when a simulated probability is supplied", () => {
    const curve = buildMonthlyNonResponseCurve(0.5, 0.3);
    expect(curve.every((p) => p.simulated !== null)).toBe(true);
    const terminal = curve[curve.length - 1];
    expect(terminal.current).toBeGreaterThan(terminal.simulated as number);
  });

  it("starts both plans at zero and rises monotonically", () => {
    const curve = buildMonthlyNonResponseCurve(0.7, 0.4);
    expect(curve[0].current).toBe(0);
    expect(curve[0].simulated).toBe(0);
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].current as number).toBeGreaterThanOrEqual(curve[i - 1].current as number);
    }
  });

  it("places the clinical review month within the plotted range", () => {
    expect(CURVE_MONTHS).toContain(CLINICAL_REVIEW_MONTH);
  });
});
