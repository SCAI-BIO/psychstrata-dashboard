import { describe, expect, it } from "vitest";

import {
  adherenceCategoryValue,
  adherenceLabel,
  parseNumberValue,
  pct,
  phqSeverity,
  responseBadgeSpec,
  supportiveSummary
} from "./format";

describe("pct", () => {
  it("formats a 0–1 probability as a one-decimal percentage", () => {
    expect(pct(0.684)).toBe("68.4%");
    expect(pct(0)).toBe("0.0%");
    expect(pct(1)).toBe("100.0%");
  });
});

describe("parseNumberValue", () => {
  it("parses numeric strings", () => {
    expect(parseNumberValue("42")).toBe(42);
    expect(parseNumberValue("3.5")).toBe(3.5);
  });

  it("falls back to 0 for non-numeric input", () => {
    expect(parseNumberValue("")).toBe(0);
    expect(parseNumberValue("abc")).toBe(0);
  });
});

describe("adherenceLabel", () => {
  it("buckets adherence percentages", () => {
    expect(adherenceLabel(59)).toBe("Low");
    expect(adherenceLabel(60)).toBe("Moderate");
    expect(adherenceLabel(84)).toBe("Moderate");
    expect(adherenceLabel(85)).toBe("High");
  });
});

describe("adherenceCategoryValue", () => {
  it("maps a percentage onto the discrete select value", () => {
    expect(adherenceCategoryValue(50)).toBe(50);
    expect(adherenceCategoryValue(70)).toBe(75);
    expect(adherenceCategoryValue(95)).toBe(90);
  });
});

describe("phqSeverity", () => {
  it("labels PHQ-9 severity by threshold", () => {
    expect(phqSeverity(21)).toBe("Severe symptoms");
    expect(phqSeverity(15)).toBe("Moderately severe");
    expect(phqSeverity(10)).toBe("Medium impairment");
    expect(phqSeverity(4)).toBe("Mild-to-moderate");
  });
});

describe("supportiveSummary", () => {
  it("returns tone-appropriate copy by risk band", () => {
    expect(supportiveSummary(0.7)).toMatch(/closer follow-up/i);
    expect(supportiveSummary(0.5)).toMatch(/mixed outlook/i);
    expect(supportiveSummary(0.2)).toMatch(/favorable/i);
  });
});

describe("responseBadgeSpec", () => {
  it("returns a label + classes for each response band", () => {
    expect(responseBadgeSpec(0.8).label).toBe("High Response");
    expect(responseBadgeSpec(0.5).label).toBe("Mixed Response");
    expect(responseBadgeSpec(0.2).label).toBe("Low Response");
  });

  it("always includes tailwind classes", () => {
    expect(responseBadgeSpec(0.8).classes).toContain("emerald");
    expect(responseBadgeSpec(0.2).classes).toContain("red");
  });
});
