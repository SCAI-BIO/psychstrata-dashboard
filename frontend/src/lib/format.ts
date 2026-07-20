/** Formatting + small clinical-derivation helpers shared across views. */

export function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function parseNumberValue(raw: string): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function supportiveSummary(probability: number): string {
  if (probability >= 0.65) {
    return "Your profile suggests this treatment plan may need closer follow-up and adjustment.";
  }
  if (probability >= 0.4) {
    return "Your profile suggests a mixed outlook, and regular check-ins can help improve response.";
  }
  return "Your profile suggests a favorable chance of response with the current treatment plan.";
}

/** Coarse adherence bucket from a 0–100 percentage. */
export function adherenceLabel(adherencePct: number): "Low" | "Moderate" | "High" {
  if (adherencePct < 60) return "Low";
  if (adherencePct < 85) return "Moderate";
  return "High";
}

/** Map an adherence percentage onto the discrete select value used by the sim panel. */
export function adherenceCategoryValue(adherencePct: number): number {
  if (adherencePct < 60) return 50;
  if (adherencePct < 85) return 75;
  return 90;
}

export function phqSeverity(phqScore: number): string {
  if (phqScore >= 20) return "Severe symptoms";
  if (phqScore >= 15) return "Moderately severe";
  if (phqScore >= 10) return "Medium impairment";
  return "Mild-to-moderate";
}

/** Label + Tailwind classes for a response badge. */
export type ResponseBadgeSpec = { label: string; classes: string };

/** Badge styling for the patient-facing "Estimated Treatment Response" figure. */
export function responseBadgeSpec(responseProbability: number): ResponseBadgeSpec {
  if (responseProbability >= 0.7) {
    return { label: "High Response", classes: "bg-emerald-100 text-emerald-800" };
  }
  if (responseProbability >= 0.45) {
    return { label: "Mixed Response", classes: "bg-amber-100 text-amber-800" };
  }
  return { label: "Low Response", classes: "bg-red-300/70 text-red-900" };
}
