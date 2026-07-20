import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResponseBadge, RiskBadge, RiskGaugeBar } from "./indicators";

describe("RiskBadge", () => {
  it("renders the high-risk label", () => {
    render(<RiskBadge isHighRisk={true} />);
    expect(screen.getByText("High Risk")).toBeInTheDocument();
  });

  it("renders the lower-risk label", () => {
    render(<RiskBadge isHighRisk={false} />);
    expect(screen.getByText("Lower Risk")).toBeInTheDocument();
  });
});

describe("ResponseBadge", () => {
  it("renders the label and applies the provided classes", () => {
    render(<ResponseBadge label="High Response" classes="bg-emerald-100 text-emerald-800" />);
    const badge = screen.getByText("High Response");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass("bg-emerald-100");
  });
});

describe("RiskGaugeBar", () => {
  it("positions the marker proportionally to the probability", () => {
    const { container } = render(<RiskGaugeBar probability={0.5} />);
    const marker = container.querySelector('[style*="left"]');
    expect(marker).toBeTruthy();
    expect(marker?.getAttribute("style")).toContain("50%");
  });

  it("clamps probabilities above 1 to the far end", () => {
    const { container } = render(<RiskGaugeBar probability={1.5} />);
    const marker = container.querySelector('[style*="left"]');
    expect(marker?.getAttribute("style")).toContain("100%");
  });
});
