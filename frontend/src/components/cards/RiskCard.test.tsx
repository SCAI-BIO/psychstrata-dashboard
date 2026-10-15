import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RiskCard } from "./RiskCard";

describe("RiskCard", () => {
  it("renders the risk percentage and a high-risk badge", () => {
    render(<RiskCard riskProbability={0.68} isHighRisk />);
    expect(screen.getByText("Treatment Resistance Risk")).toBeInTheDocument();
    expect(screen.getByText("68.0%")).toBeInTheDocument();
    expect(screen.getByText("High Risk")).toBeInTheDocument();
  });

  it("renders a lower-risk badge when not high risk", () => {
    render(<RiskCard riskProbability={0.2} isHighRisk={false} />);
    expect(screen.getByText("Lower Risk")).toBeInTheDocument();
  });

  it("hides the confidence slider when no confidence control is given", () => {
    render(<RiskCard riskProbability={0.5} isHighRisk={false} />);
    expect(screen.queryByLabelText(/Model Certainty/)).not.toBeInTheDocument();
    expect(screen.getByText(/Likelihood of treatment non-response within current regimen based on clinical and biological markers\./)).toBeInTheDocument();
  });

  it("renders and wires the confidence slider when provided", () => {
    const onChange = vi.fn();
    const onCommit = vi.fn();
    render(
      <RiskCard
        riskProbability={0.5}
        isHighRisk={false}
        confidence={{ value: 95, min: 80, max: 99, step: 1, onChange, onCommit }}
      />
    );

    const slider = screen.getByLabelText(/Model Certainty/);
    expect(slider).toHaveValue("95");
    fireEvent.change(slider, { target: { value: "90" } });
    expect(onChange).toHaveBeenCalledWith(90);
    fireEvent.mouseUp(slider);
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/using the specified model certainty\./)).toBeInTheDocument();
  });
});
