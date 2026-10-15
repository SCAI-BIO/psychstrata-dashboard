import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { SimValues } from "../types";
import { DoseSlider, PredictedImpact, SimControls } from "./SimPanel";

const baseValues: SimValues = {
  sertraline_mg: 100,
  lithium_mg: 0,
  quetiapine_mg: 0,
  adherence_pct: 75
};

describe("DoseSlider", () => {
  it("renders the label and current value and reports changes", () => {
    const onChange = vi.fn();
    render(<DoseSlider label="Sertraline Dosage (mg/day)" value={100} min={0} max={400} step={25} onChange={onChange} />);

    expect(screen.getByText("Sertraline Dosage (mg/day)")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("slider"), { target: { value: "200" } });
    expect(onChange).toHaveBeenCalledWith(200);
  });
});

describe("SimControls", () => {
  it("renders all three dose sliders and the adherence dropdown by default", () => {
    render(<SimControls values={baseValues} onChange={vi.fn()} />);
    expect(screen.getByText("Sertraline Dosage (mg/day)")).toBeInTheDocument();
    expect(screen.getByText("Lithium Dosage (mg/day)")).toBeInTheDocument();
    expect(screen.getByText("Quetiapine Augmentation (mg/day)")).toBeInTheDocument();
    expect(screen.getByLabelText("Therapeutic Adherence")).toBeInTheDocument();
  });

  it("hides the adherence dropdown when showAdherence is false", () => {
    render(<SimControls values={baseValues} onChange={vi.fn()} showAdherence={false} />);
    expect(screen.queryByLabelText("Therapeutic Adherence")).not.toBeInTheDocument();
  });

  it("reports slider patches with the right key", () => {
    const onChange = vi.fn();
    render(<SimControls values={baseValues} onChange={onChange} />);
    const sliders = screen.getAllByRole("slider");
    fireEvent.change(sliders[0], { target: { value: "150" } });
    expect(onChange).toHaveBeenCalledWith({ sertraline_mg: 150 });
    fireEvent.change(sliders[2], { target: { value: "75" } });
    expect(onChange).toHaveBeenCalledWith({ quetiapine_mg: 75 });
  });

  it("reports the adherence category change", () => {
    const onChange = vi.fn();
    render(<SimControls values={baseValues} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Therapeutic Adherence"), { target: { value: "90" } });
    expect(onChange).toHaveBeenCalledWith({ adherence_pct: 90 });
  });
});

describe("PredictedImpact", () => {
  it("shows a reduction for positive impact", () => {
    render(<PredictedImpact impactPct={12.3} />);
    expect(screen.getByText("-12.3% Resistance Risk Reduction")).toBeInTheDocument();
  });

  it("shows an increase for negative impact", () => {
    render(<PredictedImpact impactPct={-5} />);
    expect(screen.getByText("+5.0% Resistance Risk Increase")).toBeInTheDocument();
  });

  it("shows a neutral change for zero impact", () => {
    render(<PredictedImpact impactPct={0} />);
    expect(screen.getByText("0.0% Resistance Risk Change")).toBeInTheDocument();
  });

  it("renders an error message when provided", () => {
    render(<PredictedImpact impactPct={1} error="model unavailable" />);
    expect(screen.getByText("model unavailable")).toBeInTheDocument();
  });
});
