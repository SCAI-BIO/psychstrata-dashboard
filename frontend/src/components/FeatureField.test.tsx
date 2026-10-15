import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FeatureSchema } from "../api";
import { FeatureField } from "./FeatureField";

const numeric: FeatureSchema = {
  id: "phq9",
  label: "PHQ-9",
  dtype: "numeric",
  category: "clinical",
  default: 12,
  min: 0,
  max: 27,
  step: 1
};

const categorical: FeatureSchema = {
  id: "sleep_severity",
  label: "Sleep Disturbance",
  dtype: "categorical",
  category: "clinical",
  default: 0,
  options: [
    { value: 0, label: "None" },
    { value: 1, label: "Mild" },
    { value: 2, label: "Severe" }
  ]
};

describe("FeatureField", () => {
  it("renders a numeric feature as a range with a live output", () => {
    render(<FeatureField feature={numeric} value={12} onChange={vi.fn()} />);
    const slider = screen.getByLabelText("PHQ-9");
    expect(slider).toHaveAttribute("type", "range");
    expect(slider).toHaveValue("12");
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("reports numeric changes with the feature id", () => {
    const onChange = vi.fn();
    render(<FeatureField feature={numeric} value={12} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("PHQ-9"), { target: { value: "5" } });
    expect(onChange).toHaveBeenCalledWith("phq9", 5);
  });

  it("renders a categorical feature as a select with its options", () => {
    render(<FeatureField feature={categorical} value={1} onChange={vi.fn()} />);
    const select = screen.getByLabelText("Sleep Disturbance");
    expect(select.tagName).toBe("SELECT");
    expect(select).toHaveValue("1");
    expect(screen.getByRole("option", { name: "Mild" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Severe" })).toBeInTheDocument();
  });

  it("reports categorical changes with the feature id", () => {
    const onChange = vi.fn();
    render(<FeatureField feature={categorical} value={0} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Sleep Disturbance"), { target: { value: "2" } });
    expect(onChange).toHaveBeenCalledWith("sleep_severity", 2);
  });
});
