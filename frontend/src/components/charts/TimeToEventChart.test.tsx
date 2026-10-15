import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TimeToEventChart } from "./TimeToEventChart";

describe("TimeToEventChart", () => {
  it("renders the current-plan legend without a simulated line", () => {
    render(<TimeToEventChart currentProbability={0.68} />);
    expect(screen.getByText("Current Care Plan")).toBeInTheDocument();
    expect(screen.getByText("Clinical Review Point")).toBeInTheDocument();
    expect(screen.queryByText("Simulated Care Plan")).not.toBeInTheDocument();
  });

  it("renders the simulated legend when a simulated probability is supplied", () => {
    render(<TimeToEventChart currentProbability={0.68} simulatedProbability={0.4} />);
    expect(screen.getByText("Simulated Care Plan")).toBeInTheDocument();
    expect(screen.getByText("Current Care Plan")).toBeInTheDocument();
  });

  it("honours custom labels", () => {
    render(
      <TimeToEventChart currentProbability={0.5} simulatedProbability={0.3} currentLabel="Baseline" simulatedLabel="New plan" />
    );
    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("New plan")).toBeInTheDocument();
  });

  it("treats an explicit null simulated probability as absent", () => {
    render(<TimeToEventChart currentProbability={0.5} simulatedProbability={null} />);
    expect(screen.queryByText("Simulated Care Plan")).not.toBeInTheDocument();
  });
});
