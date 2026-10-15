import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatCards } from "./StatCards";

describe("StatCards", () => {
  it("renders each item's label, value, suffix and caption", () => {
    render(
      <StatCards
        items={[
          { label: "PHQ-9 Score", value: 14, caption: "Moderate" },
          { label: "Lithium Level", value: 600, suffix: "mg/day", caption: "Within augmentation range" }
        ]}
      />
    );

    expect(screen.getByText("PHQ-9 Score")).toBeInTheDocument();
    expect(screen.getByText("14")).toBeInTheDocument();
    expect(screen.getByText("Moderate")).toBeInTheDocument();
    expect(screen.getByText("Lithium Level")).toBeInTheDocument();
    expect(screen.getByText("mg/day")).toBeInTheDocument();
    expect(screen.getByText("Within augmentation range")).toBeInTheDocument();
  });

  it("omits the suffix when not supplied", () => {
    render(<StatCards items={[{ label: "Adherence", value: "High", caption: "Potential for improvement" }]} />);
    expect(screen.queryByText("mg/day")).not.toBeInTheDocument();
  });
});
