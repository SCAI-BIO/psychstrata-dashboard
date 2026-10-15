import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { InsightsCard } from "./InsightsCard";

describe("InsightsCard", () => {
  it("renders the explanation markdown and the pill label", () => {
    render(<InsightsCard explanation={"**Key drivers**\n\n- PHQ-9 high"} isRefreshing={false} onRefresh={vi.fn()} />);
    expect(screen.getByText("AI Clinical Insights & Literature Review")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "LLM-Generated Explanation" })).toBeInTheDocument();
    expect(screen.getByText("Key drivers")).toBeInTheDocument();
    expect(screen.getByText("PHQ-9 high")).toBeInTheDocument();
  });

  it("shows the refreshing placeholder and disables the pill while refreshing", () => {
    render(<InsightsCard explanation="Some text" isRefreshing onRefresh={vi.fn()} />);
    expect(screen.getByText("Refreshing explanation for the current profile…")).toBeInTheDocument();
    const pill = screen.getByRole("button", { name: "Updating…" });
    expect(pill).toBeDisabled();
  });

  it("shows the generate prompt when there is no explanation", () => {
    const onRefresh = vi.fn();
    render(<InsightsCard explanation="" isRefreshing={false} onRefresh={onRefresh} />);
    const generate = screen.getByRole("button", { name: "Generate summary" });
    fireEvent.click(generate);
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("fires onRefresh from the pill button", () => {
    const onRefresh = vi.fn();
    render(<InsightsCard explanation="text" isRefreshing={false} onRefresh={onRefresh} />);
    fireEvent.click(screen.getByRole("button", { name: "LLM-Generated Explanation" }));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("surfaces an error message", () => {
    render(<InsightsCard explanation="text" isRefreshing={false} onRefresh={vi.fn()} error="boom" />);
    expect(screen.getByText("boom")).toBeInTheDocument();
  });
});
