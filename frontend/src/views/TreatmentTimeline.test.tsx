import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TreatmentPlanRecord } from "../api";
import {
  computeTimelineLayout,
  GAP_PX,
  MIN_SEGMENT_PX,
  sortChronological,
  START_X,
  TRACK_TARGET_PX,
  TreatmentTimeline
} from "./TreatmentTimeline";

const NOW = new Date(2026, 9, 1); // 2026-10-01, local

function plan(
  id: string,
  start_date: string | null,
  created_at: string,
  end_date: string | null = null
): TreatmentPlanRecord {
  return {
    id,
    patient_id: "patient-1",
    clinician_id: "clinician-1",
    start_date,
    end_date,
    medications: { sertraline_mg: 100, lithium_mg: 0, quetiapine_mg: 0 },
    adherence: { adherence_pct: 80 },
    created_at,
    updated_at: created_at
  };
}

describe("sortChronological", () => {
  it("orders plans by start date ascending", () => {
    const b = plan("b", "2026-02-01", "2026-01-01T00:00:00Z");
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    const c = plan("c", "2026-03-01", "2026-01-01T00:00:00Z");
    expect(sortChronological([b, c, a]).map((p) => p.id)).toEqual(["a", "b", "c"]);
  });

  it("uses created_at's date when start_date is missing", () => {
    const b = plan("b", null, "2026-02-10T00:00:00Z");
    const a = plan("a", null, "2026-01-10T00:00:00Z");
    expect(sortChronological([b, a]).map((p) => p.id)).toEqual(["a", "b"]);
  });

  it("breaks ties by created_at, then id", () => {
    const a = plan("a", "2026-01-01", "2026-01-02T00:00:00Z");
    const b = plan("b", "2026-01-01", "2026-01-01T00:00:00Z");
    const c = plan("c", "2026-01-01", "2026-01-01T00:00:00Z");
    expect(sortChronological([a, b, c]).map((p) => p.id)).toEqual(["b", "c", "a"]);
  });
});

describe("computeTimelineLayout", () => {
  it("numbers plans 1..n in chronological order", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    const b = plan("b", "2026-03-01", "2026-01-01T00:00:00Z");
    const c = plan("c", "2026-02-01", "2026-01-01T00:00:00Z");
    const segments = computeTimelineLayout([b, a, c], NOW);
    expect(segments.map((s) => [s.number, s.plan.id])).toEqual([
      [1, "a"],
      [2, "c"],
      [3, "b"]
    ]);
  });

  it("gives very short plans the minimum width", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z", "2026-01-01");
    const b = plan("b", "2026-01-02", "2026-01-01T00:00:00Z", "2026-01-02");
    const c = plan("c", "2026-01-03", "2026-01-01T00:00:00Z");
    const segments = computeTimelineLayout([a, b, c], NOW);
    expect(segments[0].endX - segments[0].startX).toBe(MIN_SEGMENT_PX);
    expect(segments[1].endX - segments[1].startX).toBe(MIN_SEGMENT_PX);
  });

  it("never overlaps consecutive bars", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z", "2026-01-15");
    const b = plan("b", "2026-01-20", "2026-01-01T00:00:00Z");
    const segments = computeTimelineLayout([a, b], NOW);
    expect(segments[1].startX).toBeGreaterThanOrEqual(segments[0].endX + GAP_PX);
  });

  it("lets a long treatment gap show as empty track", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z", "2026-02-01");
    const b = plan("b", "2027-01-01", "2026-01-01T00:00:00Z");
    const segments = computeTimelineLayout([a, b], NOW);
    expect(segments[1].startX - segments[0].endX).toBeGreaterThan(60);
  });

  it("extends the still-running plan to now", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    const segments = computeTimelineLayout([a], NOW);
    expect(segments).toHaveLength(1);
    expect(segments[0].startX).toBe(START_X);
    expect(segments[0].endX).toBeCloseTo(START_X + (TRACK_TARGET_PX - 2 * START_X), 0);
  });

  it("positions by created_at when start_date is missing", () => {
    const a = plan("a", null, "2026-01-10T00:00:00Z");
    const b = plan("b", null, "2026-03-10T00:00:00Z");
    const segments = computeTimelineLayout([b, a], NOW);
    expect(segments[0].plan.id).toBe("a");
    expect(segments[0].startX).toBe(START_X);
    expect(segments[1].startX).toBeGreaterThan(segments[0].endX);
  });
});

describe("TreatmentTimeline", () => {
  const onOpen = vi.fn();

  beforeEach(() => {
    onOpen.mockReset();
  });

  it("renders nothing when there are no plans", () => {
    const { container } = render(<TreatmentTimeline plans={[]} latestPlanId={null} onOpen={onOpen} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders one bar per plan, numbered in chronological order", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    const b = plan("b", "2026-02-01", "2026-01-01T00:00:00Z");
    render(<TreatmentTimeline plans={[b, a]} latestPlanId="b" onOpen={onOpen} />);
    const bars = screen.getAllByTestId("treatment-timeline-segment");
    expect(bars).toHaveLength(2);
    expect(bars[0]).toHaveAttribute("data-plan-id", "a");
    expect(bars[0]).toHaveAttribute("data-number", "1");
    expect(bars[1]).toHaveAttribute("data-plan-id", "b");
    expect(bars[1]).toHaveAttribute("data-number", "2");
  });

  it("opens the clicked plan's results", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    render(<TreatmentTimeline plans={[a]} latestPlanId="a" onOpen={onOpen} />);
    fireEvent.click(screen.getAllByTestId("treatment-timeline-segment")[0]);
    expect(onOpen).toHaveBeenCalledWith(a);
  });

  it("marks the end of the history with a Today label", () => {
    const a = plan("a", "2026-01-01", "2026-01-01T00:00:00Z");
    render(<TreatmentTimeline plans={[a]} latestPlanId="a" onOpen={onOpen} />);
    expect(screen.getByText("Today")).toBeInTheDocument();
  });
});
