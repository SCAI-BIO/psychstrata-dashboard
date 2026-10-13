import type { TreatmentPlanRecord } from "../api";

// ── Timeline geometry ───────────────────────────────────────────────────────

export const START_X = 16; // px from the left so the first segment isn't clipped
export const MIN_SEGMENT_PX = 48; // shortest a plan's bar can be (a "quick" switch)
export const GAP_PX = 2; // breathing room between back-to-back plans
export const TRACK_TARGET_PX = 560; // the full time span is scaled to fit this width
const DAY_MS = 86_400_000;

export type TimelineSegment = {
  plan: TreatmentPlanRecord;
  number: number; // 1 = the patient's initial treatment
  startX: number; // px left edge of the plan's bar
  endX: number; // px right edge of the plan's bar
};

/** Ascending by start date (falling back to created_at's date), then created_at, then id. */
export function sortChronological(plans: TreatmentPlanRecord[]): TreatmentPlanRecord[] {
  return [...plans].sort((a, b) => {
    const aDate = a.start_date ?? a.created_at.slice(0, 10);
    const bDate = b.start_date ?? b.created_at.slice(0, 10);
    if (aDate !== bDate) return aDate.localeCompare(bDate);
    if (a.created_at !== b.created_at) return a.created_at.localeCompare(b.created_at);
    return a.id.localeCompare(b.id);
  });
}

function parseIsoDate(iso: string): Date | null {
  const [year, month, day] = iso.split("-").map(Number);
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
}

function planDate(plan: TreatmentPlanRecord): Date | null {
  return parseIsoDate(plan.start_date ?? plan.created_at.slice(0, 10));
}

/**
 * The date a plan's bar runs to: its end_date, the next plan's start when the
 * end is missing, or `now` for the final (still running) plan.
 */
function planEndDate(plan: TreatmentPlanRecord, index: number, sorted: TreatmentPlanRecord[], now: Date): Date {
  if (plan.end_date) {
    const end = parseIsoDate(plan.end_date);
    if (end) return end;
  }
  if (index < sorted.length - 1) {
    const next = planDate(sorted[index + 1]);
    if (next) return next;
  }
  return now;
}

/**
 * Shared time axis: the first plan's start → `now`, scaled so the whole span
 * fits TRACK_TARGET_PX. Used by both the bar layout and the year gridlines.
 */
function timelineScale(plans: TreatmentPlanRecord[], now: Date) {
  const sorted = sortChronological(plans);
  if (sorted.length === 0) return { sorted, anchor: now, scale: 1 };
  const anchor = planDate(sorted[0]) ?? now;
  const last = sorted[sorted.length - 1];
  const anchorEnd = planEndDate(last, sorted.length - 1, sorted, now);
  const totalDays = Math.max(1, (anchorEnd.getTime() - anchor.getTime()) / DAY_MS);
  return { sorted, anchor, scale: (TRACK_TARGET_PX - 2 * START_X) / totalDays };
}

/**
 * Lays out one bar per plan on a shared time axis anchored at the first plan's
 * start date and running to `now`. Bars are positioned by real time, but each
 * is at least MIN_SEGMENT_PX wide and back-to-back bars are nudged so they
 * never overlap — a long gap between treatments still shows up as empty track.
 */
export function computeTimelineLayout(
  plans: TreatmentPlanRecord[],
  now: Date = new Date()
): TimelineSegment[] {
  const { sorted, anchor, scale } = timelineScale(plans, now);
  if (sorted.length === 0) return [];

  const segments: TimelineSegment[] = [];
  let prevEndX = -Infinity;

  for (let i = 0; i < sorted.length; i++) {
    const plan = sorted[i];
    const start = planDate(plan) ?? now;
    const end = planEndDate(plan, i, sorted, now);

    const naturalStartX = START_X + ((start.getTime() - anchor.getTime()) / DAY_MS) * scale;
    const startX = Math.max(naturalStartX, prevEndX + GAP_PX);
    const naturalEndX = START_X + ((end.getTime() - anchor.getTime()) / DAY_MS) * scale;
    const endX = Math.max(startX + MIN_SEGMENT_PX, naturalEndX);

    segments.push({ plan, number: i + 1, startX, endX });
    prevEndX = endX;
  }
  return segments;
}

// ── View ─────────────────────────────────────────────────────────────────────

const captionFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
const rangeFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });

function formatRange(plan: TreatmentPlanRecord): string {
  const start = plan.start_date ? rangeFormat.format(parseIsoDate(plan.start_date) as Date) : "—";
  const end = plan.end_date ? rangeFormat.format(parseIsoDate(plan.end_date) as Date) : "Ongoing";
  return `${start} → ${end}`;
}

function formatMeds(plan: TreatmentPlanRecord): string {
  return Object.entries(plan.medications)
    .filter(([, value]) => value > 0)
    .map(([id, value]) => `${id.replace("_mg", "")} ${value} mg`)
    .join(", ");
}

// Bar row sits on the axis at AXIS_Y (px from the container's top).
const AXIS_Y = 34;

/**
 * Horizontal per-patient treatment timeline. Each plan is a bar spanning its
 * start → end date (the latest, still-running plan reaches "Today"), so the
 * length of each bar shows how long that treatment ran. Bars are numbered
 * (matching the plan list badges) and open that plan's results when clicked.
 */
export function TreatmentTimeline({
  plans,
  onOpen,
  latestPlanId
}: {
  plans: TreatmentPlanRecord[];
  onOpen: (plan: TreatmentPlanRecord) => void;
  latestPlanId: string | null;
}) {
  const segments = computeTimelineLayout(plans);
  if (segments.length === 0) return null;

  const last = segments[segments.length - 1];
  const trackWidth = Math.max(last.endX + 40, 240);

  return (
    <div className="mt-4 overflow-x-auto">
      <div className="flex justify-center">
        <div className="relative" style={{ width: trackWidth, height: 68 }}>
          {/* faint baseline the whole history sits on */}
          <div className="absolute left-0 right-0 top-[34px] h-px bg-slate-200 dark:bg-slate-700" />

          {segments.map(({ plan, number, startX, endX }) => {
            const isLatest = plan.id === latestPlanId;
            return (
              <button
                key={plan.id}
                type="button"
                onClick={() => onOpen(plan)}
                title={`Plan ${number} · ${formatRange(plan)}${formatMeds(plan) ? ` · ${formatMeds(plan)}` : ""}`}
                data-testid="treatment-timeline-segment"
                data-plan-id={plan.id}
                data-number={number}
                className="group absolute flex items-center rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5f8459]"
                style={{ left: startX, top: AXIS_Y - 7, width: Math.max(endX - startX, MIN_SEGMENT_PX) }}
              >
                <span
                  className={`h-3.5 w-full rounded-full transition-colors ${
                    isLatest
                      ? "bg-emerald-600 group-hover:bg-emerald-500"
                      : "bg-slate-300 dark:bg-slate-600 group-hover:bg-slate-400 dark:group-hover:bg-slate-500"
                  }`}
                />
                <span
                  className={`absolute flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold ring-2 ring-white dark:ring-slate-900 ${
                    isLatest
                      ? "bg-emerald-600 text-white"
                      : "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  }`}
                  style={{ left: -12, top: -5 }}
                >
                  {number}
                </span>
              </button>
            );
          })}

          {segments.map(({ plan, startX }) => {
            const date = planDate(plan);
            if (!date) return null;
            return (
              <span
                key={`${plan.id}-caption`}
                className="absolute whitespace-nowrap text-[10px] text-slate-400 dark:text-slate-500"
                style={{ left: startX, top: AXIS_Y + 14, transform: "translateX(-50%)" }}
              >
                {captionFormat.format(date)}
              </span>
            );
          })}

          {/* Today marker at the end of the running plan */}
          <div className="absolute flex flex-col items-center" style={{ left: last.endX + 6, top: AXIS_Y - 14 }}>
            <span className="h-2.5 w-px bg-emerald-500" />
            <span className="mt-0.5 whitespace-nowrap text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
              Today
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
