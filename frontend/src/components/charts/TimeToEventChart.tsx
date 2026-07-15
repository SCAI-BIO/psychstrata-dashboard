import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  XAxis,
  YAxis
} from "recharts";
import { CLINICAL_REVIEW_MONTH, buildMonthlyNonResponseCurve } from "../../lib/survival";

interface TimeToEventChartProps {
  /** Resistance probability (0–1) driving the primary care-plan curve. */
  currentProbability: number;
  /** Optional simulated plan; when supplied, renders a second solid line. */
  simulatedProbability?: number | null;
  currentLabel?: string;
  simulatedLabel?: string;
  height?: number;
}

const AXIS_COLOR = "#94a3b8";

function monthTick(value: number): string {
  return value === 0 ? "Start" : `${value} months`;
}

/**
 * "Probability of Non-Response Over Time (Time-to-Event)" line chart.
 * Mirrors the mockups: month axis (Start → 15 months), a current-plan line,
 * an optional simulated-plan line, and the red Clinical Review Point marker.
 */
export function TimeToEventChart({
  currentProbability,
  simulatedProbability = null,
  currentLabel = "Current Care Plan",
  simulatedLabel = "Simulated Care Plan",
  height = 220
}: TimeToEventChartProps) {
  const hasSimulated = simulatedProbability !== null && simulatedProbability !== undefined;
  const data = buildMonthlyNonResponseCurve(currentProbability, simulatedProbability);

  // The review point sits on the "primary" plan (simulated if present).
  const reviewPoint = data.find((d) => d.month === CLINICAL_REVIEW_MONTH);
  const reviewValue = reviewPoint ? (hasSimulated ? reviewPoint.simulated : reviewPoint.current) : null;

  return (
    <div className="flex flex-col">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="4 4" stroke="#eef2f6" vertical={false} />
          <XAxis
            dataKey="month"
            type="number"
            domain={[0, 15]}
            ticks={[0, 3, 6, 9, 12, 15]}
            tickFormatter={monthTick}
            tick={{ fontSize: 10, fill: AXIS_COLOR }}
            axisLine={{ stroke: "#e2e8f0" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0.25, 0.5, 0.75, 1.0]}
            tickFormatter={(v: number) => v.toFixed(2)}
            tick={{ fontSize: 10, fill: AXIS_COLOR }}
            axisLine={false}
            tickLine={false}
            width={34}
          />

          {/* Current care plan — dashed grey when a simulated overlay exists, else solid black */}
          <Line
            type="monotone"
            dataKey="current"
            stroke={hasSimulated ? "#cbd5e1" : "#0f172a"}
            strokeWidth={hasSimulated ? 2 : 2.5}
            strokeDasharray={hasSimulated ? "6 5" : undefined}
            dot={false}
            isAnimationActive={false}
          />

          {/* Simulated care plan — solid black */}
          {hasSimulated && (
            <Line
              type="monotone"
              dataKey="simulated"
              stroke="#0f172a"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          )}

          {reviewValue !== null && reviewValue !== undefined && (
            <ReferenceDot
              x={CLINICAL_REVIEW_MONTH}
              y={reviewValue}
              r={5}
              fill="#dc2626"
              stroke="#fff"
              strokeWidth={1.5}
              isFront
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-1.5 mt-3 justify-center">
        <LegendItem color={hasSimulated ? "#cbd5e1" : "#0f172a"} label={currentLabel} dashed={hasSimulated} />
        {hasSimulated && <LegendItem color="#0f172a" label={simulatedLabel} />}
        <LegendItem color="#dc2626" label="Clinical Review Point" dot />
      </div>
    </div>
  );
}

function LegendItem({ color, label, dashed, dot }: { color: string; label: string; dashed?: boolean; dot?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      {dot ? (
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      ) : (
        <svg width="18" height="8" aria-hidden>
          <line
            x1="0"
            y1="4"
            x2="18"
            y2="4"
            stroke={color}
            strokeWidth="2.5"
            strokeDasharray={dashed ? "5 4" : undefined}
          />
        </svg>
      )}
      <span className="text-[11px] text-slate-500">{label}</span>
    </div>
  );
}
