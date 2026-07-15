/** Colour stops from low→high (good→bad). Reversed when `inverse` is set. */
const GAUGE_STOPS = ["#2f7d4f", "#d9c37a", "#d98a7a"];

function gaugeGradient(inverse: boolean): string {
  const stops = inverse ? [...GAUGE_STOPS].reverse() : GAUGE_STOPS;
  const parts = stops.map((color, i) => `${color} ${(i / (stops.length - 1)) * 100}%`);
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

interface RiskGaugeBarProps {
  probability: number;
  /** Flip the colour meaning (green on the right) — e.g. "response", where higher is better. */
  inverse?: boolean;
}

export function RiskGaugeBar({ probability, inverse = false }: RiskGaugeBarProps) {
  const value = Math.min(100, Math.max(0, Math.round(probability * 100)));
  return (
    <div className="mt-4">
      <div
        className="relative h-2.5 w-full rounded-full overflow-visible"
        style={{ background: gaugeGradient(inverse) }}
      >
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-white border-2 border-slate-800 shadow-md"
          style={{ left: `calc(${value}% - 8px)` }}
        />
      </div>
    </div>
  );
}

export function RiskBadge({ isHighRisk }: { isHighRisk: boolean }) {
  return (
    <span
      className={
        isHighRisk
          ? "bg-red-300/70 text-red-900 text-xs font-bold uppercase tracking-wide px-3 py-1.5 rounded-md"
          : "bg-emerald-200/70 text-emerald-900 text-xs font-bold uppercase tracking-wide px-3 py-1.5 rounded-md"
      }
    >
      {isHighRisk ? "High Risk" : "Lower Risk"}
    </span>
  );
}

export function ResponseBadge({ label, classes }: { label: string; classes: string }) {
  return (
    <span className={`text-[11px] font-bold uppercase tracking-wide px-3 py-1.5 rounded-md ${classes}`}>
      {label}
    </span>
  );
}
