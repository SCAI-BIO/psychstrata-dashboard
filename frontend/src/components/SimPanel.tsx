import { parseNumberValue, adherenceCategoryValue } from "../lib/format";
import type { SimValues } from "../types";

interface DoseSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

/** A labelled dose slider: name + live value on top, full-width track below. */
export function DoseSlider({ label, value, min, max, step, onChange }: DoseSliderProps) {
  return (
    <label className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-slate-500">{label}</span>
        <span className="text-base font-semibold text-slate-900">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value ?? 0}
        onChange={(event) => onChange(parseNumberValue(event.target.value))}
        className="w-full accent-slate-900"
      />
    </label>
  );
}

interface SimControlsProps {
  values: SimValues;
  onChange: (patch: Partial<SimValues>) => void;
  /** Whether to include the Therapeutic Adherence dropdown in the control set. */
  showAdherence?: boolean;
}

/** The shared dose-slider set (Sertraline / Lithium / Quetiapine) + adherence. */
export function SimControls({ values, onChange, showAdherence = true }: SimControlsProps) {
  return (
    <div className="space-y-4">
      <DoseSlider
        label="Sertraline Dosage (mg/day)"
        value={values.sertraline_mg ?? 0}
        min={0}
        max={400}
        step={25}
        onChange={(v) => onChange({ sertraline_mg: v })}
      />
      <DoseSlider
        label="Lithium Dosage (mg/day)"
        value={values.lithium_mg ?? 0}
        min={0}
        max={1200}
        step={50}
        onChange={(v) => onChange({ lithium_mg: v })}
      />
      <DoseSlider
        label="Quetiapine Augmentation (mg/day)"
        value={values.quetiapine_mg ?? 0}
        min={0}
        max={300}
        step={25}
        onChange={(v) => onChange({ quetiapine_mg: v })}
      />
      {showAdherence && (
        <label className="flex flex-col gap-1.5">
          <span className="text-xs text-slate-500">Therapeutic Adherence</span>
          <select
            value={String(adherenceCategoryValue(values.adherence_pct))}
            onChange={(event) => onChange({ adherence_pct: parseNumberValue(event.target.value) })}
            className="w-full h-11 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
>
            <option value="50">Low</option>
            <option value="75">Moderate</option>
            <option value="90">High</option>
          </select>
        </label>
      )}
    </div>
  );
}

interface PredictedImpactProps {
  impactPct: number;
  error?: string | null;
}

/** The blue "Predicted Impact" summary box shown beside the clinician sim. */
export function PredictedImpact({ impactPct, error }: PredictedImpactProps) {
  const sign = impactPct > 0 ? "-" : impactPct < 0 ? "+" : "";
  const magnitude = Math.abs(impactPct).toFixed(1);
  const direction = impactPct > 0 ? "Reduction" : impactPct < 0 ? "Increase" : "Change";

  return (
    <div className="rounded-lg bg-blue-50 border border-blue-100 px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 text-center">Predicted Impact</p>
      <p className="text-sm font-medium text-slate-700 text-center mt-1">
        {sign}
        {magnitude}% Resistance Risk {direction}
      </p>
      {error && <p className="text-[11px] text-red-600 mt-1 text-center">{error}</p>}
    </div>
  );
}
