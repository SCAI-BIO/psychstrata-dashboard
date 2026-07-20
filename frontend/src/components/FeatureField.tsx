import type { FeatureSchema } from "../api";
import { parseNumberValue } from "../lib/format";

interface FeatureFieldProps {
  feature: FeatureSchema;
  value: number;
  onChange: (featureId: string, value: number) => void;
  compact?: boolean;
}

/**
 * Renders a single backend-described feature as either a range slider
 * (numeric) or a select (categorical), driven entirely by the schema.
 */
export function FeatureField({ feature, value, onChange }: FeatureFieldProps) {
  const inputId = `feature-${feature.id}`;
  const min = feature.min ?? feature.params.min ?? 0;
  const max = feature.max ?? feature.params.max ?? 100;
  const step = feature.step ?? feature.params.step ?? 1;

  return (
    <label className="flex flex-col gap-1.5" htmlFor={inputId}>
      <span className="text-xs font-medium text-slate-600">{feature.label}</span>
      {feature.kind === "numeric" ? (
        <div className="flex items-center gap-3">
          <input
            id={inputId}
            type="range"
            min={min}
            max={max}
            step={step}
            value={value ?? 0}
            onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}
            className="w-full accent-slate-900"
          />
          <output className="text-sm font-medium text-slate-800 min-w-10 text-right">{value}</output>
        </div>
      ) : (
        <select
          id={inputId}
          value={String(value) ?? ""}
          onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}
          className="w-full h-11 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {(feature.options ?? feature.params.options ?? []).map((option) => (
            <option key={option.value} value={String(option.value)}>
              {option.label}
            </option>
          ))}
        </select>
      )}
    </label>
  );
}
