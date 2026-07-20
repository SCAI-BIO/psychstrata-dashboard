import type { PredictionResponse } from "../../api";
import { SectionLabel } from "../Card";

type ShapValues = PredictionResponse["shap_values"];
interface ShapRow {
  name: string;
  value: number;
}

const MEDICATION_KEYWORDS = [
  "dose",
  "dosage",
  "adherence",
  "side effect",
  "sertraline",
  "lithium",
  "quetiapine",
  "medication",
  "crp",
  "il-6",
  "il6",
  "tnf",
  "cyp2d6",
  "proteomic",
  "genotype"
];

function isMedicationFactor(label: string): boolean {
  const lower = label.toLowerCase();
  return MEDICATION_KEYWORDS.some((keyword) => lower.includes(keyword));
}

/** Split raw SHAP values into the two labelled groups shown in the mockups. */
function partition(shapValues: ShapValues): { clinical: ShapRow[]; medication: ShapRow[] } {
  const clinical: ShapRow[] = [];
  const medication: ShapRow[] = [];
  shapValues.forEach((entry) => {
    const row = { name: entry.feature_label, value: +entry.shap_value.toFixed(2) };
    (isMedicationFactor(entry.feature_label) ? medication : clinical).push(row);
  });
  const byMagnitude = (a: ShapRow, b: ShapRow) => Math.abs(b.value) - Math.abs(a.value);
  return { clinical: clinical.sort(byMagnitude), medication: medication.sort(byMagnitude) };
}

/**
 * Grouped SHAP contribution chart. Each row is a diverging bar around a centre
 * axis: bars to the right (positive) increase risk, bars to the left decrease
 * it. Colours follow the mockup — muted rose for positive, green for negative.
 */
export function ShapChart({ shapValues }: { shapValues: ShapValues }) {
  const { clinical, medication } = partition(shapValues);
  const maxAbs = Math.max(0.001, ...shapValues.map((e) => Math.abs(e.shap_value)));

  return (
    <div className="space-y-5">
      {clinical.length > 0 && (
        <div>
          <SectionLabel>Clinical Factors</SectionLabel>
          <ShapRows rows={clinical} maxAbs={maxAbs} />
        </div>
      )}
      {medication.length > 0 && (
        <div>
          <SectionLabel>Medication &amp; Bio-Markers</SectionLabel>
          <ShapRows rows={medication} maxAbs={maxAbs} />
        </div>
      )}
      <p className="text-[11px] text-slate-400 leading-relaxed text-center pt-1">
        SHAP values indicate clinical and biological variables influencing resistance prediction. Bars to the left indicate a
        decrease in risk due to the factor, while bars to the right indicate an increase.
      </p>
    </div>
  );
}

function ShapRows({ rows, maxAbs }: { rows: ShapRow[]; maxAbs: number }) {
  return (
    <div className="space-y-2">
      {rows.map((row) => (
        <ShapBar key={row.name} row={row} maxAbs={maxAbs} />
      ))}
    </div>
  );
}

function ShapBar({ row, maxAbs }: { row: ShapRow; maxAbs: number }) {
  const positive = row.value >= 0;
  const widthPct = (Math.abs(row.value) / maxAbs) * 48; // half-width, leave margin
  const fill = positive
    ? "linear-gradient(to right, #c9b7b0, #cf7a6d)"
    : "linear-gradient(to left, #a9c3a1, #b9c7b2)";

  return (
    <div className="grid grid-cols-[120px_1fr_44px] items-center gap-2">
      <span className="text-xs text-slate-600 truncate" title={row.name}>
        {row.name}
      </span>
      <div className="relative h-4">
        <div className="absolute inset-y-0 left-1/2 w-px bg-slate-200" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 rounded-sm"
          style={{
            background: fill,
            width: `${widthPct}%`,
            left: positive ? "50%" : undefined,
            right: positive ? undefined : "50%"
          }}
        />
      </div>
      <span className={`text-xs font-medium text-right ${positive ? "text-rose-700" : "text-slate-500"}`}>
        {positive ? "+" : ""}
        {row.value.toFixed(2)}
      </span>
    </div>
  );
}
