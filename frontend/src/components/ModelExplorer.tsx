import { RotateCcw, SlidersHorizontal } from "lucide-react";
import type { FeatureSchema, PredictionResponse } from "../api";
import { ShapChart } from "./charts/ShapChart";
import { Card, SectionLabel } from "./Card";
import { FeatureField } from "./FeatureField";
import { useModelExplorer } from "../hooks/useModelExplorer";
import { RiskBadge, RiskGaugeBar } from "./indicators";
import { pct } from "../lib/format";

interface ConfidenceControl {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Fired when the user finishes dragging (mouse/touch/key release). */
  onCommit?: () => void;
}

interface ModelExplorerProps {
  features: FeatureSchema[];
  baselineFeatures: Record<string, number>;
  baselinePrediction: PredictionResponse;
  confidenceLevel: number;
  dateOfBirth?: string | null;
  /** Optional model-certainty slider (scientific view). */
  confidence?: ConfidenceControl;
}

const GROUP_LABELS: Record<FeatureSchema["category"], string> = {
  clinical: "Clinical Factors",
  medications: "Medication",
  adherence: "Adherence"
};

const GROUP_ORDER: FeatureSchema["category"][] = ["clinical", "medications", "adherence"];

/**
 * Read-only "play with the model" sandbox for the Scientific View. Lets the
 * user drag any model feature and see the predicted resistance probability and
 * SHAP feature importances update (debounced) against the backend — without
 * ever changing the real patient.
 */
export function ModelExplorer({
  features,
  baselineFeatures,
  baselinePrediction,
  confidenceLevel,
  dateOfBirth,
  confidence
}: ModelExplorerProps) {
  const { draft, result, isSubmitting, error, setValue, reset } = useModelExplorer({
    baselineFeatures,
    features,
    confidenceLevel,
    dateOfBirth
  });

  const effective = result ?? baselinePrediction;
  const risk = effective.prediction.probability_resistance;
  const baselineRisk = baselinePrediction.prediction.probability_resistance;
  const isHighRisk = effective.prediction.predicted_class === "Resistant";
  const deltaPts = (risk - baselineRisk) * 100;
  const deltaText =
    deltaPts === 0
      ? "Same risk as patient baseline"
      : `${deltaPts > 0 ? "▲" : "▼"} ${Math.abs(deltaPts).toFixed(1)} pts ${
          deltaPts > 0 ? "higher" : "lower"
        } risk vs patient baseline`;

  const groups = GROUP_ORDER.map((category) => ({
    category,
    label: GROUP_LABELS[category],
    items: features.filter((f) => f.category === category)
  })).filter((g) => g.items.length > 0);

  return (
    <Card
      icon={SlidersHorizontal}
      title="Model Explorer"
      action={
        <button
          type="button"
          onClick={reset}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 bg-slate-900 text-white text-xs font-semibold uppercase tracking-wide px-3 py-2 rounded-lg hover:bg-slate-700 disabled:opacity-50 transition-colors"
        >
          <RotateCcw size={14} />
          Reset to patient
        </button>
      }
      bodyClassName="grid grid-cols-[5fr_7fr] gap-6"
    >
      <div className="space-y-5">
        {groups.map((group) => (
          <div key={group.category}>
            <SectionLabel>{group.label}</SectionLabel>
            <div className="space-y-3">
              {group.items.map((feature) => (
                <FeatureField
                  key={feature.id}
                  feature={feature}
                  value={draft[feature.id] ?? feature.default}
                  onChange={setValue}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-5">
        <div>
          <SectionLabel>Predicted Outcome</SectionLabel>
          <div className="rounded-lg border border-slate-200 dark:border-slate-700 px-4 py-4">
            <p className="text-5xl font-bold text-slate-900 dark:text-slate-100 text-center">{pct(risk)}</p>
            <RiskGaugeBar probability={risk} />
            <div className="flex items-center justify-center gap-2 mt-4">
              <RiskBadge isHighRisk={isHighRisk} />
              <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                {effective.prediction.conformal_prediction.label}
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 text-center mt-3">{deltaText}</p>
            {confidence && (
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Model Certainty — Confidence Interval (%)
                </span>
                <input
                  type="range"
                  aria-label="Model Certainty — Confidence Interval (%)"
                  min={confidence.min}
                  max={confidence.max}
                  step={confidence.step}
                  value={confidence.value}
                  onChange={(event) => confidence.onChange(Number(event.target.value))}
                  onMouseUp={() => confidence.onCommit?.()}
                  onTouchEnd={() => confidence.onCommit?.()}
                  onKeyUp={() => confidence.onCommit?.()}
                  className="mt-2 w-full accent-slate-900 dark:accent-slate-100"
                />
                <div className="flex justify-between text-[11px] text-slate-400 dark:text-slate-500">
                  <span>{confidence.min}</span>
                  <span>{confidence.value}</span>
                  <span>{confidence.max}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <SectionLabel>Feature Importance (SHAP)</SectionLabel>
          {isSubmitting ? (
            <p className="text-sm text-slate-400 py-4 text-center">Updating prediction…</p>
          ) : (
            <ShapChart shapValues={effective.shap_values} />
          )}
        </div>
        {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
      </div>
    </Card>
  );
}
