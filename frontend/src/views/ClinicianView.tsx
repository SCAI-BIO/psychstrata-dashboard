import { BrainCircuit, LineChart as LineChartIcon } from "lucide-react";
import { useMemo } from "react";
import type { PredictionResponse } from "../api";
import { Card } from "../components/Card";
import { InsightsCard } from "../components/cards/InsightsCard";
import { RiskCard } from "../components/cards/RiskCard";
import { StatCards, type StatItem } from "../components/cards/StatCards";
import { ShapChart } from "../components/charts/ShapChart";
import { TimeToEventChart } from "../components/charts/TimeToEventChart";
import { PredictedImpact, SimControls } from "../components/SimPanel";
import { patientToFeatures } from "../domain/patient";
import type { DashboardApi } from "../hooks/useDashboard";
import { useScenario } from "../hooks/useScenario";
import { deriveClinical } from "../lib/derive";
import { adherenceCategoryValue, parseNumberValue, phqSeverity } from "../lib/format";
import type { ReadyState, SimValues } from "../types";

interface ClinicianViewProps {
  dashboard: DashboardApi;
  ready: ReadyState;
  prediction: PredictionResponse;
}

export function ClinicianView({ dashboard, ready, prediction }: ClinicianViewProps) {
  const d = deriveClinical(dashboard.featureValues, prediction);

  // ids the backend model knows — routes draft edits to clinical vs extras
  const knownIds = useMemo(() => new Set(ready.features.map((f) => f.id)), [ready.features]);

  // throwaway copy of the patient; sliders edit this, the real patient is untouched
  const { draft, setFeature } = useScenario(dashboard.patient, knownIds);

  // SimControls speaks SimValues, so project the four fields out of the draft…
  const simValues: SimValues = {
    sertraline_mg: draft.clinical.sertraline_mg ?? 0,
    lithium_mg: draft.clinical.lithium_mg ?? 0,
    quetiapine_mg: draft.clinical.quetiapine_mg ?? 0,
    adherence_pct: draft.clinical.adherence_pct ?? 75
  };
  // …and translate its patches back into draft edits
  const patchDraft = (patch: Partial<SimValues>) =>
    (Object.entries(patch) as [keyof SimValues, number][]).forEach(([id, value]) => setFeature(id, value));

  const statItems: StatItem[] = [
    { label: "PHQ-9 Score", value: d.phqScore, caption: phqSeverity(d.phqScore) },
    { label: "Medication Adherence", value: d.adherenceBucket, caption: "Potential for improvement" },
    {
      label: "Medication Load",
      value: d.medicationLoad >= 900 ? "High" : d.medicationLoad >= 400 ? "Medium" : "Low",
      suffix: `(${d.activeAgents} Agents)`,
      caption: d.medicationLoad >= 900 ? "Complex Polypharmacy" : "Current regimen intensity"
    },
    {
      label: "Lithium Level",
      value: d.lithiumDose,
      suffix: "mg/day",
      caption: d.lithiumDose >= 600 ? "Within augmentation range" : "Below therapeutic range"
    }
  ];

  return (
    <div className="space-y-4">
      <section className="grid grid-cols-[5fr_7fr] gap-4">
        <RiskCard riskProbability={d.riskProbability} isHighRisk={d.isHighRisk} />

        <Card
          icon={BrainCircuit}
          title="Probability of Non-Response Over Time (Time-to-Event)"
          action={
            <div className="flex items-center gap-2">
              <button
                type="button"
                 onClick={() => void dashboard.applyScenario(draft)}
                disabled={dashboard.isClinicianImpactSubmitting || dashboard.isClinicianApplySubmitting}
                className="bg-slate-900 text-white text-xs font-semibold uppercase tracking-wide px-4 py-2 rounded-lg hover:bg-slate-700 disabled:opacity-50 transition-colors"
              >
                {dashboard.isClinicianImpactSubmitting ? "Calculating…" : "Calculate"}
              </button>
            </div>
          }
        >
          <div className="grid grid-cols-2 gap-6">
            <SimControls values={simValues} onChange={patchDraft} showAdherence={false} />
            <div className="space-y-4">
              <label className="flex flex-col gap-1.5">
                <span className="text-xs text-slate-500">Therapeutic Adherence</span>
                <select
                  value={String(adherenceCategoryValue(simValues.adherence_pct))}
                  onChange={(e) => patchDraft({ adherence_pct: parseNumberValue(e.target.value) })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="50">Low</option>
                  <option value="75">Moderate</option>
                  <option value="90">High</option>
                </select>
              </label>
              <PredictedImpact impactPct={dashboard.clinicianAppliedImpactPct} error={dashboard.clinicianApplyError} />
            </div>
          </div>
        </Card>
      </section>

      <InsightsCard
        explanation={ready.explanation}
        isRefreshing={ready.isSummaryRefreshing}
        onRefresh={() => void dashboard.refreshExplanation()}
        error={ready.error}
      />

      <section className="grid grid-cols-2 gap-4">
        <Card icon={LineChartIcon} title="Risk Factor Analysis (SHAP)">
          <ShapChart shapValues={prediction.shap_values} />
        </Card>
        <Card icon={LineChartIcon} title="Probability of Non-Response Over Time (Time-to-Event)">
          <TimeToEventChart
            currentProbability={d.riskProbability}
            simulatedProbability={dashboard.clinicianImpactPct !== 0 ? d.riskProbability - dashboard.clinicianImpactPct / 100 : null}
          />
        </Card>
      </section>

      <StatCards items={statItems} />
    </div>
  );
}