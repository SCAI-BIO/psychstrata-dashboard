import { AlertTriangle, CheckCircle2, ShieldCheck } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import type { PredictionResponse } from "../api";
import { usePatient } from "../context/PatientContext";
import { firstName as patientFirstName, patientToFeatures } from "../domain/patient";
import { Card } from "../components/Card";
import { TimeToEventChart } from "../components/charts/TimeToEventChart";
import { ResponseBadge, RiskGaugeBar } from "../components/indicators";
import { SimControls } from "../components/SimPanel";
import type { DashboardApi } from "../hooks/useDashboard";
import { useScenario } from "../hooks/useScenario";
import { deriveClinical } from "../lib/derive";
import { pct, responseBadgeSpec } from "../lib/format";
import type { ReadyState, SimValues } from "../types";

type Contributor = PredictionResponse["top_contributors"]["negative"][number];

interface PatientViewProps {
  dashboard: DashboardApi;
  ready: ReadyState;
  prediction: PredictionResponse;
}

/** "Patient View" — plain-language, patient-facing dashboard. */
export function PatientView({ dashboard, ready, prediction }: PatientViewProps) {
  const { patient } = usePatient();
  const d = deriveClinical(dashboard.featureValues, prediction);

  // ids the backend model knows — routes draft edits to clinical vs extras
  const knownIds = useMemo(() => new Set(ready.features.map((f) => f.id)), [ready.features]);

  // throwaway copy of the patient the simulator edits; the real patient is untouched
  const { draft, setFeature } = useScenario(patient, knownIds);

  const simValues: SimValues = {
    sertraline_mg: draft.clinical.sertraline_mg ?? 0,
    lithium_mg: draft.clinical.lithium_mg ?? 0,
    quetiapine_mg: draft.clinical.quetiapine_mg ?? 0,
    adherence_pct: draft.clinical.adherence_pct ?? 75
  };
  const patchSim = (patch: Partial<SimValues>) =>
    (Object.entries(patch) as [keyof SimValues, number][]).forEach(([id, value]) => setFeature(id, value));

  // Scenario response derived from the shared impact figure (0 until "Calculate" runs).
  const simulatedResistance = Math.min(1, Math.max(0, d.riskProbability - dashboard.clinicianImpactPct / 100));
  const displayedResponse = 1 - simulatedResistance;
  const responseBadge = responseBadgeSpec(displayedResponse);

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200/70 border-l-4 border-l-slate-900 px-8 py-7">
        <h1 className="text-4xl font-bold tracking-tight text-slate-900 mb-3">Welcome, {patientFirstName(patient)}.</h1>
        <p className="text-slate-500 text-sm max-w-2xl leading-relaxed">
          We've reviewed your recent clinical assessments. This portal is here to help you navigate your journey with
          depression and find the most effective strategies for long-term treatment.
        </p>
      </div>

      <section className="grid grid-cols-2 gap-4">
        <ContributorCard
          tone="strength"
          title="Your Strengths"
          icon={<ShieldCheck size={18} className="text-emerald-800" />}
          items={d.strengths.slice(0, 3)}
        />
        <ContributorCard
          tone="action"
          title="Action Items"
          icon={<AlertTriangle size={18} className="text-red-900" />}
          items={d.actionItems.slice(0, 3)}
        />
      </section>

      <Card title="Your Treatment Path Simulator">
        <div className="grid grid-cols-[1.4fr_1fr] gap-8">
          <div>
            <p className="text-sm text-slate-500 leading-relaxed mb-5">
              Adjust these clinical variables to see how they might influence the estimated treatment response over time.
            </p>
            <SimControls values={simValues} onChange={patchSim} />
            <button
              type="button"
              onClick={() => void dashboard.runScenarioImpact(patientToFeatures(draft))}
              disabled={dashboard.isClinicianImpactSubmitting}
              className="mt-6 bg-slate-900 text-white text-sm font-semibold px-6 py-2.5 rounded-lg hover:bg-slate-700 disabled:opacity-50 transition-colors"
            >
              {dashboard.isClinicianImpactSubmitting ? "Calculating…" : "Calculate"}
            </button>
          </div>

          <div className="rounded-xl border border-slate-200/70 bg-slate-50/60 p-6 flex flex-col items-center text-center">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">Estimated Treatment Response</p>
            <ResponseBadge label={responseBadge.label} classes={responseBadge.classes} />
            <p className="text-5xl font-bold text-slate-900 tracking-tight mt-3">{pct(displayedResponse)}</p>
            <div className="w-full max-w-[220px]">
              <RiskGaugeBar probability={displayedResponse} inverse={true} />
            </div>
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">
              This indicates the predicted likelihood of treatment response based on your individual factors.
            </p>
            <div className="w-full mt-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-1 text-center">Response Over Time</p>
              <TimeToEventChart
                currentProbability={d.riskProbability}
                simulatedProbability={simulatedResistance}
                currentLabel="Current Care Plan"
                simulatedLabel="Adapted Care Plan"
                height={140}
              />
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

function ContributorCard({
  tone,
  title,
  icon,
  items
}: {
  tone: "strength" | "action";
  title: string;
  icon: ReactNode;
  items: Contributor[];
}) {
  const isStrength = tone === "strength";
  const headerClass = isStrength ? "bg-emerald-100 text-emerald-800" : "bg-red-300/70 text-red-900";
  const chipClass = isStrength ? "bg-emerald-100 text-emerald-800" : "bg-red-300/70 text-red-900";

  const RowIcon = isStrength ? CheckCircle2 : AlertTriangle;

  return (
    <article className="rounded-xl border border-slate-200/70 overflow-hidden">
      <header className={`flex items-center justify-between px-6 py-3.5 ${headerClass}`}>
        <h2 className="text-base font-bold">{title}</h2>
        {icon}
      </header>
      <div className="bg-white p-6 space-y-4">
        {items.length === 0 && <p className="text-sm text-slate-400">No factors identified.</p>}
        {items.map((item) => (
          <div key={item.feature_id} className="flex items-start gap-3">
            <span className={`flex-none flex items-center justify-center w-8 h-8 rounded-lg ${chipClass}`}>
              <RowIcon size={16} />
            </span>
            <div className="leading-snug">
              <p className="text-sm font-semibold text-slate-900">{item.feature_label}</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Recorded as {String(item.selected_value)}.{" "}
                {isStrength
                  ? "This is associated with a better treatment response."
                  : "This is associated with a more challenging treatment response."}
              </p>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}
