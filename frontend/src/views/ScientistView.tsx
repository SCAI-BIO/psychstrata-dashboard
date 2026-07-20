import { LineChart as LineChartIcon, Users } from "lucide-react";
import type { PredictionResponse } from "../api";
import { Card } from "../components/Card";
import { InsightsCard } from "../components/cards/InsightsCard";
import { RiskCard } from "../components/cards/RiskCard";
import { ShapChart } from "../components/charts/ShapChart";
import { TimeToEventChart } from "../components/charts/TimeToEventChart";
import { TsneChart } from "../components/charts/TsneChart";
import type { DashboardApi } from "../hooks/useDashboard";
import { deriveClinical } from "../lib/derive";
import type { ReadyState } from "../types";

interface ScientistViewProps {
  dashboard: DashboardApi;
  ready: ReadyState;
  prediction: PredictionResponse;
}

/** "Scientific View" — model-facing dashboard with certainty control + t-SNE map. */
export function ScientistView({ dashboard, ready, prediction }: ScientistViewProps) {
  const d = deriveClinical(dashboard.featureValues, prediction);

  return (
    <div className="space-y-4">
      <section className="grid grid-cols-[5fr_7fr] gap-4">
        <RiskCard
          riskProbability={d.riskProbability}
          isHighRisk={d.isHighRisk}
          confidence={{
            value: ready.confidenceLevel,
            min: ready.confidenceBounds.min,
            max: ready.confidenceBounds.max,
            step: ready.confidenceBounds.step,
            onChange: dashboard.setConfidenceLevel,
            onCommit: () => void dashboard.runPrediction("scientist")
          }}
        />
        <Card icon={LineChartIcon} title="Probability of Non-Response Over Time (Time-to-Event)">
          <TimeToEventChart currentProbability={d.riskProbability} currentLabel="Current Care Plan" height={260} />
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
        <Card icon={Users} title="t-SNE Population Map">
          {ready.tsne ? (
            <TsneChart tsne={ready.tsne} selected={d.selectedTsne} />
          ) : (
            <p className="text-sm text-slate-500">t-SNE data unavailable.</p>
          )}
        </Card>
      </section>
    </div>
  );
}
