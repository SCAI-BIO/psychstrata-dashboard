import { FileText, Sparkles } from "lucide-react";
import { PUBMED_LINKS, SUPPORTING_EVIDENCE } from "../../constants";
import { renderSummaryMarkdown } from "../../lib/markdown";
import { Card } from "../Card";

interface InsightsCardProps {
  explanation: string;
  isRefreshing: boolean;
  onRefresh: () => void;
  error?: string | null;
}

/**
 * "AI Clinical Insights & Literature Review" — LLM explanation on the left with
 * PubMed reference chips, static supporting-evidence citations on the right.
 */
export function InsightsCard({ explanation, isRefreshing, onRefresh, error }: InsightsCardProps) {
  const pill = (
    <button
      type="button"
      onClick={onRefresh}
      disabled={isRefreshing}
      title="Regenerate the plain-language explanation for the current profile."
      className="text-[11px] font-medium text-slate-500 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 px-3 py-1.5 rounded-md transition-colors"
    >
      {isRefreshing ? "Updating…" : "LLM-Generated Explanation"}
    </button>
  );

  return (
    <Card icon={Sparkles} title="AI Clinical Insights & Literature Review" action={pill}>
      <div className="grid grid-cols-[1.9fr_1fr] gap-6">
        <div>
          {explanation ? (
            isRefreshing ? (
              <p className="text-sm text-slate-500">Refreshing explanation for the current profile…</p>
            ) : (
              renderSummaryMarkdown(explanation)
            )
          ) : (
            <div className="flex flex-col items-start gap-3">
              <p className="text-sm text-slate-500 leading-relaxed">
                Generate a plain-language summary of the drivers behind this prediction, grounded in the current patient profile.
              </p>
              <button
                type="button"
                onClick={onRefresh}
                disabled={isRefreshing}
                className="bg-slate-900 text-white text-xs font-medium px-3 py-2 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50"
              >
                {isRefreshing ? "Generating…" : "Generate summary"}
              </button>
            </div>
          )}
          {error && <p className="text-xs text-red-600 mt-3">{error}</p>}
        </div>
      </div>
    </Card>
  );
}
