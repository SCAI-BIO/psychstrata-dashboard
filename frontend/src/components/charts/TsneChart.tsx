import { Suspense, lazy } from "react";
import type { TsneResponse } from "../../api";

const Plot = lazy(() => import("react-plotly.js"));

const RESPONSIVE_COLOR = "#6b9e6b";
const RESISTANT_COLOR = "#c0705f";
const PATIENT_COLOR = "#111827";

interface TsneChartProps {
  tsne: TsneResponse;
  selected: { x: number; y: number };
}

/**
 * t-SNE population map: every training patient projected to 2-D, coloured by
 * outcome class, with the current patient overlaid as a solid black marker.
 */
export function TsneChart({ tsne, selected }: TsneChartProps) {
  return (
    <div className="flex flex-col">
      <Suspense
        fallback={<div className="h-64 flex items-center justify-center text-slate-400 text-sm">Loading chart…</div>}
      >
        <Plot
          data={[
            {
              x: tsne.points.map((p) => p.x),
              y: tsne.points.map((p) => p.y),
              mode: "markers",
              type: "scattergl",
              marker: {
                color: tsne.points.map((p) => (p.class_value === 1 ? RESISTANT_COLOR : RESPONSIVE_COLOR)),
                size: 6,
                opacity: 0.8
              },
              showlegend: false
            },
            {
              x: [selected.x],
              y: [selected.y],
              mode: "markers",
              type: "scattergl",
              marker: { color: PATIENT_COLOR, size: 13 },
              name: "Current Patient",
              showlegend: false
            }
          ]}
          layout={{
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            margin: { t: 8, b: 32, l: 34, r: 12 },
            xaxis: {
              zeroline: false,
              showgrid: false,
              tickfont: { size: 10, color: "#94a3b8" },
              linecolor: "#e2e8f0"
            },
            yaxis: {
              zeroline: false,
              showgrid: false,
              tickfont: { size: 10, color: "#94a3b8" },
              linecolor: "#e2e8f0"
            },
            autosize: true
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: "300px" }}
          useResizeHandler
        />
      </Suspense>

      <div className="flex items-center justify-center gap-6 mt-2">
        <LegendDot color={PATIENT_COLOR} label="Current Patient" />
        <LegendDot color={RESPONSIVE_COLOR} label="Responsive Patient" />
        <LegendDot color={RESISTANT_COLOR} label="Resistant Patient" />
      </div>
      <p className="text-[11px] text-slate-400 text-center mt-2">This map places similar patient profiles close together.</p>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      <span className="text-[11px] text-slate-500">{label}</span>
    </div>
  );
}
