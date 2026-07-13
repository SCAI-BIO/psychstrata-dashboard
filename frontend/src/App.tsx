import { Fragment, type ReactNode, Suspense, lazy, useEffect, useState } from "react";
import {
  Stethoscope,
  FlaskConical,
  User,
  PlusCircle,
  CheckCircle2,
  AlertCircle,
  LogOut
} from "lucide-react";
import {
  ComposedChart,
  Line,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from "recharts";
import {
  type FeatureSchema,
  type PredictionResponse,
  type TsneResponse,
  fetchExplain,
  fetchFeatures,
  fetchPredict,
  fetchTsne
} from "./api";
import "./App.css";

const Plot = lazy(() => import("react-plotly.js"));

type Route = "intake" | "patient" | "clinician" | "scientist";
const PSYCH_STRATA_LOGO_URL = "https://psych-strata.eu/wp-content/uploads/2023/05/logo_footer_blue.png";
const AUTH_SESSION_KEY = "psychstrata-authenticated";

type LoadState =
  | { status: "loading" }
  | {
      status: "ready";
      features: FeatureSchema[];
      confidenceBounds: { min: number; max: number; step: number };
      confidenceLevel: number;
      featureValues: Record<string, number>;
      prediction: PredictionResponse | null;
      explanation: string;
      tsne: TsneResponse | null;
      isSubmitting: boolean;
      isSummaryRefreshing: boolean;
      error: string | null;
    }
  | { status: "error"; message: string };

const ROUTE_TO_PATH: Record<Route, string> = {
  intake: "/",
  patient: "/results/patient",
  clinician: "/results/clinician",
  scientist: "/results/scientist"
};

function getConfiguredPassword(): string {
  return (import.meta.env.VITE_APP_PASSWORD ?? "").trim();
}

function getIsAuthEnabled(): boolean {
  return getConfiguredPassword().length > 0;
}

function hasAuthenticatedSession(): boolean {
  return sessionStorage.getItem(AUTH_SESSION_KEY) === "true";
}

function getInitialRoute(): Route {
  const pathname = window.location.pathname;
  if (pathname === "/results/patient") return "patient";
  if (pathname === "/results/clinician") return "clinician";
  if (pathname === "/results/scientist") return "scientist";
  return "intake";
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function parseNumberValue(raw: string): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

function supportiveSummary(probability: number): string {
  if (probability >= 0.65) {
    return "Your profile suggests this treatment plan may need closer follow-up and adjustment.";
  }
  if (probability >= 0.4) {
    return "Your profile suggests a mixed outlook, and regular check-ins can help improve response.";
  }
  return "Your profile suggests a favorable chance of response with the current treatment plan.";
}

function renderInlineMarkdown(text: string): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    const match = part.match(/^\*\*([^*]+)\*\*$/);
    if (match) {
      return <strong key={`strong-${index}`}>{match[1]}</strong>;
    }
    return <Fragment key={`text-${index}`}>{part}</Fragment>;
  });
}

function renderSummaryMarkdown(summary: string): ReactNode {
  const lines = summary.split("\n");
  const blocks: ReactNode[] = [];
  let listItems: ReactNode[] = [];
  let listKey = 0;

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push(
        <ul key={`list-${listKey}`} className="space-y-1.5 text-sm text-slate-700 list-none pl-0 mt-2">
          {listItems}
        </ul>
      );
      listItems = [];
      listKey += 1;
    }
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      return;
    }

    if (line.startsWith("- ")) {
      listItems.push(<li key={`item-${index}`}>{renderInlineMarkdown(line.slice(2))}</li>);
      return;
    }

    flushList();
    const headingMatch = line.match(/^\*\*(.+?)\*\*:?\s*$/);
    if (headingMatch) {
      blocks.push(
        <h4 key={`heading-${index}`} className="text-sm font-semibold text-slate-900 mt-4 mb-1">
          {headingMatch[1]}
        </h4>
      );
      return;
    }

    blocks.push(
      <p key={`paragraph-${index}`} className="text-sm text-slate-700 leading-relaxed">
        {renderInlineMarkdown(line)}
      </p>
    );
  });

  flushList();
  return <div className="text-sm text-slate-700 leading-relaxed space-y-2">{blocks}</div>;
}

function FeatureField(props: {
  feature: FeatureSchema;
  value: number;
  onChange: (featureId: string, value: number) => void;
  compact?: boolean;
}) {
  const { feature, value, onChange } = props;
  const inputId = `feature-${feature.id}`;
  const min = feature.min ?? feature.params.min ?? 0;
  const max = feature.max ?? feature.params.max ?? 100;
  const step = feature.step ?? feature.params.step ?? 1;

  return (
    <label className="flex flex-col gap-1" htmlFor={inputId}>
      <span className="text-xs font-medium text-slate-600">{feature.label}</span>
      {feature.kind === "numeric" ? (
        <div className="flex items-center gap-3">
          <input
            id={inputId}
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}
            className="w-full accent-slate-900"
          />
          <output className="text-sm text-slate-700 min-w-10 text-right">{value}</output>
        </div>
      ) : (
        <select
          id={inputId}
          value={String(value)}
          onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}
          className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
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

// ── Clinical & Model Rationale ─────────────────────────────────────────────
//
// 1. CLINICAL: MDD non-response accumulation is CONCAVE (decelerating).
//    Most failures occur in Trials 1–2, with diminishing new failures each
//    subsequent trial. Source: STAR*D (Rush et al., 2006, Am J Psychiatry).
//    The previous exponential curve was clinically incorrect.
//
// 2. FUTURE MODEL: A DeepHit survival model trained on UKBB data will replace
//    the current RF classifier. DeepHit outputs a per-patient CIF:
//      F(t) = P(treatment resistance by week t)
//    at discrete time horizons, plus bootstrap 95% CI bands.
//
// 3. MIGRATION PLAN: This component already accepts data in DeepHit output
//    format (SurvivalPoint[]). When DeepHit is integrated, only
//    buildMockDeepHitCurve() needs to be replaced — the chart is untouched.
//
// 4. CURRENT MOCK: The single RF classifier probability p is scaled onto the
//    STAR*D population CIF shape to produce a clinically plausible curve that
//    visually matches expected DeepHit output.
// ──────────────────────────────────────────────────────────────────────────

// DeepHit output schema — do not rename fields (used for future integration)
interface SurvivalPoint {
  week: number; // discrete time horizon (DeepHit time bin)
  cif: number; // patient F(t): cumulative probability of TR by `week`
  cifLower: number; // 95% CI lower bound
  cifUpper: number; // 95% CI upper bound
  population: number; // STAR*D / UKBB Kaplan-Meier reference CIF
}

// STAR*D population CIF reference
// Approximate cumulative non-response at each treatment trial endpoint
const STAERD_POPULATION_CIF: { week: number; cif: number }[] = [
  { week: 0, cif: 0.0 },
  { week: 8, cif: 0.38 }, // end of Trial 1
  { week: 16, cif: 0.51 }, // end of Trial 2
  { week: 24, cif: 0.61 }, // end of Trial 3
  { week: 36, cif: 0.67 } // TRD threshold (≥4 failed trials)
];

// CI half-widths widen over time — mirrors survival model uncertainty fan.
// Replace with DeepHit bootstrap CI values when model is integrated.
const MOCK_CI_HALF_WIDTH = [0.0, 0.04, 0.07, 0.09, 0.11];

const STAERD_TERMINAL_CIF = 0.67; // population median at Week 36

/**
 * Builds a mock CIF curve in DeepHit output format.
 * Scales the STAR*D population shape so the patient curve
 * terminates at the RF classifier probability at Week 36.
 *
 * MIGRATION: replace this function body with DeepHit output mapping.
 * Keep the SurvivalPoint[] return type and field names unchanged.
 */
function buildMockDeepHitCurve(riskProbability: number): SurvivalPoint[] {
  const scale = riskProbability / STAERD_TERMINAL_CIF;
  return STAERD_POPULATION_CIF.map((pt, i) => {
    const cif = parseFloat(Math.min(1.0, pt.cif * scale).toFixed(3));
    const half = MOCK_CI_HALF_WIDTH[i];
    return {
      week: pt.week,
      cif,
      cifLower: parseFloat(Math.max(0, cif - half).toFixed(3)),
      cifUpper: parseFloat(Math.min(1, cif + half).toFixed(3)),
      population: pt.cif
    };
  });
}

interface TreatmentResistanceCurveProps {
  riskProbability: number; // 0–1 from RF classifier
  isHighRisk: boolean;
}

function TreatmentResistanceCurve({ riskProbability, isHighRisk }: TreatmentResistanceCurveProps) {
  const data = buildMockDeepHitCurve(riskProbability);
  const patientColor = isHighRisk ? "#be123c" : "#16a34a";
  const ciBandColor = isHighRisk ? "#fca5a5" : "#86efac";

  const TRIAL_LABELS: Record<number, string> = {
    8: "Trial 1",
    16: "Trial 2",
    24: "Trial 3",
    36: "TRD"
  };

  return (
    <div className="flex flex-col h-full">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Probability of Non-Response Over Time (Time-to-Event)</p>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 8, right: 20, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />

          <XAxis
            dataKey="week"
            type="number"
            domain={[0, 36]}
            ticks={[0, 8, 16, 24, 36]}
            tickFormatter={(v: number) => (v === 0 ? "Baseline" : `Wk ${v}`)}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
          />

          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1.0]}
            tickFormatter={(v: number) => v.toFixed(2)}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
            width={38}
          />

          {/* Custom tooltip — shows patient CIF + CI range + population */}
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const pt = payload.find((p) => p.dataKey === "cif");
              const lower = payload.find((p) => p.dataKey === "cifLower");
              const upper = payload.find((p) => p.dataKey === "cifUpper");
              const pop = payload.find((p) => p.dataKey === "population");
              const weekNum = label as number;
              return (
                <div
                  style={{
                    background: "#fff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    padding: "8px 12px",
                    fontSize: 12
                  }}
                >
                  <p style={{ fontWeight: 600, color: "#374151", marginBottom: 4 }}>
                    {`Week ${weekNum}${TRIAL_LABELS[weekNum] ? ` — ${TRIAL_LABELS[weekNum]}` : ""}`}
                  </p>
                  {pt && (
                    <p style={{ color: patientColor }}>
                      {`This patient: ${((pt.value as number) * 100).toFixed(1)}%`}
                      {lower && upper && ` (${((lower.value as number) * 100).toFixed(0)}–${((upper.value as number) * 100).toFixed(0)}%)`}
                    </p>
                  )}
                  {pop && <p style={{ color: "#94a3b8" }}>{`Population median: ${((pop.value as number) * 100).toFixed(1)}%`}</p>}
                </div>
              );
            }}
          />

          {/* Trial milestone vertical reference lines */}
          {[8, 16, 24].map((w) => (
            <ReferenceLine
              key={w}
              x={w}
              stroke="#e2e8f0"
              strokeWidth={1}
              label={{
                value: TRIAL_LABELS[w],
                position: "top",
                fontSize: 9,
                fill: "#cbd5e1"
              }}
            />
          ))}
          <ReferenceLine
            x={36}
            stroke="#fca5a5"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{ value: "TRD", position: "top", fontSize: 9, fill: "#f87171" }}
          />

          {/* CI band — Area to cifUpper, then white mask up to cifLower */}
          {/* Note: chart card must have white background for mask to work */}
          <Area type="monotone" dataKey="cifUpper" stroke="none" fill={ciBandColor} fillOpacity={0.25} legendType="none" isAnimationActive={false} />
          <Area type="monotone" dataKey="cifLower" stroke="none" fill="#ffffff" fillOpacity={1} legendType="none" isAnimationActive={false} />

          {/* Population KM reference — will be UKBB reference after DeepHit integration */}
          <Line type="monotone" dataKey="population" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 3" dot={false} legendType="none" />

          {/* Patient CIF — primary patient curve */}
          <Line
            type="monotone"
            dataKey="cif"
            stroke={patientColor}
            strokeWidth={2.5}
            dot={{ r: 4, fill: patientColor, strokeWidth: 0 }}
            activeDot={{ r: 6, stroke: patientColor, strokeWidth: 2, fill: "#fff" }}
            legendType="none"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Manual legend */}
      <div className="flex items-center gap-5 mt-2 px-1">
        <div className="flex items-center gap-1.5">
          <svg width="16" height="8">
            <line x1="0" y1="4" x2="16" y2="4" stroke={patientColor} strokeWidth="2.5" />
          </svg>
          <span className="text-[10px] text-slate-400">This patient</span>
        </div>
        <div className="flex items-center gap-1.5">
          <svg width="16" height="8">
            <line x1="0" y1="4" x2="16" y2="4" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4 3" />
          </svg>
          <span className="text-[10px] text-slate-400">Population median</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2.5 rounded-sm" style={{ backgroundColor: ciBandColor, opacity: 0.4 }} />
          <span className="text-[10px] text-slate-400">95% CI</span>
        </div>
      </div>

      <p className="text-[9px] text-slate-300 mt-1.5 text-right leading-tight">
        Mock: STAR*D shape (Rush et al., 2006) scaled to classifier output · DeepHit / UKBB replaces on integration
      </p>
    </div>
  );
}

function RiskGaugeBar({ probability }: { probability: number }) {
  const pct = Math.min(100, Math.max(0, Math.round(probability * 100)));
  return (
    <div className="mt-4">
      <div
        className="relative h-2 w-full rounded-full overflow-visible"
        style={{ background: "linear-gradient(to right, #22c55e 0%, #eab308 50%, #ef4444 100%)" }}
      >
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-white border-2 border-slate-800 shadow-md"
          style={{ left: `calc(${pct}% - 7px)` }}
        />
      </div>
      <div className="flex justify-between mt-1.5">
        <span className="text-xs text-slate-400">Low risk</span>
        <span className="text-xs text-slate-400">High risk</span>
      </div>
    </div>
  );
}

function RiskBadge({ isHighRisk }: { isHighRisk: boolean }) {
  return (
    <span
      className={
        isHighRisk
          ? "bg-red-50 text-red-700 border border-red-200 text-xs font-semibold px-2.5 py-0.5 rounded-full"
          : "bg-green-50 text-green-700 border border-green-200 text-xs font-semibold px-2.5 py-0.5 rounded-full"
      }
    >
      {isHighRisk ? "High Risk" : "Lower Risk"}
    </span>
  );
}

function TsneChart({ tsne, selected }: { tsne: TsneResponse; selected: { x: number; y: number } }) {
  return (
    <Suspense
      fallback={
        <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
          Loading chart…
        </div>
      }
    >
      <Plot
        data={[
          {
            x: tsne.points.map((p) => p.x),
            y: tsne.points.map((p) => p.y),
            mode: "markers",
            type: "scattergl",
            marker: {
              color: tsne.points.map((p) => (p.class_value === 1 ? "#b45309" : "#6b9e6b")),
              size: 5,
              opacity: 0.75
            },
            showlegend: false
          },
          {
            x: [selected.x],
            y: [selected.y],
            mode: "markers",
            type: "scattergl",
            marker: { color: "#111827", size: 12, symbol: "star" },
            name: "Patient",
            showlegend: false
          }
        ]}
        layout={{
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          margin: { t: 8, b: 30, l: 30, r: 8 },
          xaxis: { showgrid: false, zeroline: false, showticklabels: false },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false },
          autosize: true
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "280px" }}
        useResizeHandler
      />
    </Suspense>
  );
}

function ShapChart({ shapValues }: { shapValues: PredictionResponse["shap_values"] }) {
  const data = shapValues
    .slice(0, 10)
    .map((e) => ({ name: e.feature_label, value: +e.shap_value.toFixed(3) }))
    .reverse();

  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }} barCategoryGap="25%">
        <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 11, fill: "#374151" }} axisLine={false} tickLine={false} />
        <Tooltip
          formatter={(v) => {
            const raw = Array.isArray(v) ? v[0] : v;
            const numeric = Number(raw ?? 0);
            return [numeric > 0 ? `+${numeric}` : numeric, "SHAP value"];
          }}
          contentStyle={{
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            fontSize: 12
          }}
        />
        <ReferenceLine x={0} stroke="#e2e8f0" strokeWidth={1.5} />
        <Bar dataKey="value" barSize={18} radius={[0, 3, 3, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? "#be123c" : "#6b9e6b"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ResultsShell(props: {
  role: Exclude<Route, "intake">;
  onNavigate: (route: Route) => void;
  onLogout?: () => void;
  children: ReactNode;
}) {
  const { role, onNavigate, onLogout, children } = props;

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <aside className="w-56 flex-none flex flex-col bg-white border-r border-slate-100 py-6 px-3 gap-1">
        <div className="px-2 mb-6">
          <img src={PSYCH_STRATA_LOGO_URL} alt="TheraPath" className="h-8 w-auto object-contain" />
        </div>
        <button
          type="button"
          onClick={() => onNavigate("clinician")}
          className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
            role === "clinician" ? "bg-blue-50 text-blue-700 font-medium" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          <Stethoscope size={15} />
          Medical View
        </button>
        <button
          type="button"
          onClick={() => onNavigate("scientist")}
          className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
            role === "scientist" ? "bg-blue-50 text-blue-700 font-medium" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          <FlaskConical size={15} />
          Scientific View
        </button>
        <button
          type="button"
          onClick={() => onNavigate("patient")}
          className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
            role === "patient" ? "bg-blue-50 text-blue-700 font-medium" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          <User size={15} />
          Patient View
        </button>
        <div className="mt-auto">
          <div className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-3 text-[10px] leading-relaxed text-blue-900 mb-3">
            <p className="font-semibold uppercase tracking-wide text-blue-700 mb-1">Preview Mode</p>
            This interface is for demonstration only. It does not provide medical advice or diagnoses and is not a certified medical
            device. Data used are synthetic, do not correspond to actual patient outcomes, and should not be interpreted as such.
          </div>
          <button
            type="button"
            onClick={() => onNavigate("intake")}
            className="flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-50 hover:text-slate-900 transition-colors"
          >
            <PlusCircle size={15} />
            New assessment
          </button>
          {onLogout && (
            <button
              type="button"
              onClick={onLogout}
              className="flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm text-slate-500 hover:bg-slate-50 hover:text-slate-900 transition-colors mt-1"
            >
              <LogOut size={15} />
              Sign out
            </button>
          )}
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto p-6">{children}</main>
    </div>
  );
}

function App() {
  const authEnabled = getIsAuthEnabled();
  const configuredPassword = getConfiguredPassword();
  const [route, setRoute] = useState<Route>(getInitialRoute());
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => !authEnabled || hasAuthenticatedSession());
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [clinicianSimValues, setClinicianSimValues] = useState({
    sertraline_mg: 0,
    lithium_mg: 0,
    quetiapine_mg: 0,
    adherence_pct: 75
  });
  const [clinicianImpactPct, setClinicianImpactPct] = useState(0);
  const [isClinicianImpactSubmitting, setIsClinicianImpactSubmitting] = useState(false);
  const [clinicianImpactError, setClinicianImpactError] = useState<string | null>(null);
  const [isClinicianApplySubmitting, setIsClinicianApplySubmitting] = useState(false);
  const [clinicianApplyError, setClinicianApplyError] = useState<string | null>(null);

  useEffect(() => {
    const onPopState = () => setRoute(getInitialRoute());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (state.status !== "ready" || state.prediction === null) {
      return;
    }
    setClinicianSimValues({
      sertraline_mg: state.featureValues.sertraline_mg ?? 0,
      lithium_mg: state.featureValues.lithium_mg ?? 0,
      quetiapine_mg: state.featureValues.quetiapine_mg ?? 0,
      adherence_pct: state.featureValues.adherence_pct ?? 75
    });
    setClinicianImpactPct(0);
    setClinicianImpactError(null);
    setClinicianApplyError(null);
  }, [state.status, state.status === "ready" ? state.prediction : null]);

  useEffect(() => {
    if (authEnabled && !isAuthenticated) {
      return;
    }

    let isMounted = true;
    Promise.all([fetchFeatures(), fetchTsne()])
      .then(([featuresPayload, tsnePayload]) => {
        if (!isMounted) {
          return;
        }
        setState({
          status: "ready",
          features: featuresPayload.features,
          confidenceBounds: {
            min: featuresPayload.confidence_level.min,
            max: featuresPayload.confidence_level.max,
            step: featuresPayload.confidence_level.step
          },
          confidenceLevel: featuresPayload.confidence_level.default,
          featureValues: { ...featuresPayload.defaults },
          prediction: null,
          explanation: "",
          tsne: tsnePayload,
          isSubmitting: false,
          isSummaryRefreshing: false,
          error: null
        });
      })
      .catch((error: unknown) => {
        if (!isMounted) {
          return;
        }
        setState({
          status: "error",
          message: error instanceof Error ? error.message : "Unable to reach the backend API."
        });
      });

    return () => {
      isMounted = false;
    };
  }, [authEnabled, isAuthenticated]);

  const navigate = (nextRoute: Route) => {
    const nextPath = ROUTE_TO_PATH[nextRoute];
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setRoute(nextRoute);
  };

  const signOut = () => {
    sessionStorage.removeItem(AUTH_SESSION_KEY);
    setIsAuthenticated(false);
    setLoginPassword("");
    setLoginError(null);
    setState({ status: "loading" });
    if (window.location.pathname !== ROUTE_TO_PATH.intake) {
      window.history.pushState({}, "", ROUTE_TO_PATH.intake);
    }
    setRoute("intake");
  };

  if (authEnabled && !isAuthenticated) {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
        <section className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8 w-full max-w-md">
          <div className="flex items-center gap-3 mb-6">
            <img src={PSYCH_STRATA_LOGO_URL} alt="Psych-STRATA" className="h-8 w-auto" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900 mb-1">Dashboard login</h1>
          <p className="text-sm text-slate-500 mb-6">Enter the shared password to access the dashboard.</p>
          <form
            className="flex flex-col gap-3"
            onSubmit={(event) => {
              event.preventDefault();
              if (loginPassword === configuredPassword) {
                sessionStorage.setItem(AUTH_SESSION_KEY, "true");
                setIsAuthenticated(true);
                setLoginError(null);
                setLoginPassword("");
                return;
              }
              setLoginError("Incorrect password.");
            }}
          >
            <label className="flex flex-col gap-1" htmlFor="dashboard-password">
              <span className="text-xs font-medium text-slate-600">Password</span>
              <input
                id="dashboard-password"
                type="password"
                autoComplete="current-password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </label>
            {loginError && <p className="text-sm text-red-600">{loginError}</p>}
            <button
              type="submit"
              className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors cursor-pointer"
            >
              Sign in
            </button>
          </form>
        </section>
      </main>
    );
  }

  const runPrediction = async (targetRoute: Exclude<Route, "intake">) => {
    if (state.status !== "ready") {
      return;
    }

    setState((prev) => {
      if (prev.status !== "ready") return prev;
      return { ...prev, isSubmitting: true, isSummaryRefreshing: false, error: null };
    });

    try {
      const payload = {
        features: state.featureValues,
        confidence_level: state.confidenceLevel
      };
      const prediction = await fetchPredict(payload);

      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return {
          ...prev,
          prediction,
          explanation: "",
          featureValues: prediction.features,
          isSubmitting: false,
          isSummaryRefreshing: false
        };
      });
      navigate(targetRoute);
    } catch (error: unknown) {
      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return {
          ...prev,
          isSubmitting: false,
          isSummaryRefreshing: false,
          error: error instanceof Error ? error.message : "Prediction request failed."
        };
      });
    }
  };

  const refreshExplanation = async () => {
    if (state.status !== "ready" || state.prediction === null) {
      return;
    }

    setState((prev) => {
      if (prev.status !== "ready") return prev;
      return { ...prev, isSummaryRefreshing: true, error: null };
    });

    try {
      const explanationPayload = await fetchExplain({
        features: state.featureValues,
        confidence_level: state.confidenceLevel
      });
      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return {
          ...prev,
          explanation: explanationPayload.explanation,
          isSummaryRefreshing: false
        };

      });
    } catch (error: unknown) {
      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return {
          ...prev,
          isSummaryRefreshing: false,
          error: error instanceof Error ? error.message : "Explanation refresh failed."
        };
      });
    }
  };

  const runClinicianImpactPrediction = async () => {
    if (state.status !== "ready" || state.prediction === null) {
      return;
    }

    setIsClinicianImpactSubmitting(true);
    setClinicianImpactError(null);
    try {
      const response = await fetchPredict({
        features: {
          ...state.featureValues,
          sertraline_mg: clinicianSimValues.sertraline_mg,
          lithium_mg: clinicianSimValues.lithium_mg,
          quetiapine_mg: clinicianSimValues.quetiapine_mg,
          adherence_pct: clinicianSimValues.adherence_pct
        },
        confidence_level: state.confidenceLevel
      });

      const baselineRisk = state.prediction.prediction.probability_resistance;
      const simulatedRisk = response.prediction.probability_resistance;
      setClinicianImpactPct((baselineRisk - simulatedRisk) * 100);
    } catch (error: unknown) {
      setClinicianImpactError(error instanceof Error ? error.message : "Simulation request failed.");
    } finally {
      setIsClinicianImpactSubmitting(false);
    }
  };

  const applyClinicianSimulationToPage = async () => {
    if (state.status !== "ready") {
      return;
    }

    setIsClinicianApplySubmitting(true);
    setClinicianApplyError(null);
    try {
      const response = await fetchPredict({
        features: {
          ...state.featureValues,
          sertraline_mg: clinicianSimValues.sertraline_mg,
          lithium_mg: clinicianSimValues.lithium_mg,
          quetiapine_mg: clinicianSimValues.quetiapine_mg,
          adherence_pct: clinicianSimValues.adherence_pct
        },
        confidence_level: state.confidenceLevel
      });

      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return {
          ...prev,
          prediction: response,
          featureValues: response.features,
          explanation: "",
          error: null
        };
      });
    } catch (error: unknown) {
      setClinicianApplyError(error instanceof Error ? error.message : "Apply request failed.");
    } finally {
      setIsClinicianApplySubmitting(false);
    }
  };

  if (state.status === "loading") {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center">
        <p className="text-sm text-slate-500">Loading clinical model configuration…</p>
      </main>
    );
  }

  if (state.status === "error") {
    return (
      <main className="min-h-screen bg-slate-50 flex flex-col items-center justify-center gap-2">
        <h1 className="text-lg font-semibold text-slate-900">Backend unavailable</h1>
        <p className="text-sm text-slate-500">{state.message}</p>
      </main>
    );
  }

  const { features, featureValues, confidenceBounds, confidenceLevel, prediction, explanation, tsne } = state;
  const quickAdjustIds = ["sertraline_mg", "lithium_mg", "quetiapine_mg", "adherence_pct"];
  const quickAdjustFeatures = features.filter((feature) => quickAdjustIds.includes(feature.id));

  const strengths = prediction?.top_contributors.negative ?? [];
  const actionItems = prediction?.top_contributors.positive ?? [];

  const setFeatureValue = (featureId: string, value: number) => {
    setState((prev) => {
      if (prev.status !== "ready") return prev;
      return {
        ...prev,
        featureValues: {
          ...prev.featureValues,
          [featureId]: value
        }
      };
    });
  };

  const hasResult = prediction !== null;

  if (route === "intake") {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
        <section className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8 w-full max-w-2xl">
          <div className="flex items-center justify-between gap-3 mb-6">
            <img src={PSYCH_STRATA_LOGO_URL} alt="Psych-STRATA" className="h-8 w-auto" />
            {authEnabled && (
              <button
                type="button"
                onClick={signOut}
                className="bg-white text-slate-700 text-xs font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors"
              >
                Sign out
              </button>
            )}
          </div>
          <h1 className="text-2xl font-bold text-slate-900 mb-1">Patient feature intake</h1>
          <p className="text-sm text-slate-500 mb-6">
            Enter the current patient profile. Values are prefilled from backend defaults and will be reused across patient, clinician, and scientist views.
          </p>
          <form
            className="grid grid-cols-2 gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              void runPrediction("patient");
            }}
          >
            {features.map((feature) => (
              <FeatureField key={feature.id} feature={feature} value={featureValues[feature.id]} onChange={setFeatureValue} />
            ))}
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-slate-600">Conformal confidence level (%)</span>
              <input
                type="number"
                min={confidenceBounds.min}
                max={confidenceBounds.max}
                step={confidenceBounds.step}
                value={confidenceLevel}
                onChange={(event) => {
                  const value = parseNumberValue(event.target.value);
                  setState((prev) => {
                    if (prev.status !== "ready") return prev;
                    return { ...prev, confidenceLevel: value };
                  });
                }}
                className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </label>
            <div className="flex items-center gap-4 mt-6 col-span-2">
              <button
                type="submit"
                disabled={state.isSubmitting}
                className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50 cursor-pointer"
              >
                {state.isSubmitting ? "Calculating..." : "Generate results"}
              </button>
              {state.error && <p className="text-sm text-red-600 mt-2">{state.error}</p>}
            </div>
          </form>
        </section>
      </main>
    );
  }

  if (!hasResult || prediction === null) {
    return (
      <main className="min-h-screen bg-slate-50 flex flex-col items-center justify-center gap-3">
        <p className="text-sm text-slate-500">No results yet. Start with the patient feature intake.</p>
        <button
          type="button"
          onClick={() => navigate("intake")}
          className="bg-white text-slate-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs"
        >
          Go to intake
        </button>
      </main>
    );
  }

  const selectedTsne = prediction.tsne.selected;
  const riskClass = prediction.prediction.predicted_class;
  const riskProbability = prediction.prediction.probability_resistance;
  const isHighRisk = riskClass === "Resistant";
  const sertralineDose = clinicianSimValues.sertraline_mg;
  const lithiumDose = clinicianSimValues.lithium_mg;
  const quetiapineDose = clinicianSimValues.quetiapine_mg;
  const adherencePct = clinicianSimValues.adherence_pct;
  const currentSertralineDose = featureValues.sertraline_mg ?? 0;
  const currentLithiumDose = featureValues.lithium_mg ?? 0;
  const currentQuetiapineDose = featureValues.quetiapine_mg ?? 0;
  const currentAdherencePct = featureValues.adherence_pct ?? 0;

  const adherenceCategoryValue = adherencePct < 60 ? 50 : adherencePct < 85 ? 75 : 90;
  const adherenceCategoryLabel = currentAdherencePct < 60 ? "Low" : currentAdherencePct < 85 ? "Moderate" : "High";
  const medicationLoad = currentSertralineDose + currentLithiumDose + currentQuetiapineDose;
  const phqScore = featureValues.phq9 ?? 0;
  const responseProbability = 1 - riskProbability;
  const responseBadge =
    responseProbability >= 0.7
      ? { label: "High Response", classes: "bg-emerald-50 text-emerald-700 border border-emerald-200" }
      : responseProbability >= 0.45
        ? { label: "Mixed Response", classes: "bg-amber-50 text-amber-700 border border-amber-200" }
        : { label: "Low Response", classes: "bg-sky-50 text-sky-700 border border-sky-200" };

  const riskCard = (
    <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">Treatment Resistance Risk</p>
      <div className="flex items-baseline gap-3">
        <span className="text-6xl font-bold text-slate-900 tracking-tight">{pct(riskProbability)}</span>
        <RiskBadge isHighRisk={isHighRisk} />
      </div>
      <p className="text-xs text-slate-400 mt-1">
        Conformal subset: {isHighRisk ? "Resistant" : "Responsive"} at 95% confidence.
      </p>
      <RiskGaugeBar probability={riskProbability} />
    </article>
  );

  const scientistHeader = (
    <header className="grid grid-cols-[5fr_8fr] gap-4 mb-4">
      {riskCard}
      <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
        <TreatmentResistanceCurve riskProbability={riskProbability} isHighRisk={isHighRisk} />
      </article>
    </header>
  );

  const clinicianHeader = (
    <header className="grid grid-cols-[5fr_8fr] gap-4 mb-4">
      {riskCard}
      <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-900">Probability of Non-Response Over Time (Time-to-Event)</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void runClinicianImpactPrediction()}
              disabled={isClinicianImpactSubmitting || isClinicianApplySubmitting}
              className="bg-slate-900 text-white text-[11px] font-medium px-3 py-1.5 rounded-md hover:bg-slate-700 transition-colors disabled:opacity-50 cursor-pointer uppercase tracking-wide"
            >
              {isClinicianImpactSubmitting ? "Calculating..." : "Calculate"}
            </button>
            <button
              type="button"
              title="Applies these simulated settings to the full clinician view and refreshes risk, SHAP, and trajectory charts."
              onClick={() => void applyClinicianSimulationToPage()}
              disabled={isClinicianApplySubmitting || isClinicianImpactSubmitting}
              className="bg-white text-slate-700 text-[11px] font-medium px-3 py-1.5 rounded-md border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 cursor-pointer uppercase tracking-wide"
            >
              {isClinicianApplySubmitting ? "Applying..." : "Apply to Page"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-500">Sertraline Dosage (mg/day)</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={200}
                step={5}
                value={sertralineDose}
                onChange={(event) =>
                  setClinicianSimValues((prev) => ({ ...prev, sertraline_mg: parseNumberValue(event.target.value) }))
                }
                className="w-full accent-slate-900"
              />
              <output className="text-xs text-slate-700 min-w-8 text-right">{sertralineDose}</output>
            </div>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-500">Therapeutic Adherence</span>
            <select
              value={String(adherenceCategoryValue)}
              onChange={(event) =>
                setClinicianSimValues((prev) => ({ ...prev, adherence_pct: parseNumberValue(event.target.value) }))
              }
              className="mt-0.5 w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="50">Low</option>
              <option value="75">Moderate</option>
              <option value="90">High</option>
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[10px] text-slate-500">Lithium Dosage (mg/day)</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={1200}
                step={100}
                value={lithiumDose}
                onChange={(event) =>
                  setClinicianSimValues((prev) => ({ ...prev, lithium_mg: parseNumberValue(event.target.value) }))
                }
                className="w-full accent-slate-900"
              />
              <output className="text-xs text-slate-700 min-w-8 text-right">{lithiumDose}</output>
            </div>
          </label>

          <div className="rounded-md bg-blue-100/70 border border-blue-100 px-3 py-2 flex flex-col justify-center">
            <p className="text-[10px] uppercase tracking-wide text-slate-500">Predicted Impact</p>
            <p className="text-xs text-slate-700">
              {`${clinicianImpactPct > 0 ? "-" : clinicianImpactPct < 0 ? "+" : ""}${Math.abs(clinicianImpactPct).toFixed(1)}% Resistance Risk Change`}
            </p>
            {clinicianImpactError && <p className="text-[10px] text-red-600 mt-1">{clinicianImpactError}</p>}
            {clinicianApplyError && <p className="text-[10px] text-red-600 mt-1">{clinicianApplyError}</p>}
          </div>

          <label className="flex flex-col gap-1 col-span-1">
            <span className="text-[10px] text-slate-500">Quetiapine Augmentation (mg/day)</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={300}
                step={25}
                value={quetiapineDose}
                onChange={(event) =>
                  setClinicianSimValues((prev) => ({ ...prev, quetiapine_mg: parseNumberValue(event.target.value) }))
                }
                className="w-full accent-slate-900"
              />
              <output className="text-xs text-slate-700 min-w-8 text-right">{quetiapineDose}</output>
            </div>
          </label>

          <div className="col-span-1" />
        </div>
      </article>
    </header>
  );

  const simulator = (
    <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-900">Your Treatment Path Simulator</h2>
        <button
          type="button"
          onClick={() => void runPrediction(route)}
          disabled={state.isSubmitting}
          className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50 cursor-pointer"
        >
          {state.isSubmitting ? "Calculating..." : "Calculate"}
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {quickAdjustFeatures.map((feature) => (
          <FeatureField key={feature.id} feature={feature} value={featureValues[feature.id]} onChange={setFeatureValue} compact />
        ))}
      </div>
    </article>
  );

  const scientistControls = (
    <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-900">Exploratory Parameter Controls</h2>
        <button
          type="button"
          onClick={() => void runPrediction("scientist")}
          disabled={state.isSubmitting}
          className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50 cursor-pointer"
        >
          {state.isSubmitting ? "Calculating..." : "Calculate"}
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3 mb-4">
        {features.map((feature) => (
          <FeatureField key={`scientist-${feature.id}`} feature={feature} value={featureValues[feature.id]} onChange={setFeatureValue} compact />
        ))}
      </div>
      <label className="flex flex-col gap-1">
        <span className="text-xs font-medium text-slate-600">Conformal confidence level (%)</span>
        <input
          type="number"
          min={confidenceBounds.min}
          max={confidenceBounds.max}
          step={confidenceBounds.step}
          value={confidenceLevel}
          onChange={(event) => {
            const value = parseNumberValue(event.target.value);
            setState((prev) => {
              if (prev.status !== "ready") return prev;
              return { ...prev, confidenceLevel: value };
            });
          }}
          className="mt-1 w-full max-w-xs rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </label>
    </article>
  );

  const explanationCard = (
    <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
      {explanation ? (
        <>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-900">AI Clinical Insights & Literature Review</h2>
            <button
              type="button"
              onClick={() => void refreshExplanation()}
              disabled={state.isSummaryRefreshing}
              className="bg-white text-slate-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs"
            >
              {state.isSummaryRefreshing ? "Updating..." : "Refresh summary"}
            </button>
          </div>
          {state.isSummaryRefreshing ? (
            <p className="text-sm text-slate-500">Refreshing explanation for the current profile...</p>
          ) : (
            renderSummaryMarkdown(explanation)
          )}
        </>
      ) : (
        <div className="flex flex-col gap-3">
          <h2 className="text-sm font-semibold text-slate-900">AI Clinical Insights & Literature Review</h2>
          <button
            type="button"
            onClick={() => void refreshExplanation()}
            disabled={state.isSummaryRefreshing}
            className="bg-white text-slate-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs w-fit"
          >
            {state.isSummaryRefreshing ? "Generating..." : "Generate summary"}
          </button>
          <p className="text-sm text-slate-500">Generate a plain-language summary for the current profile.</p>
        </div>
      )}
      {state.error && <p className="text-sm text-red-600 mt-2">{state.error}</p>}
    </article>
  );

  const shapCard = (
    <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
      <h2 className="text-sm font-semibold text-slate-900 mb-4">Risk Factor Analysis (SHAP)</h2>
      <ShapChart shapValues={prediction.shap_values} />
    </article>
  );

  if (route === "patient") {
    return (
      <ResultsShell role="patient" onNavigate={navigate} onLogout={authEnabled ? signOut : undefined}>
        <div className="mb-6">
          <h1 className="text-4xl font-bold text-slate-900 mb-3">Welcome, John.</h1>
          <p className="text-slate-500 text-sm max-w-2xl leading-relaxed">
            We've reviewed your recent clinical assessments. The goal is to help you navigate your treatment journey and find the most
            effective approach for long-term resilience.
          </p>
        </div>

        <section className="grid grid-cols-2 gap-4 mt-4">
          <article className="bg-green-50/40 rounded-xl border border-green-100 shadow-sm p-6">
            <h2 className="text-sm font-semibold text-green-900 mb-3">Your Strengths</h2>
            <ul className="list-none pl-0">
              {strengths.slice(0, 3).map((item) => (
                <li key={item.feature_id} className="flex items-start gap-2 text-sm text-slate-700 py-1">
                  <CheckCircle2 size={15} className="text-green-600 mt-0.5 flex-none" />
                  <span>
                    <strong>{item.feature_label}</strong> ({item.selected_value})
                  </span>
                </li>
              ))}
            </ul>
          </article>

          <article className="bg-red-50/40 rounded-xl border border-red-100 shadow-sm p-6">
            <h2 className="text-sm font-semibold text-red-900 mb-3">Action Items</h2>
            <ul className="list-none pl-0">
              {actionItems.slice(0, 3).map((item) => (
                <li key={item.feature_id} className="flex items-start gap-2 text-sm text-slate-700 py-1">
                  <AlertCircle size={15} className="text-red-500 mt-0.5 flex-none" />
                  <span>
                    <strong>{item.feature_label}</strong> ({item.selected_value})
                  </span>
                </li>
              ))}
            </ul>
          </article>
        </section>

        {/* Clinical summary hidden — not in target patient-facing design
        <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6 mt-4">
          {explanation ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-900">Your clinical summary</h2>
                <button
                  type="button"
                  onClick={() => void refreshExplanation()}
                  disabled={state.isSummaryRefreshing}
                  className="bg-white text-slate-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs"
                >
                  {state.isSummaryRefreshing ? "Updating..." : "Refresh summary"}
                </button>
              </div>
              {state.isSummaryRefreshing ? (
                <p className="text-sm text-slate-500">Refreshing explanation for the current profile...</p>
              ) : (
                renderSummaryMarkdown(explanation)
              )}
            </>
          ) : (
            <div className="flex flex-col gap-3">
              <h2 className="text-sm font-semibold text-slate-900">Your clinical summary</h2>
              <button
                type="button"
                onClick={() => void refreshExplanation()}
                disabled={state.isSummaryRefreshing}
                className="bg-white text-slate-700 text-sm font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-50 text-xs w-fit"
              >
                {state.isSummaryRefreshing ? "Generating..." : "Generate summary"}
              </button>
              <p className="text-sm text-slate-500">{supportiveSummary(riskProbability)}</p>
            </div>
          )}
          {state.error && <p className="text-sm text-red-600 mt-2">{state.error}</p>}
        </article>
        */}

        <section className="grid grid-cols-2 gap-4 mt-4">
          {simulator}
          <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6 flex flex-col items-center justify-center text-center">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">ESTIMATED TREATMENT RESPONSE</p>
            <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-full ${responseBadge.classes}`}>{responseBadge.label}</span>
            <p className="text-6xl font-bold text-slate-900 tracking-tight mt-2">{pct(responseProbability)}</p>
            <p className="text-sm text-slate-500 mt-3">Based on current patient profile</p>
          </article>
        </section>
      </ResultsShell>
    );
  }

  if (route === "clinician") {
    return (
      <ResultsShell role="clinician" onNavigate={navigate} onLogout={authEnabled ? signOut : undefined}>
        {clinicianHeader}
        {explanationCard}
        <section className="grid grid-cols-2 gap-4 mt-4">
          {shapCard}
          <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
            <TreatmentResistanceCurve riskProbability={riskProbability} isHighRisk={isHighRisk} />
          </article>
        </section>
        <section className="grid grid-cols-4 gap-3 mt-4">
          <article className="bg-slate-100 rounded-lg border border-slate-200 px-3 py-2.5">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">PHQ-9 Score</p>
            <p className="text-xl font-semibold text-slate-900 leading-tight mt-0.5">{phqScore}</p>
            <p className="text-[10px] text-slate-500 mt-0.5">{phqScore >= 20 ? "Severe symptoms" : phqScore >= 15 ? "Moderately severe" : "Mild-to-moderate"}</p>
          </article>
          <article className="bg-slate-100 rounded-lg border border-slate-200 px-3 py-2.5">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">Medication Adherence</p>
            <p className="text-xl font-semibold text-slate-900 leading-tight mt-0.5">{adherenceCategoryLabel}</p>
            <p className="text-[10px] text-slate-500 mt-0.5">Potential for improvement</p>
          </article>
          <article className="bg-slate-100 rounded-lg border border-slate-200 px-3 py-2.5">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">Medication Load</p>
            <p className="text-xl font-semibold text-slate-900 leading-tight mt-0.5">{medicationLoad >= 900 ? "High" : medicationLoad >= 400 ? "Medium" : "Low"} <span className="text-sm font-medium text-slate-600">({medicationLoad} mg)</span></p>
            <p className="text-[10px] text-slate-500 mt-0.5">{medicationLoad >= 900 ? "Complex polypharmacy" : "Current regimen intensity"}</p>
          </article>
          <article className="bg-slate-100 rounded-lg border border-slate-200 px-3 py-2.5">
            <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">Lithium Level</p>
            <p className="text-xl font-semibold text-slate-900 leading-tight mt-0.5">{currentLithiumDose} <span className="text-sm font-medium text-slate-600">mg/day</span></p>
            <p className="text-[10px] text-slate-500 mt-0.5">{currentLithiumDose >= 600 ? "Within augmentation range" : "Below therapeutic range"}</p>
          </article>
        </section>
      </ResultsShell>
    );
  }

  return (
    <ResultsShell role="scientist" onNavigate={navigate} onLogout={authEnabled ? signOut : undefined}>
      {scientistHeader}
      {scientistControls}
      {explanationCard}
      <section className="grid grid-cols-2 gap-4 mt-4">
        {shapCard}
        <article className="bg-white rounded-xl border border-slate-100 shadow-sm p-6">
          <h2 className="text-sm font-semibold text-slate-900 mb-4">t-SNE Population Map</h2>
          {tsne ? <TsneChart tsne={tsne} selected={selectedTsne} /> : <p className="text-sm text-slate-500">t-SNE data unavailable.</p>}
        </article>
      </section>
    </ResultsShell>
  );
}

export default App;
