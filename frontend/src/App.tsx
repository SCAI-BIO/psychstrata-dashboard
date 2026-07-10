import { Fragment, type ReactNode, useEffect, useState } from "react";

import {
  API_BASE_URL,
  type FeatureSchema,
  type PredictionResponse,
  type TsneResponse,
  fetchExplain,
  fetchFeatures,
  fetchPredict,
  fetchTsne
} from "./api";
import "./App.css";

type Route = "intake" | "patient" | "clinician" | "scientist";

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

function optionLabel(feature: FeatureSchema, value: number): string {
  if (feature.kind !== "categorical") {
    return String(value);
  }
  const option = (feature.options ?? feature.params.options ?? []).find((item) => item.value === value);
  return option?.label ?? String(value);
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
        <ul key={`list-${listKey}`} className="summary-list">
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
        <h4 key={`heading-${index}`} className="summary-heading">
          {headingMatch[1]}
        </h4>
      );
      return;
    }

    blocks.push(
      <p key={`paragraph-${index}`} className="summary-paragraph">
        {renderInlineMarkdown(line)}
      </p>
    );
  });

  flushList();
  return <div className="summary-content">{blocks}</div>;
}

function trendSeries(probability: number): number[] {
  const baseline = Math.max(0.02, probability * 0.12);
  const horizon = Math.min(0.95, probability * 0.98 + 0.02);
  return [0, 1, 2, 3, 4, 5].map((idx) => baseline + (horizon - baseline) * (1 / (1 + Math.exp(-1.2 * (idx - 2)))));
}

function FeatureField(props: {
  feature: FeatureSchema;
  value: number;
  onChange: (featureId: string, value: number) => void;
  compact?: boolean;
}) {
  const { feature, value, onChange, compact = false } = props;
  const inputId = `feature-${feature.id}`;
  const min = feature.min ?? feature.params.min ?? 0;
  const max = feature.max ?? feature.params.max ?? 100;
  const step = feature.step ?? feature.params.step ?? 1;

  return (
    <label className={`feature-field ${compact ? "compact" : ""}`} htmlFor={inputId}>
      <span>{feature.label}</span>
      {feature.kind === "numeric" ? (
        <div className="numeric-input-wrap">
          <input
            id={inputId}
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}
          />
          <output>{value}</output>
        </div>
      ) : (
        <select id={inputId} value={value} onChange={(event) => onChange(feature.id, parseNumberValue(event.target.value))}>
          {(feature.options ?? feature.params.options ?? []).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      )}
    </label>
  );
}

function TimeCurve(props: { values: number[] }) {
  const { values } = props;
  const width = 520;
  const height = 180;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * (width - 30) + 10;
      const y = height - value * 150 - 10;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="curve-svg" aria-label="Risk trend over time">
      <polyline fill="none" stroke="#111827" strokeWidth="3" points={points} />
      <circle cx={width * 0.64} cy={height - values[3] * 150 - 10} r="5" fill="#be123c" />
    </svg>
  );
}

function TsneChart(props: { tsne: TsneResponse; selected: { x: number; y: number } }) {
  const { tsne, selected } = props;
  const xs = tsne.points.map((point) => point.x);
  const ys = tsne.points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const normalizeX = (x: number) => ((x - minX) / (maxX - minX || 1)) * 92 + 4;
  const normalizeY = (y: number) => 96 - ((y - minY) / (maxY - minY || 1)) * 92;

  return (
    <svg viewBox="0 0 100 100" className="tsne-svg" aria-label="t-SNE population map">
      {tsne.points.map((point, index) => (
        <circle
          key={`${point.x}-${point.y}-${index}`}
          cx={normalizeX(point.x)}
          cy={normalizeY(point.y)}
          r="0.8"
          fill={point.class_value === 1 ? "#b45309" : "#6b9e6b"}
          opacity="0.8"
        />
      ))}
      <circle cx={normalizeX(selected.x)} cy={normalizeY(selected.y)} r="2.2" fill="#111827" />
    </svg>
  );
}

function ResultsShell(props: {
  role: Exclude<Route, "intake">;
  onNavigate: (route: Route) => void;
  children: ReactNode;
}) {
  const { role, onNavigate, children } = props;

  return (
    <div className="results-layout">
      <aside className="sidebar">
        <div className="brand">TheraPath</div>
        <button type="button" className={`nav-link ${role === "clinician" ? "active" : ""}`} onClick={() => onNavigate("clinician")}>
          Medical View
        </button>
        <button type="button" className={`nav-link ${role === "scientist" ? "active" : ""}`} onClick={() => onNavigate("scientist")}>
          Scientific View
        </button>
        <button type="button" className={`nav-link ${role === "patient" ? "active" : ""}`} onClick={() => onNavigate("patient")}>
          Patient View
        </button>
        <button type="button" className="secondary-nav" onClick={() => onNavigate("intake")}>
          New assessment
        </button>
      </aside>
      <main className="results-main">{children}</main>
    </div>
  );
}

function App() {
  const [route, setRoute] = useState<Route>(getInitialRoute());
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    const onPopState = () => setRoute(getInitialRoute());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
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
  }, []);

  const navigate = (nextRoute: Route) => {
    const nextPath = ROUTE_TO_PATH[nextRoute];
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setRoute(nextRoute);
  };

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

  if (state.status === "loading") {
    return (
      <main className="center-stage">
        <p>Loading clinical model configuration...</p>
      </main>
    );
  }

  if (state.status === "error") {
    return (
      <main className="center-stage">
        <h1>Backend unavailable</h1>
        <p>{state.message}</p>
        <p className="meta">Configured API base URL: {API_BASE_URL}</p>
      </main>
    );
  }

  const { features, featureValues, confidenceBounds, confidenceLevel, prediction, explanation, tsne } = state;
  const trend = trendSeries(prediction?.prediction.probability_resistance ?? 0.5);
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
      <main className="intake-layout">
        <section className="intake-card">
          <p className="eyebrow">TheraPath Assessment</p>
          <h1>Patient feature intake</h1>
          <p className="lede">
            Enter the current patient profile. Values are prefilled from backend defaults and will be reused across patient, clinician, and scientist views.
          </p>
          <form
            className="feature-grid"
            onSubmit={(event) => {
              event.preventDefault();
              void runPrediction("patient");
            }}
          >
            {features.map((feature) => (
              <FeatureField
                key={feature.id}
                feature={feature}
                value={featureValues[feature.id]}
                onChange={setFeatureValue}
              />
            ))}
            <label className="feature-field">
              <span>Conformal confidence level (%)</span>
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
              />
            </label>
            <div className="actions-row">
              <button type="submit" disabled={state.isSubmitting}>
                {state.isSubmitting ? "Calculating..." : "Generate results"}
              </button>
              {state.error && <p className="error-text">{state.error}</p>}
            </div>
          </form>
        </section>
      </main>
    );
  }

  if (!hasResult || prediction === null) {
    return (
      <main className="center-stage">
        <p>No results yet. Start with the patient feature intake.</p>
        <button type="button" onClick={() => navigate("intake")}>
          Go to intake
        </button>
      </main>
    );
  }

  const selectedTsne = prediction.tsne.selected;
  const riskLabel = prediction.prediction.conformal_prediction.label;
  const riskClass = prediction.prediction.predicted_class;
  const riskProbability = prediction.prediction.probability_resistance;

  const sharedHeader = (
    <header className="top-cards">
      <article className="card risk-card">
        <div className="card-title-row">
          <h2>Treatment Resistance Risk</h2>
          <span className={`risk-pill ${riskClass === "Resistant" ? "high" : "low"}`}>{riskClass === "Resistant" ? "High Risk" : "Lower Risk"}</span>
        </div>
        <p className="risk-value">{pct(riskProbability)}</p>
        <p className="muted">
          Conformal output: <strong>{riskLabel}</strong> at {prediction.prediction.conformal_prediction.confidence_level}% confidence.
        </p>
      </article>
      <article className="card">
        <h2>Probability of Non-Response Over Time</h2>
        <TimeCurve values={trend} />
      </article>
    </header>
  );

  const simulator = (
    <article className="card">
      <div className="card-title-row">
        <h2>Your Treatment Path Simulator</h2>
        <button type="button" onClick={() => void runPrediction(route)} disabled={state.isSubmitting}>
          {state.isSubmitting ? "Calculating..." : "Calculate"}
        </button>
      </div>
      <div className="simulator-grid">
        {quickAdjustFeatures.map((feature) => (
          <FeatureField
            key={feature.id}
            feature={feature}
            value={featureValues[feature.id]}
            onChange={setFeatureValue}
            compact
          />
        ))}
      </div>
      <label className="feature-field compact">
        <span>Therapeutic confidence interval (%)</span>
        <input
          type="range"
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
        />
        <output>{confidenceLevel}</output>
      </label>
    </article>
  );

  const explanationCard = (
    <article className="card">
      {explanation ? (
        <>
          <div className="card-title-row">
            <h2>AI Clinical Insights & Literature Review</h2>
            <button type="button" onClick={() => void refreshExplanation()} disabled={state.isSummaryRefreshing}>
              {state.isSummaryRefreshing ? "Updating..." : "Refresh summary"}
            </button>
          </div>
          {state.isSummaryRefreshing ? (
            <p className="muted">Refreshing explanation for the current profile...</p>
          ) : (
            renderSummaryMarkdown(explanation)
          )}
        </>
      ) : (
        <div className="summary-empty-state">
          <h2>AI Clinical Insights & Literature Review</h2>
          <button type="button" onClick={() => void refreshExplanation()} disabled={state.isSummaryRefreshing}>
            {state.isSummaryRefreshing ? "Generating..." : "Generate summary"}
          </button>
          <p className="muted">Generate a plain-language summary for the current profile.</p>
        </div>
      )}
      {state.error && <p className="error-text">{state.error}</p>}
    </article>
  );

  const shapCard = (
    <article className="card">
      <h2>Risk Factor Analysis (SHAP)</h2>
      <ul className="shap-list">
        {prediction.shap_values.slice(0, 10).map((entry) => (
          <li key={entry.feature_id}>
            <span>{entry.feature_label}</span>
            <div className="shap-bar-wrap">
              <div className="zero-line" />
              <div
                className={`shap-bar ${entry.shap_value >= 0 ? "positive" : "negative"}`}
                style={{
                  width: `${Math.min(100, entry.abs_shap_value * 260)}px`
                }}
              />
            </div>
            <strong>{entry.shap_value > 0 ? "+" : ""}{entry.shap_value.toFixed(3)}</strong>
          </li>
        ))}
      </ul>
    </article>
  );

  if (route === "patient") {
    return (
      <ResultsShell role="patient" onNavigate={navigate}>
        <section className="patient-hero card">
          <h1>Welcome.</h1>
          <p>{supportiveSummary(riskProbability)}</p>
        </section>
        <section className="split-grid">
          <article className="card success-card">
            <h2>Your Strengths</h2>
            <ul>
              {strengths.slice(0, 3).map((item) => (
                <li key={item.feature_id}>
                  <strong>{item.feature_label}</strong> ({item.selected_value})
                </li>
              ))}
            </ul>
          </article>
          <article className="card alert-card">
            <h2>Action Items</h2>
            <ul>
              {actionItems.slice(0, 3).map((item) => (
                <li key={item.feature_id}>
                  <strong>{item.feature_label}</strong> ({item.selected_value})
                </li>
              ))}
            </ul>
          </article>
        </section>
        <article className="card">
          {explanation ? (
            <>
              <div className="card-title-row">
                <h2>Your clinical summary</h2>
                <button type="button" onClick={() => void refreshExplanation()} disabled={state.isSummaryRefreshing}>
                  {state.isSummaryRefreshing ? "Updating..." : "Refresh summary"}
                </button>
              </div>
              {state.isSummaryRefreshing ? (
                <p className="muted">Refreshing explanation for the current profile...</p>
              ) : (
                renderSummaryMarkdown(explanation)
              )}
            </>
          ) : (
            <div className="summary-empty-state">
              <h2>Your clinical summary</h2>
              <button type="button" onClick={() => void refreshExplanation()} disabled={state.isSummaryRefreshing}>
                {state.isSummaryRefreshing ? "Generating..." : "Generate summary"}
              </button>
              <p className="muted">{supportiveSummary(riskProbability)}</p>
            </div>
          )}
          {state.error && <p className="error-text">{state.error}</p>}
        </article>
        <section className="split-grid">
          {simulator}
          <article className="card patient-result">
            <h2>Estimated Treatment Response</h2>
            <p className="risk-value">{pct(1 - riskProbability)}</p>
            <p className="muted">This estimate is based on your current profile.</p>
          </article>
        </section>
      </ResultsShell>
    );
  }

  if (route === "clinician") {
    return (
      <ResultsShell role="clinician" onNavigate={navigate}>
        {sharedHeader}
        {simulator}
        {explanationCard}
        <section className="split-grid">
          {shapCard}
          <article className="card">
            <h2>Probability Trend (Time-to-Event)</h2>
            <TimeCurve values={trend} />
          </article>
        </section>
      </ResultsShell>
    );
  }

  return (
    <ResultsShell role="scientist" onNavigate={navigate}>
      {sharedHeader}
      {explanationCard}
      <section className="split-grid">
        {shapCard}
        <article className="card">
          <h2>t-SNE Population Map</h2>
          {tsne ? <TsneChart tsne={tsne} selected={selectedTsne} /> : <p className="muted">t-SNE data unavailable.</p>}
        </article>
      </section>
    </ResultsShell>
  );
}

export default App;
