const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, "");

export type FeatureOption = {
  label: string;
  value: number;
};

export type FeatureSchema = {
  id: string;
  label: string;
  kind: "numeric" | "categorical";
  default: number;
  params: {
    min?: number;
    max?: number;
    step?: number;
    options?: FeatureOption[];
  };
  min?: number;
  max?: number;
  step?: number;
  options?: FeatureOption[];
};

export type FeaturesResponse = {
  features: FeatureSchema[];
  defaults: Record<string, number>;
  model_feature_order: string[];
  confidence_level: {
    default: number;
    min: number;
    max: number;
    step: number;
  };
};

export type ConformalPrediction = {
  confidence_level: number;
  alpha: number;
  label: "Responsive" | "Resistant" | "Uncertain";
  included_classes: string[];
};

export type PredictionResponse = {
  features: Record<string, number>;
  prediction: {
    probability_resistance: number;
    predicted_class: "Responsive" | "Resistant";
    conformal_prediction: ConformalPrediction;
  };
  shap_values: Array<{
    feature_id: string;
    feature_label: string;
    selected_value: number;
    selected_value_label: string;
    shap_value: number;
    abs_shap_value: number;
    direction: "raises" | "lowers" | "neutral";
  }>;
  top_contributors: {
    positive: Array<{
      feature_id: string;
      feature_label: string;
      selected_value: string;
      shap_value: number;
      direction: "raises" | "lowers";
    }>;
    negative: Array<{
      feature_id: string;
      feature_label: string;
      selected_value: string;
      shap_value: number;
      direction: "raises" | "lowers";
    }>;
  };
  tsne: {
    selected: {
      x: number;
      y: number;
    };
  };
  disclaimer: string;
};

export type ExplainResponse = {
  features: Record<string, number>;
  prediction: PredictionResponse["prediction"];
  top_contributors: PredictionResponse["top_contributors"];
  explanation: string;
};

export type TsneResponse = {
  points: Array<{
    x: number;
    y: number;
    class_value: 0 | 1;
    class_label: "Responsive" | "Resistant";
  }>;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail;
    } catch {
      detail = undefined;
    }
    throw new Error(detail ? `Backend API error (${response.status}): ${detail}` : `Backend API request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchFeatures(): Promise<FeaturesResponse> {
  return requestJson<FeaturesResponse>("/api/features");
}

export function fetchTsne(): Promise<TsneResponse> {
  return requestJson<TsneResponse>("/api/tsne");
}

export function fetchPredict(payload: {
  features: Record<string, number>;
  confidence_level: number;
}): Promise<PredictionResponse> {
  return requestJson<PredictionResponse>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export function fetchExplain(payload: {
  features: Record<string, number>;
  confidence_level: number;
}): Promise<ExplainResponse> {
  return requestJson<ExplainResponse>("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}
