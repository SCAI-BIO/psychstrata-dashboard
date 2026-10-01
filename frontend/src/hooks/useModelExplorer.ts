import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchPredict, type FeatureSchema, type PredictionResponse } from "../api";

interface UseModelExplorerOptions {
  /** The patient's feature vector; seeds the draft and defines "reset". */
  baselineFeatures: Record<string, number>;
  /** Backend feature schema used to validate which ids are editable. */
  features: FeatureSchema[];
  confidenceLevel: number;
  dateOfBirth?: string | null;
  /** How long to wait after the last edit before recomputing the prediction. */
  debounceMs?: number;
}

function recordsEqual(a: Record<string, number>, b: Record<string, number>): boolean {
  const aKeys = Object.keys(a);
  if (aKeys.length !== Object.keys(b).length) return false;
  return aKeys.every((key) => a[key] === b[key]);
}

/**
 * Read-only model sandbox for the Scientific View. Holds a draft feature vector
 * the user can tweak freely; predictions are recomputed against the backend
 * (debounced) and kept entirely local — the real patient is never mutated.
 * `result` stays null until the draft deviates from the baseline, so callers
 * fall back to the patient's prediction while the explorer is untouched.
 */
export function useModelExplorer({
  baselineFeatures,
  features,
  confidenceLevel,
  dateOfBirth = null,
  debounceMs = 500
}: UseModelExplorerOptions) {
  const [draft, setDraft] = useState<Record<string, number>>(() => ({ ...baselineFeatures }));
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestIdRef = useRef(0);
  const knownIds = useMemo(() => new Set(features.map((f) => f.id)), [features]);
  const baselineKey = JSON.stringify(baselineFeatures);

  // Re-seed the draft when a NEW baseline arrives (e.g. a fresh prediction),
  // but not on every explorer keystroke of its own.
  useEffect(() => {
    setDraft({ ...baselineFeatures });
    setResult(null);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baselineKey]);

  const setValue = useCallback(
    (id: string, value: number) => {
      setDraft((prev) => (knownIds.has(id) ? { ...prev, [id]: value } : prev));
    },
    [knownIds]
  );

  const reset = useCallback(() => {
    setDraft({ ...baselineFeatures });
    setResult(null);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baselineKey]);

  // Debounced auto-run: recompute the prediction shortly after the draft changes.
  useEffect(() => {
    if (recordsEqual(draft, baselineFeatures)) return;
    setIsSubmitting(true);
    setError(null);
    const timer = window.setTimeout(() => {
      const requestId = ++requestIdRef.current;
      fetchPredict({
        features: draft,
        confidence_level: confidenceLevel,
        date_of_birth: dateOfBirth
      })
        .then((response) => {
          if (requestIdRef.current !== requestId) return;
          setResult(response);
          setIsSubmitting(false);
        })
        .catch((e: unknown) => {
          if (requestIdRef.current !== requestId) return;
          setError(e instanceof Error ? e.message : "Prediction request failed.");
          setIsSubmitting(false);
        });
    }, debounceMs);
    return () => window.clearTimeout(timer);
  }, [draft, baselineFeatures, confidenceLevel, dateOfBirth, debounceMs]);

  return { draft, result, isSubmitting, error, setValue, reset };
}
