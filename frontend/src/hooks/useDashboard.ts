import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchExplain, fetchFeatures, fetchPredict, fetchTsne, fetchAuthStatus, buildBasicAuthHeader, verifyBasicAuth, setBasicAuthHeader as setApiAuthHeader } from "../api";
import { ROUTE_TO_PATH } from "../constants";
import type { PatientApi } from "../context/PatientContext";
import {
  createDefaultPatient,
  patientToFeatures,
  withClinical,
  withClinicalMerged,
  withDemographics,
  withExtra,
  withGenetics,
  withProfile,
  withProteomics,
  type Patient
} from "../domain/patient";
import {
  clearAuthHeader,
  getInitialRoute,
  getStoredAuthHeader,
  persistAuthHeader,
} from "../lib/auth";
import type { LoadState, ResultRoute, Route, SimValues } from "../types";

/**
 * Container hook: owns dashboard state, the shared Patient model, and every
 * async action that talks to ./api. The patient is the single source of truth
 * for model inputs; `featureValues` is a derived projection of it.
 */
export function useDashboard() {
  const [route, setRoute] = useState<Route>(getInitialRoute());
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [patient, setPatient] = useState<Patient>(() => createDefaultPatient());
  const [loginError, setLoginError] = useState<string | null>(null);
  // Whether the backend requires auth (from /api/auth/status), and whether that
  // check has completed yet — used to hold data loading until we know.
  const [authRequired, setAuthRequired] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [basicAuthHeader, setBasicAuthHeader] = useState<string | null>(() => getStoredAuthHeader());
  const [isLoginSubmitting, setIsLoginSubmitting] = useState(false);
  const isAuthenticated = !authRequired || basicAuthHeader !== null;


  const [clinicianSimValues, setClinicianSimValues] = useState<SimValues>({
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
  const [clinicianAppliedImpactPct, setClinicianAppliedImpactPct] = useState(0);

  const stateRef = useRef(state);
  stateRef.current = state;
  const patientRef = useRef(patient);
  patientRef.current = patient;
  const simRef = useRef(clinicianSimValues);
  simRef.current = clinicianSimValues;

  const featureValues = useMemo(() => patientToFeatures(patient), [patient]);
  const knownFeatureIds = useMemo(
    () => (state.status === "ready" ? new Set(state.features.map((f) => f.id)) : new Set<string>()),
    [state]
  );

  // ── Routing ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const onPopState = () => setRoute(getInitialRoute());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((nextRoute: Route) => {
    const nextPath = ROUTE_TO_PATH[nextRoute];
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setRoute(nextRoute);
  }, []);

  // ── Sync clinician sim panel with the latest prediction ────────────────────
  const predictionKey = state.status === "ready" ? state.prediction : null;
  useEffect(() => {
    if (state.status !== "ready" || state.prediction === null) {
      return;
    }
    const clinical = patientRef.current.clinical;
    setClinicianSimValues({
      sertraline_mg: clinical.sertraline_mg ?? 0,
      lithium_mg: clinical.lithium_mg ?? 0,
      quetiapine_mg: clinical.quetiapine_mg ?? 0,
      adherence_pct: clinical.adherence_pct ?? 75
    });
    setClinicianImpactPct(0);
    setClinicianImpactError(null);
    setClinicianApplyError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.status, predictionKey]);

  // ── Initial data load (features + t-SNE), seeding the patient vector ────────
  useEffect(() => {
    // Wait until we know whether auth is required, then hold if a login is needed.
    if (!authChecked) return;
    if (authRequired && !isAuthenticated) {
      return;
    }

    let isMounted = true;
    Promise.all([fetchFeatures(), fetchTsne()])
      .then(([featuresPayload, tsnePayload]) => {
        if (!isMounted) return;
        setPatient((prev) => withClinicalMerged({ ...prev, clinical: {} }, featuresPayload.defaults));
        setState({
          status: "ready",
          features: featuresPayload.features,
          confidenceBounds: {
            min: featuresPayload.confidence_level.min,
            max: featuresPayload.confidence_level.max,
            step: featuresPayload.confidence_level.step
          },
          confidenceLevel: featuresPayload.confidence_level.default,
          prediction: null,
          explanation: "",
          tsne: tsnePayload,
          isSubmitting: false,
          isSummaryRefreshing: false,
          error: null
        });
      })
      .catch((error: unknown) => {
        if (!isMounted) return;
        setState({
          status: "error",
          message: error instanceof Error ? error.message : "Unable to reach the backend API."
        });
      });

    return () => {
      isMounted = false;
    };
  }, [authChecked, authRequired, isAuthenticated]);

  // ── Auth actions ───────────────────────────────────────────────────────────
    
  // learn whether auth is on, once, on mount:
  useEffect(() => {
    fetchAuthStatus()
      .then((status) => setAuthRequired(status.auth_enabled))
      .catch(() => setAuthRequired(false))
      .finally(() => setAuthChecked(true));
  }, []);
  // keep the api layer's header in sync so every request carries it:
  useEffect(() => { setApiAuthHeader(basicAuthHeader); }, [basicAuthHeader]);

  const attemptLogin = useCallback(async (username: string, password: string) => {
    const header = buildBasicAuthHeader(username.trim(), password);
    setIsLoginSubmitting(true);
    setLoginError(null);
    try {
      await verifyBasicAuth(header);
      persistAuthHeader(header);
      // Set the api-layer header synchronously so the data-load effect that
      // fires on this same state change already carries it.
      setApiAuthHeader(header);
      setBasicAuthHeader(header);
    } catch (e) {
      setLoginError(e instanceof Error ? e.message : "Unable to sign in.");
    } finally {
      setIsLoginSubmitting(false);
    }
  }, []);

  const signOut = useCallback(() => {
    clearAuthHeader();
    setApiAuthHeader(null);
    setBasicAuthHeader(null);
    setLoginError(null);
    setState({ status: "loading" });
    setPatient(createDefaultPatient());
    if (window.location.pathname !== ROUTE_TO_PATH.intake) window.history.pushState({}, "", ROUTE_TO_PATH.intake);
    setRoute("intake");
  }, []);

  // ── Patient model editing ───────────────────────────────────────────────────
  const setFeatureValue = useCallback(
    (id: string, value: number) => {
      setPatient((prev) => (knownFeatureIds.has(id) ? withClinical(prev, id, value) : withExtra(prev, id, value)));
    },
    [knownFeatureIds]
  );

  const readFeature = useCallback(
    (id: string): number | undefined =>
      knownFeatureIds.has(id) ? patient.clinical[id] : patient.extras[id],
    [knownFeatureIds, patient]
  );

  const resetPatient = useCallback(() => {
    const current = stateRef.current;
    const seed = current.status === "ready" ? Object.fromEntries(current.features.map((f) => [f.id, 0])) : {};
    setPatient(createDefaultPatient());
    // optional: clear the previous patient's results so nothing stale lingers
    setState((prev) => (prev.status === "ready" ? { ...prev, prediction: null, explanation: "" } : prev));
  }, []);

  const patientApi: PatientApi = useMemo(
    () => ({
      patient,
      featureValues,
      readFeature,
      writeFeature: setFeatureValue,
      updateDemographics: (patch) => setPatient((prev) => withDemographics(prev, patch)),
      updateGenetics: (patch) => setPatient((prev) => withGenetics(prev, patch)),
      updateProteomics: (patch) => setPatient((prev) => withProteomics(prev, patch)),
      updateProfile: (patch) => setPatient((prev) => withProfile(prev, patch)),
      resetPatient
    }),
    [patient, featureValues, readFeature, setFeatureValue, resetPatient]
  );

  const setConfidenceLevel = useCallback((value: number) => {
    setState((prev) => {
      if (prev.status !== "ready") return prev;
      return { ...prev, confidenceLevel: value };
    });
  }, []);

  // ── Core prediction ────────────────────────────────────────────────────────
  const runPrediction = useCallback(
    async (targetRoute: ResultRoute) => {
      const current = stateRef.current;
      if (current.status !== "ready") return;

      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return { ...prev, isSubmitting: true, isSummaryRefreshing: false, error: null };
      });

      try {
        const prediction = await fetchPredict({
          features: patientToFeatures(patientRef.current),
          confidence_level: current.confidenceLevel
        });
        setPatient((prev) => withClinicalMerged(prev, prediction.features));
        setState((prev) => {
          if (prev.status !== "ready") return prev;
          return { ...prev, prediction, explanation: "", isSubmitting: false, isSummaryRefreshing: false };
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
    },
    [navigate]
  );

  const refreshExplanation = useCallback(async () => {
    const current = stateRef.current;
    if (current.status !== "ready" || current.prediction === null) return;

    setState((prev) => {
      if (prev.status !== "ready") return prev;
      return { ...prev, isSummaryRefreshing: true, error: null };
    });

    try {
      const explanationPayload = await fetchExplain({
        features: patientToFeatures(patientRef.current),
        confidence_level: current.confidenceLevel
      });
      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return { ...prev, explanation: explanationPayload.explanation, isSummaryRefreshing: false };
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
  }, []);

  // ── Clinician "what-if" simulation ─────────────────────────────────────────
  const runClinicianImpactPrediction = useCallback(async () => {
    const current = stateRef.current;
    if (current.status !== "ready" || current.prediction === null) return;

    setIsClinicianImpactSubmitting(true);
    setClinicianImpactError(null);
    try {
      const response = await fetchPredict({
        features: { ...patientToFeatures(patientRef.current), ...simRef.current },
        confidence_level: current.confidenceLevel
      });
      const baselineRisk = current.prediction.prediction.probability_resistance;
      const simulatedRisk = response.prediction.probability_resistance;
      setClinicianImpactPct((baselineRisk - simulatedRisk) * 100);
    } catch (error: unknown) {
      setClinicianImpactError(error instanceof Error ? error.message : "Simulation request failed.");
    } finally {
      setIsClinicianImpactSubmitting(false);
    }
  }, []);

  const runScenarioImpact = useCallback(async (features: Record<string, number>) => {
    const current = stateRef.current;
    if (current.status !== "ready" || current.prediction === null) return;
    setIsClinicianImpactSubmitting(true);
    setClinicianImpactError(null);
    try {
      const response = await fetchPredict({ features, confidence_level: current.confidenceLevel });
      const baseline = current.prediction.prediction.probability_resistance;
      setClinicianImpactPct((baseline - response.prediction.probability_resistance) * 100);
    } catch (e) {
      setClinicianImpactError(e instanceof Error ? e.message : "Simulation request failed.");
    } finally {
      setIsClinicianImpactSubmitting(false);
    }
  }, []);

  const applyClinicianSimulationToPage = useCallback(async () => {
    const current = stateRef.current;
    if (current.status !== "ready") return;

    setIsClinicianApplySubmitting(true);
    setClinicianApplyError(null);
    try {
      const response = await fetchPredict({
        features: { ...patientToFeatures(patientRef.current), ...simRef.current },
        confidence_level: current.confidenceLevel
      });
      setPatient((prev) => withClinicalMerged(prev, response.features));
      setState((prev) => {
        if (prev.status !== "ready") return prev;
        return { ...prev, prediction: response, explanation: "", error: null };
      });
    } catch (error: unknown) {
      setClinicianApplyError(error instanceof Error ? error.message : "Apply request failed.");
    } finally {
      setIsClinicianApplySubmitting(false);
    }
  }, []);

  const applyScenario = useCallback(async (scenario: Patient) => {
    const current = stateRef.current;
    if (current.status !== "ready" || current.prediction === null) return;
    setIsClinicianApplySubmitting(true);
    setClinicianApplyError(null);
    try {
      const baseline = current.prediction.prediction.probability_resistance;
      const response = await fetchPredict({
        features: patientToFeatures(scenario),
        confidence_level: current.confidenceLevel
      });
      setClinicianAppliedImpactPct((baseline - response.prediction.probability_resistance) * 100);
      // commit the scenario as the real patient (syncing model-returned features)…
      setPatient(withClinicalMerged(scenario, response.features));
      // …and make its prediction the page's prediction, so risk/SHAP/stats update
      setState((prev) =>
        prev.status === "ready" ? { ...prev, prediction: response, explanation: "", error: null } : prev
      );
    } catch (e) {
      setClinicianApplyError(e instanceof Error ? e.message : "Apply request failed.");
    } finally {
      setIsClinicianApplySubmitting(false);
    }
  }, []);

  return {
    authRequired,
    isAuthenticated,
    loginError,
    isLoginSubmitting,
    attemptLogin,
    signOut,
    route,
    navigate,
    state,
    patient,
    patientApi,
    featureValues,
    setFeatureValue,
    setConfidenceLevel,
    runPrediction,
    refreshExplanation,
    clinicianSimValues,
    setClinicianSimValues,
    clinicianImpactPct,
    isClinicianImpactSubmitting,
    clinicianImpactError,
    isClinicianApplySubmitting,
    clinicianApplyError,
    clinicianAppliedImpactPct,
    runClinicianImpactPrediction,
    applyClinicianSimulationToPage,
    runScenarioImpact,
    applyScenario
  };
}

export type DashboardApi = ReturnType<typeof useDashboard>;
