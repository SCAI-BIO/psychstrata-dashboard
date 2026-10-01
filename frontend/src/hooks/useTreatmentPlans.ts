import { useCallback, useEffect, useState } from "react";
import { fetchTreatmentPlans, type TreatmentPlanRecord } from "../api";

export type TreatmentPlansState =
  | { status: "loading" }
  | { status: "ready"; plans: TreatmentPlanRecord[] }
  | { status: "error"; message: string };

/**
 * Loads the treatment plans for one patient. Re-fetches whenever patientId
 * changes. Same rule as usePatients: only mount a component that calls this
 * once the dashboard is `ready`, so the Basic auth header is already attached.
 */
export function useTreatmentPlans(patientId: string) {
  const [state, setState] = useState<TreatmentPlansState>({ status: "loading" });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let isActive = true;
    setState({ status: "loading" });

    fetchTreatmentPlans(patientId)
      .then((plans) => {
        if (isActive) setState({ status: "ready", plans });
      })
      .catch((error: unknown) => {
        if (isActive) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "Unable to load treatment plans."
          });
        }
      });

    return () => {
      isActive = false;
    };
  }, [patientId, reloadKey]);

  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  return { state, reload };
}