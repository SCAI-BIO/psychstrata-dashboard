import { useCallback, useEffect, useState } from "react";
import { fetchPatients, type PatientRecord } from "../api";

export type PatientsState =
  | { status: "loading" }
  | { status: "ready"; patients: PatientRecord[] }
  | { status: "error"; message: string };

/**
 * Loads the current clinician's patients. Self-contained on purpose: the list is
 * server state that only this page needs, so it doesn't live in useDashboard.
 *
 * Only mount a component that calls this once the dashboard is `ready` (i.e.
 * auth has been resolved) — otherwise the request can fire before the Basic
 * auth header is attached and come back as a 401.
 */
export function usePatients() {
  const [state, setState] = useState<PatientsState>({ status: "loading" });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let isActive = true;
    setIsRefreshing(true);

    fetchPatients()
      .then((patients) => {
        if (isActive) setState({ status: "ready", patients });
      })
      .catch((error: unknown) => {
        if (isActive) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "Unable to load patients."
          });
        }
      })
      .finally(() => {
        if (isActive) setIsRefreshing(false);
      });

    return () => {
      isActive = false;
    };
  }, [reloadKey]);

  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  return { state, isRefreshing, reload };
}