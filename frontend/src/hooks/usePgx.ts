import { useCallback, useEffect, useState } from "react";
import {
  fetchPgxInterpretation,
  fetchPgxOptions,
  type PgxInterpretRequest,
  type PgxInterpretResponse,
  type PgxOptionsResponse
} from "../api";
import type { PatientGenetics } from "../domain/patient";

export type PgxOptionsState =
  | { status: "loading" }
  | { status: "ready"; options: PgxOptionsResponse }
  | { status: "error"; message: string };

/** Loads the allele/drug catalogue for the intake allele selects. */
export function usePgxOptions() {
  const [state, setState] = useState<PgxOptionsState>({ status: "loading" });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let isActive = true;
    setState({ status: "loading" });

    fetchPgxOptions()
      .then((options) => {
        if (isActive) setState({ status: "ready", options });
      })
      .catch((error: unknown) => {
        if (isActive) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "Unable to load pharmacogenetic options."
          });
        }
      });

    return () => {
      isActive = false;
    };
  }, [reloadKey]);

  const reload = useCallback(() => setReloadKey((key) => key + 1), []);

  return { state, reload };
}

export type PgxInterpretationState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; result: PgxInterpretResponse }
  | { status: "error"; message: string };

/**
 * Interprets the patient's current diplotypes. Idle when genetics is disabled
 * or no diplotype has been entered; refetches only when an allele changes.
 */
export function usePgxInterpretation(genetics: PatientGenetics): PgxInterpretationState {
  const { available, cyp2c19, cyp2d6 } = genetics;
  const hasDiplotype = available && Boolean(cyp2c19 || cyp2d6);
  const alleleKey = hasDiplotype
    ? [cyp2c19?.allele1, cyp2c19?.allele2, cyp2d6?.allele1, cyp2d6?.allele2].join("|")
    : "";

  const [state, setState] = useState<PgxInterpretationState>({ status: "idle" });

  useEffect(() => {
    if (!hasDiplotype) {
      setState({ status: "idle" });
      return;
    }

    let isActive = true;
    setState({ status: "loading" });

    const payload: PgxInterpretRequest = {};
    if (cyp2c19) payload.cyp2c19 = cyp2c19;
    if (cyp2d6) payload.cyp2d6 = cyp2d6;

    fetchPgxInterpretation(payload)
      .then((result) => {
        if (isActive) setState({ status: "ready", result });
      })
      .catch((error: unknown) => {
        if (isActive) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "Unable to interpret pharmacogenetic profile."
          });
        }
      });

    return () => {
      isActive = false;
    };
  }, [alleleKey, hasDiplotype]);

  return state;
}
