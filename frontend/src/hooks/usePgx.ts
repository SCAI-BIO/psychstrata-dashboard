import { useCallback, useEffect, useState } from "react";
import {
  fetchPgxInterpretation,
  fetchPgxOptions,
  type PgxInterpretRequest,
  type PgxInterpretResponse,
  type PgxOptionsResponse
} from "../api";
import { isDiplotypeComplete, type PatientGenetics } from "../domain/patient";

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
  const entries = Object.entries(genetics.genes)
    .filter(([, diplotype]) => isDiplotypeComplete(diplotype))
    .sort(([left], [right]) => left.localeCompare(right));
  const hasDiplotype = genetics.available && entries.length > 0;
  const alleleKey = hasDiplotype
    ? entries.map(([gene, diplotype]) => `${gene}:${diplotype.allele1}/${diplotype.allele2}`).join("|")
    : "";

  const [state, setState] = useState<PgxInterpretationState>({ status: "idle" });

  useEffect(() => {
    if (!hasDiplotype) {
      setState({ status: "idle" });
      return;
    }

    let isActive = true;
    setState({ status: "loading" });

    const payload: PgxInterpretRequest = { genes: Object.fromEntries(entries) };

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
