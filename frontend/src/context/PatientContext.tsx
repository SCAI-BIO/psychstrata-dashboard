import { createContext, useContext } from "react";
import type {
  Patient,
  PatientDemographics,
  PatientGenetics,
  PatientProfilePatch,
  PatientProteomics
} from "../domain/patient";

/**
 * Read/write surface for the shared patient model. Produced by useDashboard and
 * distributed via context so the sidebar, wizard and views all read one source.
 */
export interface PatientApi {
  patient: Patient;
  /** Feature vector projection (what the model receives). */
  featureValues: Record<string, number>;
  /** Read a clinical value by feature id (known feature or captured extra). */
  readFeature: (id: string) => number | undefined;
  /** Write a clinical value; routed to the feature vector or extras by schema. */
  writeFeature: (id: string, value: number) => void;
  resetPatient: () => void;
  updateDemographics: (patch: Partial<PatientDemographics>) => void;
  updateGenetics: (patch: Partial<PatientGenetics>) => void;
  updateProteomics: (patch: Partial<PatientProteomics>) => void;
  updateProfile: (patch: PatientProfilePatch) => void;
}

const PatientContext = createContext<PatientApi | null>(null);

export const PatientProvider = PatientContext.Provider;

export function usePatient(): PatientApi {
  const ctx = useContext(PatientContext);
  if (!ctx) {
    throw new Error("usePatient must be used within a PatientProvider");
  }
  return ctx;
}
