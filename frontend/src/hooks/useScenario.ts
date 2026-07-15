// hooks/useScenario.ts
import { useEffect, useState } from "react";
import { clonePatient, withClinical, withExtra, type Patient } from "../domain/patient";

export function useScenario(patient: Patient, knownIds: Set<string>) {
  const [draft, setDraft] = useState<Patient>(() => clonePatient(patient));

  // re-seed the scenario when a NEW baseline arrives (e.g. new prediction),
  // but not on every keystroke of the sim itself:
  useEffect(() => { setDraft(clonePatient(patient)); }, [patient]);

  const setFeature = (id: string, value: number) =>
    setDraft((d) => (knownIds.has(id) ? withClinical(d, id, value) : withExtra(d, id, value)));

  const reset = () => setDraft(clonePatient(patient));

  return { draft, setFeature, reset };
}