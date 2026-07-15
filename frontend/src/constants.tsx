import { Stethoscope, BarChart3, Eye, type LucideIcon } from "lucide-react";
import type { ResultRoute, Route } from "./types";

export const PSYCH_STRATA_LOGO_URL =
  "https://psych-strata.eu/wp-content/uploads/2023/05/logo_footer_blue.png";

export const AUTH_SESSION_KEY = "psychstrata-authenticated";

export const ROUTE_TO_PATH: Record<Route, string> = {
  intake: "/",
  patient: "/results/patient",
  clinician: "/results/clinician",
  scientist: "/results/scientist"
};

/**
 * Static demographic header shown in the sidebar patient card.
 * The backend model is anonymous, so these are presentation-only placeholders
 * that mirror the mockups. Swap for real patient metadata when available.
 */
export const PATIENT_META = {
  name: "John Doe",
  id: "98421",
  dob: "05/12/78",
  firstName: "John",
  avatarUrl:
    "https://images.unsplash.com/photo-1633332755192-727a05c4013d?w=96&h=96&fit=crop&crop=faces"
};

export const PREVIEW_MODE_TEXT =
  "This interface is for demonstration only. It does not provide medical advice or definitive diagnosis and is not a certified medical product.";

/** Sidebar navigation model. Order matches the mockups. */
export const NAV_ITEMS: { route: ResultRoute; label: string; icon: LucideIcon }[] = [
  { route: "clinician", label: "Medical View", icon: Stethoscope },
  { route: "scientist", label: "Scientific View", icon: BarChart3 },
  { route: "patient", label: "Patient View", icon: Eye }
];

/**
 * Static literature shown in the "AI Clinical Insights & Literature Review" card.
 * These are placeholder citations/links to match the mockups — there is no
 * backend feed for supporting evidence yet.
 */
export const SUPPORTING_EVIDENCE = [
  {
    title: "C-Reactive Protein as a Predictor of Treatment Resistance in Schizophrenia",
    source: "Journal of Clinical Psychiatry, 2023"
  },
  {
    title: "The Role of CYP2D6 Genotyping in Antipsychotic Efficacy",
    source: "Pharmacogenomics Journal, 2022"
  }
];

export const PUBMED_LINKS = [
  { label: "CRP & Resistance (PubMed)", href: "https://pubmed.ncbi.nlm.nih.gov/?term=CRP+treatment+resistance" },
  { label: "CYP2D6 Efficacy (PubMed)", href: "https://pubmed.ncbi.nlm.nih.gov/?term=CYP2D6+antipsychotic+efficacy" },
  { label: "PHQ-9 Predictive Value", href: "https://pubmed.ncbi.nlm.nih.gov/?term=PHQ-9+predictive+value" }
];
