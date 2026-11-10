import { Dna, ExternalLink } from "lucide-react";
import type { PgxDrugRecommendation, PgxGeneInterpretation } from "../../api";
import type { Patient } from "../../domain/patient";
import { usePgxInterpretation } from "../../hooks/usePgx";
import { Card, SectionLabel } from "../Card";

const NEUTRAL_BADGE = "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300";
const CAUTION_BADGE = "bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-200";
const NORMAL_BADGE = "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-800 dark:text-emerald-200";

function geneBadge(gene: PgxGeneInterpretation): string {
  if (gene.is_indeterminate || !gene.priority) return NEUTRAL_BADGE;
  return gene.priority.toLowerCase().includes("high") ? CAUTION_BADGE : NORMAL_BADGE;
}

/** Active agents inferred from the `<drug>_mg` clinical values. */
export function activeRegimenDrugs(patient: Patient): string[] {
  return Object.entries(patient.clinical)
    .filter(([id, value]) => id.endsWith("_mg") && value > 0)
    .map(([id]) => id.slice(0, -"_mg".length).toLowerCase());
}

function GeneTile({ gene }: { gene: PgxGeneInterpretation }) {
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-700 px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{gene.gene}</p>
        <span className={`text-[11px] font-bold uppercase tracking-wide px-2 py-1 rounded-md ${geneBadge(gene)}`}>
          {gene.phenotype}
        </span>
      </div>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
        {gene.diplotype}
        {gene.activity_score !== null && ` · Activity score ${gene.activity_score}`}
      </p>
      {gene.priority && (
        <p className="mt-1 text-[11px] text-slate-400 dark:text-slate-500">{gene.priority}</p>
      )}
    </div>
  );
}

function DrugRow({ recommendation }: { recommendation: PgxDrugRecommendation }) {
  const isAvailable = recommendation.status === "available";
  return (
    <li className="py-3 border-b border-slate-100 dark:border-slate-800 last:border-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{recommendation.drug}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
            {recommendation.recommendation ?? recommendation.unavailable_reason}
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-400 dark:text-slate-500">
            {recommendation.cpic_level && (
              <span>
                CPIC Level {recommendation.cpic_level}
                {recommendation.cpic_level_status ? ` · ${recommendation.cpic_level_status}` : ""}
              </span>
            )}
            {recommendation.guideline && (
              <a
                href={recommendation.guideline}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 hover:text-slate-600 dark:hover:text-slate-300"
              >
                Guideline <ExternalLink size={11} />
              </a>
            )}
            {recommendation.pmids.map((pmid) => (
              <a
                key={pmid}
                href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                target="_blank"
                rel="noreferrer"
                className="hover:text-slate-600 dark:hover:text-slate-300"
              >
                PMID {pmid}
              </a>
            ))}
          </div>
        </div>
        <span className={`shrink-0 text-[10px] font-bold uppercase tracking-wide px-2 py-1 rounded-md ${isAvailable ? NORMAL_BADGE : NEUTRAL_BADGE}`}>
          {isAvailable ? "Guidance" : "Unavailable"}
        </span>
      </div>
    </li>
  );
}

/**
 * Pharmacogenetic interpretation for the current patient, shown to the
 * clinician as a standalone layer. Never fed into the prediction model.
 */
export function GeneticsCard({ patient }: { patient: Patient }) {
  const state = usePgxInterpretation(patient.genetics);
  const regimen = activeRegimenDrugs(patient);

  const body = () => {
    if (!patient.genetics.available) {
      return (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Genomic data is marked unavailable for this patient. Enable it in the intake wizard to see CPIC guidance.
        </p>
      );
    }
    if (state.status === "idle") {
      return (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Enter CYP2C19 and CYP2D6 diplotypes in the intake wizard to interpret the current regimen.
        </p>
      );
    }
    if (state.status === "loading") {
      return <p className="text-sm text-slate-500 dark:text-slate-400">Interpreting diplotypes…</p>;
    }
    if (state.status === "error") {
      return <p className="text-xs text-red-600 dark:text-red-400">{state.message}</p>;
    }

    const { genes, drug_recommendations } = state.result;
    const byDrug = new Map(drug_recommendations.map((rec) => [rec.drug.toLowerCase(), rec]));
    const matching = regimen.flatMap((drug) => {
      const rec = byDrug.get(drug);
      return rec ? [rec] : [];
    });
    const uncovered = regimen.filter((drug) => !byDrug.has(drug));

    return (
      <div className="space-y-5">
        <div>
          <SectionLabel>Metabolizer Status</SectionLabel>
          <div className="grid grid-cols-2 gap-4">
            {genes.map((gene) => (
              <GeneTile key={gene.gene} gene={gene} />
            ))}
          </div>
        </div>

        <div>
          <SectionLabel>Current Regimen Guidance</SectionLabel>
          {matching.length > 0 ? (
            <ul>{matching.map((rec) => <DrugRow key={rec.drug} recommendation={rec} />)}</ul>
          ) : (
            <p className="text-sm text-slate-500 dark:text-slate-400">No active medications on the current regimen.</p>
          )}
          {uncovered.length > 0 && (
            <p className="mt-2 text-xs text-slate-400 dark:text-slate-500">
              No CPIC guidance in this demo for {uncovered.join(", ")}.
            </p>
          )}
        </div>

        <p className="text-[11px] text-slate-400 dark:text-slate-500 leading-relaxed">{state.result.disclaimer}</p>
      </div>
    );
  };

  return (
    <Card
      icon={Dna}
      title="Pharmacogenomic Profile (CPIC)"
      action={
        <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400 px-3 py-1.5 rounded-md">
          Interpretive layer — not a model feature
        </span>
      }
    >
      {body()}
    </Card>
  );
}
