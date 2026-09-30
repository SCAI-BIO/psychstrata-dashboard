import { ArrowRight, ChevronRight, LogOut, Plus, RefreshCw } from "lucide-react";
import { Fragment,  useState, type ReactNode } from "react";
import { fetchTreatmentPlans, type PatientRecord, type TreatmentPlanRecord } from "../api";
import { PSYCH_STRATA_LOGO_URL } from "../constants";
import type { DashboardApi } from "../hooks/useDashboard";
import { usePatients } from "../hooks/usePatients";
import { useTreatmentPlans } from "../hooks/useTreatmentPlans";

// ── Formatting helpers ───────────────────────────────────────────────────────

const dateFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });

/** Parse a "YYYY-MM-DD" string as a local date (avoids the UTC-shift of `new Date(iso)`). */
function parseIsoDate(iso: string): Date | null {
  const [year, month, day] = iso.split("-").map(Number);
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
}

function formatDateOfBirth(iso: string): string {
  const date = parseIsoDate(iso);
  return date ? dateFormat.format(date) : iso;
}

function ageFromDateOfBirth(iso: string): number | null {
  const dob = parseIsoDate(iso);
  if (!dob) return null;
  const today = new Date();
  const hadBirthday =
    today.getMonth() > dob.getMonth() || (today.getMonth() === dob.getMonth() && today.getDate() >= dob.getDate());
  return today.getFullYear() - dob.getFullYear() - (hadBirthday ? 0 : 1);
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "—" : dateFormat.format(date);
}

/** "F33.1 — Major depressive disorder…" → code + optional description. Plain codes pass through. */
function splitDiagnosis(diagnosis: string): { code: string; description: string | null } {
  const [code, ...rest] = diagnosis.split(" — ");
  return { code: code.trim(), description: rest.length > 0 ? rest.join(" — ").trim() : null };
}

// ── View ─────────────────────────────────────────────────────────────────────

/** Read-only list of the clinician's patients, loaded from GET /api/patients. */
export function PatientListView({ dashboard }: { dashboard: DashboardApi }) {
  const { state } = dashboard;

  return (
    <main className="min-h-screen bg-[#faf7f5] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <header className="flex items-center justify-between px-8 py-4 border-b border-slate-200/60 dark:border-slate-700/60 bg-white dark:bg-slate-900">
        <div className="flex items-center">
          <img src={PSYCH_STRATA_LOGO_URL} alt="" className="h-7 w-7 object-contain" />
          <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100">PsychStrata CDSS</span>
        </div>
        <div className="flex items-center gap-3">
          
          {dashboard.authRequired && (
            <button
              type="button"
              onClick={dashboard.signOut}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 px-3 py-1.5 rounded-lg transition-colors"
            >
              <LogOut size={14} />
              Sign out
            </button>
          )}
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-10">
        {state.status === "loading" && <Notice>Loading clinical model configuration…</Notice>}
        {state.status === "error" && (
          <Notice>
            <span className="font-semibold text-slate-900 dark:text-slate-100">Backend unavailable.</span> {state.message}
          </Notice>
        )}
        {/* Mounted only once the dashboard is ready so auth is settled before we fetch. */}
        {state.status === "ready" && (
          <PatientsPanel
            onOpenResults={dashboard.openPatientResults}
            onAddPatient={() => dashboard.navigate("intake")}
          />
        )}
      </div>
    </main>
  );
}

type OpenResults = (patient: PatientRecord, plan: TreatmentPlanRecord | null) => Promise<void>;

/** Newest start date first (the list endpoint orders by created_at, which the seed data shares). */
function sortNewestFirst(plans: TreatmentPlanRecord[]): TreatmentPlanRecord[] {
  return [...plans].sort((a, b) => (b.start_date ?? "").localeCompare(a.start_date ?? ""));
}

function PatientsPanel({
  onOpenResults,
  onAddPatient
}: {
  onOpenResults: OpenResults;
  onAddPatient: () => void;
}) {
  const { state, isRefreshing, reload } = usePatients();
  // undefined = the user hasn't touched the dropdowns yet, so default to the latest patient
  // (the API returns newest first). null = the user collapsed everything.
  const [selectedId, setSelectedId] = useState<string | null | undefined>(undefined);
  const [openingKey, setOpeningKey] = useState<string | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);

  const patients = state.status === "ready" ? state.patients : [];
  const count = state.status === "ready" ? patients.length : null;
  const expandedId = selectedId === undefined ? (patients[0]?.id ?? null) : selectedId;

  /** `key` identifies which button is busy; plan === "latest" resolves the newest plan first. */
  const openResults = async (patient: PatientRecord, plan: TreatmentPlanRecord | "latest", key: string) => {
    setOpeningKey(key);
    setOpenError(null);
    try {
      const target =
        plan === "latest" ? (sortNewestFirst(await fetchTreatmentPlans(patient.id))[0] ?? null) : plan;
      await onOpenResults(patient, target); // navigates away on success
    } catch (error: unknown) {
      setOpenError(error instanceof Error ? error.message : "Unable to open results.");
      setOpeningKey(null);
    }
  };

  return (
    <>
      <div className="flex items-end justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patients</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {count === null ? "Current patients in the database." : `${count} ${count === 1 ? "patient" : "patients"} in the database.`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={reload}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 px-3 py-1.5 rounded-lg transition-colors"
          >
            <RefreshCw size={14} className={isRefreshing ? "animate-spin" : ""} />
            Refresh
          </button>
          <button
            type="button"
            onClick={onAddPatient}
            className="flex items-center gap-1.5 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 text-xs font-semibold px-3 py-1.5 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-300 transition-colors"
          >
            <Plus size={14} />
            Add New Patient
          </button>
        </div>
      </div>

      {openError && <p className="text-sm text-red-600 dark:text-red-400 mb-3">{openError}</p>}

      {state.status === "loading" && <Notice>Loading patients…</Notice>}
      {state.status === "error" && (
        <Notice>
          <span className="font-semibold text-slate-900 dark:text-slate-100">Couldn't load patients.</span> {state.message}
        </Notice>
      )}
      {state.status === "ready" &&
        (patients.length === 0 ? (
          <Notice>No patients yet. Use "Add New Patient" to create the first one.</Notice>
        ) : (
          <PatientTable
            patients={patients}
            expandedId={expandedId}
            openingKey={openingKey}
            onToggle={(id) => setSelectedId(expandedId === id ? null : id)}
            onOpenLatest={(patient) => void openResults(patient, "latest", patient.id)}
            onOpenPlan={(patient, plan) => void openResults(patient, plan, plan.id)}
          />
        ))}
    </>
  );
}

function PatientTable({
  patients,
  expandedId,
  openingKey,
  onToggle,
  onOpenLatest,
  onOpenPlan
}: {
  patients: PatientRecord[];
  expandedId: string | null;
  openingKey: string | null;
  onToggle: (id: string) => void;
  onOpenLatest: (patient: PatientRecord) => void;
  onOpenPlan: (patient: PatientRecord, plan: TreatmentPlanRecord) => void;
}) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200/70 dark:border-slate-700/70 shadow-sm overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs font-semibold text-slate-500 dark:text-slate-400 border-b border-slate-200/70 dark:border-slate-700/70">
            <th className="px-6 py-3">Patient</th>
            <th className="px-6 py-3">Date of birth</th>
            <th className="px-6 py-3">Diagnosis</th>
            <th className="px-6 py-3">Added</th>
            <th className="px-6 py-3">Last updated</th>
            <th className="px-6 py-3 text-right">Results</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {patients.map((patient) => {
            const expanded = patient.id === expandedId;
            return (
              <Fragment key={patient.id}>
                <PatientRow
                  patient={patient}
                  expanded={expanded}
                  isOpening={openingKey === patient.id}
                  onToggle={onToggle}
                  onOpenLatest={onOpenLatest}
                />
                {expanded && (
                  <tr className="bg-slate-50/70 dark:bg-slate-800/30">
                    <td colSpan={6} className="px-6 py-5">
                      <PatientDetails
                        patient={patient}
                        openingKey={openingKey}
                        onOpenPlan={(plan) => onOpenPlan(patient, plan)}
                      />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PatientRow({
  patient,
  expanded,
  isOpening,
  onToggle,
  onOpenLatest
}: {
  patient: PatientRecord;
  expanded: boolean;
  isOpening: boolean;
  onToggle: (id: string) => void;
  onOpenLatest: (patient: PatientRecord) => void;
}) {
  const { date_of_birth, diagnosis } = patient.clinical_data;
  const age = ageFromDateOfBirth(date_of_birth);
  const { code, description } = splitDiagnosis(diagnosis);

  return (
    <tr
      onClick={() => onToggle(patient.id)}
      aria-expanded={expanded}
      className={`align-top cursor-pointer transition-colors ${
        expanded ? "bg-blue-50/60 dark:bg-blue-950/30" : "hover:bg-slate-50 dark:hover:bg-slate-800/50"
      }`}
    >
      <td className="px-6 py-4">
        <div className="flex items-start gap-2">
          <ChevronRight
            size={14}
            className={`mt-1 flex-none text-slate-400 dark:text-slate-500 transition-transform ${expanded ? "rotate-90" : ""}`}
          />
          <div>
            <p className="font-semibold text-slate-900 dark:text-slate-100">
              {patient.first_name} {patient.last_name}
            </p>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">ID {patient.id.slice(0, 8)}</p>
          </div>
        </div>
      </td>
      <td className="px-6 py-4 text-slate-700 dark:text-slate-300">
        {formatDateOfBirth(date_of_birth)}
        {age !== null && <span className="text-slate-400 dark:text-slate-500"> ({age})</span>}
      </td>
      <td className="px-6 py-4">
        <span className="inline-block rounded-md bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 text-xs font-semibold px-2 py-0.5">
          {code}
        </span>
        {description && <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xs">{description}</p>}
      </td>
      <td className="px-6 py-4 text-slate-700 dark:text-slate-300">{formatTimestamp(patient.created_at)}</td>
      <td className="px-6 py-4 text-slate-700 dark:text-slate-300">{formatTimestamp(patient.updated_at)}</td>
      <td className="px-6 py-4 text-right">
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation(); // don't toggle the dropdown
            onOpenLatest(patient);
          }}
          disabled={isOpening}
          title="Open results using the latest treatment plan"
          className="inline-flex items-center gap-1.5 border border-[#5f8459] text-[#315f2c] bg-white dark:bg-slate-900 dark:border-[#6f9b68] dark:text-[#9bc596] text-xs font-semibold px-4 py-2 rounded-2xl hover:bg-[#f3f8f2] dark:hover:bg-[#182218] disabled:opacity-50 transition-colors"
        >
          {isOpening ? "Opening…" : "View Results"}
          {!isOpening && <ArrowRight size={14} />}
        </button>
      </td>
    </tr>
  );
}

function PatientDetails({
  patient,
  openingKey,
  onOpenPlan
}: {
  patient: PatientRecord;
  openingKey: string | null;
  onOpenPlan: (plan: TreatmentPlanRecord) => void;
}) {
  const { state, reload } = useTreatmentPlans(patient.id);
  const plans = state.status === "ready" ? sortNewestFirst(state.plans) : [];

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Treatment plans</h2>
        <button
          type="button"
          onClick={reload}
          className="text-xs font-medium text-slate-500 dark:text-slate-400 hover:underline"
        >
          Refresh
        </button>
      </div>

      {state.status === "loading" && <p className="text-sm text-slate-500 dark:text-slate-400">Loading treatment plans…</p>}
      {state.status === "error" && <p className="text-sm text-red-600 dark:text-red-400">{state.message}</p>}
      {state.status === "ready" &&
        (plans.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">
            No treatment plans for this patient yet. "View results" uses the default medication values.
          </p>
        ) : (
          <>
            <ul className="divide-y divide-slate-200/70 dark:divide-slate-700/70">
              {plans.map((plan, index) => (
                <PlanRow
                  key={plan.id}
                  plan={plan}
                  isLatest={index === 0}
                  isOpening={openingKey === plan.id}
                  onOpen={onOpenPlan}
                />
              ))}
            </ul>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-3">
              Click a plan to open its results in the Medical view.
            </p>
          </>
        ))}
    </div>
  );
}

function PlanRow({
  plan,
  isLatest,
  isOpening,
  onOpen
}: {
  plan: TreatmentPlanRecord;
  isLatest: boolean;
  isOpening: boolean;
  onOpen: (plan: TreatmentPlanRecord) => void;
}) {
  const start = plan.start_date ? formatDateOfBirth(plan.start_date) : "—";
  const end = plan.end_date ? formatDateOfBirth(plan.end_date) : "Ongoing";
  const meds = Object.entries(plan.medications)
    .filter(([, value]) => value > 0)
    .map(([id, value]) => `${id.replace("_mg", "")} ${value} mg`)
    .join(", ");

  return (
    <li>
      <button
        type="button"
        onClick={() => onOpen(plan)}
        disabled={isOpening}
        className="group flex w-full items-center justify-between gap-4 py-3 px-2 -mx-2 rounded-lg text-left text-sm hover:bg-white dark:hover:bg-slate-800/60 disabled:opacity-60 transition-colors"
      >
        <div>
          <p className="font-medium text-slate-900 dark:text-slate-100">
            {start} → {end}
            {isLatest && (
              <span className="ml-2 rounded-md bg-emerald-100 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5">
                Latest
              </span>
            )}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            {meds || "No active medication"}
            {plan.adherence.adherence_pct !== undefined && ` · adherence ${plan.adherence.adherence_pct}%`}
          </p>
        </div>
        <span className="flex flex-none items-center gap-1 text-xs font-medium text-slate-400 dark:text-slate-500 group-hover:text-slate-700 dark:group-hover:text-slate-200 transition-colors">
          {isOpening ? "Opening…" : ""}
          {!isOpening && <ArrowRight size={14} />}
        </span>
      </button>
    </li>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200/70 dark:border-slate-700/70 bg-white dark:bg-slate-900 px-6 py-10 text-center">
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">{children}</p>
    </div>
  );
}