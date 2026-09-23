import { LogOut, Plus, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import type { PatientRecord } from "../api";
import { PSYCH_STRATA_LOGO_URL } from "../constants";
import type { DashboardApi } from "../hooks/useDashboard";
import { usePatients } from "../hooks/usePatients";

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
          <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100">TheraPath</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => dashboard.navigate("intake")}
            className="flex items-center gap-2 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 text-sm font-semibold px-4 py-2 rounded-lg hover:bg-slate-700 dark:hover:bg-slate-300 transition-colors"
          >
            <Plus size={16} />
            Add New Patient
          </button>
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
        {state.status === "ready" && <PatientsPanel />}
      </div>
    </main>
  );
}

function PatientsPanel() {
  const { state, isRefreshing, reload } = usePatients();
  const count = state.status === "ready" ? state.patients.length : null;

  return (
    <>
      <div className="flex items-end justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patients</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {count === null ? "Current patients in the database." : `${count} ${count === 1 ? "patient" : "patients"} in the database.`}
          </p>
        </div>
        <button
          type="button"
          onClick={reload}
          disabled={isRefreshing}
          className="flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 px-3 py-1.5 rounded-lg transition-colors"
        >
          <RefreshCw size={14} className={isRefreshing ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {state.status === "loading" && <Notice>Loading patients…</Notice>}
      {state.status === "error" && (
        <Notice>
          <span className="font-semibold text-slate-900 dark:text-slate-100">Couldn't load patients.</span> {state.message}
        </Notice>
      )}
      {state.status === "ready" &&
        (state.patients.length === 0 ? (
          <Notice>No patients yet. Use "Add New Patient" to create the first one.</Notice>
        ) : (
          <PatientTable patients={state.patients} />
        ))}
    </>
  );
}

function PatientTable({ patients }: { patients: PatientRecord[] }) {
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
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {patients.map((patient) => (
            <PatientRow key={patient.id} patient={patient} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PatientRow({ patient }: { patient: PatientRecord }) {
  const { date_of_birth, diagnosis } = patient.clinical_data;
  const age = ageFromDateOfBirth(date_of_birth);
  const { code, description } = splitDiagnosis(diagnosis);

  return (
    <tr className="align-top">
      <td className="px-6 py-4">
        <p className="font-semibold text-slate-900 dark:text-slate-100">
          {patient.first_name} {patient.last_name}
        </p>
        <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">ID {patient.id.slice(0, 8)}</p>
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
    </tr>
  );
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200/70 dark:border-slate-700/70 bg-white dark:bg-slate-900 px-6 py-10 text-center">
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">{children}</p>
    </div>
  );
}