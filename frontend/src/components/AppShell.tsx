import { Info, LogOut, Plus, Moon, Sun, Users } from "lucide-react";
import type { ReactNode } from "react";
import { NAV_ITEMS, PATIENT_META, PREVIEW_MODE_TEXT } from "../constants";
import { usePatient } from "../context/PatientContext";
import { useTheme } from "../context/ThemeContext";
import { displayName } from "../domain/patient";
import type { ResultRoute, Route } from "../types";
import { VersionBadge } from "./VersionBadge";
import { TopBar } from "./TopBar";


interface AppShellProps {
  role?: ResultRoute;
  onNavigate: (route: Route) => void;
  onLogout?: () => void;
  children: ReactNode;
}

/** Full result-view chrome: sidebar + top bar + scrolling content + footer. */
export function AppShell({ role, onNavigate, onLogout, children }: AppShellProps) {
 
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#faf7f5] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <TopBar onNavigate={onNavigate} onLogout={onLogout} />
      {}
      <div className="flex flex-1 overflow-hidden">
        {role && <Sidebar role={role} onNavigate={onNavigate} />}
        <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto px-8 py-6">{children}</main>
          <Footer />
        </div>
      </div>
    </div>
  );
}

function Sidebar({ role, onNavigate }: { role: ResultRoute; onNavigate: (route: Route) => void }) {
  return (
    <aside className="w-60 flex-none flex flex-col bg-white dark:bg-slate-800 border-r border-slate-200/70 dark:border-slate-700/70 px-4 py-5">
      <PatientCard />

      <nav className="flex flex-col gap-1 mt-2">
        {NAV_ITEMS.map(({ route, label, icon: Icon }) => {
          const active = role === route;
          return (
            <button
              key={route}
              type="button"
              onClick={() => onNavigate(route)}
              className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 font-medium"
                  : "text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}
            >
              <Icon size={16} />
              {label}
            </button>
          );
        })}
      </nav>

      <div className="mt-auto">
        <div className="rounded-xl border border-blue-100 dark:border-blue-900/50 bg-blue-50 dark:bg-blue-950/50 px-3 py-3">
          <div className="flex items-center gap-1.5 text-blue-700 dark:text-blue-300 mb-1.5">
            <Info size={13} />
            <span className="text-[11px] font-semibold uppercase tracking-wide">Assessment Context</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
            <span className="font-semibold text-slate-600 dark:text-slate-300">PREVIEW MODE</span>
            <br />
            {PREVIEW_MODE_TEXT}
          </p>
        </div>
      </div>
    </aside>
  );
}

function PatientCard() {
  const { patient } = usePatient();
  const { dob } = patient.demographics;
  return (
    <div className="flex items-center gap-3 px-1 mb-6">
      <div className="leading-tight">
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{displayName(patient)}</p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">DOB: {dob}</p>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="flex-none flex items-center justify-end gap-6 px-8 py-3 text-xs text-slate-400 dark:text-slate-500 border-t border-slate-200/60 dark:border-slate-700/60">
      <a href="#" className="hover:text-slate-600 dark:hover:text-slate-300 transition-colors">Privacy Policy</a>
      <a href="#" className="hover:text-slate-600 dark:hover:text-slate-300 transition-colors">Terms of Service</a>
    </footer>
  );
}


