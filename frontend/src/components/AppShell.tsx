import { Info, LogOut, Plus, Moon, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { NAV_ITEMS, PATIENT_META, PREVIEW_MODE_TEXT, PSYCH_STRATA_LOGO_URL } from "../constants";
import { usePatient } from "../context/PatientContext";
import { useTheme } from "../context/ThemeContext";
import type { ResultRoute, Route } from "../types";


interface SidebarProps {
  role: ResultRoute;
  onNavigate: (route: Route) => void;
  onLogout?: () => void;
}

interface AppShellProps extends SidebarProps {
  children: ReactNode;
}

/** Full result-view chrome: sidebar + top bar + scrolling content + footer. */
export function AppShell({ role, onNavigate, onLogout, children }: AppShellProps) {
  // const { resetPatient } = usePatient();
  
  // const handleNewPatient = () => {
  //   resetPatient();
  //   onNavigate("intake");
  // };

  return (
    <div className="flex h-screen overflow-hidden bg-[#faf7f5] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <Sidebar role={role} onNavigate={onNavigate} onLogout={onLogout} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-y-auto px-8 py-6">{children}</main>
        <Footer />
      </div>
    </div>
  );
}

function Sidebar({ role, onNavigate, onLogout }: SidebarProps) {
  return (
    <aside className="w-60 flex-none flex flex-col bg-white dark:bg-slate-800 border-r border-slate-200/70 dark:border-slate-700/70 px-4 py-5">
      <div className="flex items-center gap-2 px-1 mb-6">
        <img src={PSYCH_STRATA_LOGO_URL} alt="" className="h-7 w-7 object-contain" />
        <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100">TheraPath</span>
      </div>

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
        <div className="rounded-xl border border-blue-100 dark:border-blue-900/50 bg-blue-50 dark:bg-blue-950/50 px-3 py-3 mb-3">
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

        <button
          type="button"
          onClick={() => onNavigate("intake")}
          className="flex items-center justify-center gap-2 w-full px-3 py-2.5 rounded-lg bg-slate-700 dark:bg-slate-700 text-white text-sm font-medium hover:bg-slate-800 dark:hover:bg-slate-600 transition-colors"
        >
          <Plus size={16} />
          Add New Patient
        </button>

        {onLogout && (
          <button
            type="button"
            onClick={onLogout}
            className="flex items-center gap-2 w-full text-left px-3 py-2 mt-1 rounded-lg text-xs text-slate-400 dark:text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
          >
            <LogOut size={14} />
            Sign out
          </button>
        )}
        <ThemeToggle />      
      </div>
    </aside>
  );
}

function PatientCard() {
  const { patient } = usePatient();
  const { name, patientId, dob } = patient.demographics;
  return (
    <div className="flex items-center gap-3 px-1 mb-6">
      <div className="leading-tight">
        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{name}</p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">ID: {patientId}</p>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">DOB: {dob}</p>
      </div>
    </div>
  );
}

function TopBar() {
  return (
    <div className="flex-none flex items-center justify-end px-8 py-4 border-b border-slate-200/60 dark:border-slate-700/60">
      
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

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label="Toggle dark mode"
      className="flex items-center gap-2 w-full text-left px-3 py-2 rounded-lg text-xs text-slate-400 dark:text-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
    >
      {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
      {theme === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}
