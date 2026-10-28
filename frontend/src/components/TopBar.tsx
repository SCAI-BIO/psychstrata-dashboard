import { LogOut, Moon, Plus, Sun, Users } from "lucide-react";
import { PSYCH_STRATA_LOGO_URL } from "../constants";
import { useTheme } from "../context/ThemeContext";
import type { Route } from "../types";
import { VersionBadge } from "./VersionBadge";

interface TopBarProps {
  onNavigate: (route: Route) => void;
  /** Only passed when the backend requires auth; hides "Sign out" otherwise. */
  onLogout?: () => void;
}

const actionClass =
  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors";
const outlineClass =
  "border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800";

/** Global header shown on every page: logo + version on the left, shared actions on the right. */
export function TopBar({ onNavigate, onLogout }: TopBarProps) {
  return (
    <header className="flex-none flex items-center justify-between px-8 py-3 border-b border-slate-200/60 dark:border-slate-700/60 bg-white dark:bg-slate-900">
      <div className="flex items-center gap-3">
        <img src={PSYCH_STRATA_LOGO_URL} alt="" className="h-7 w-7 object-contain" />
        <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100">PsychStrata CDSS</span>
        <VersionBadge />
      </div>

      <div className="flex items-center gap-2">
        <button type="button" onClick={() => onNavigate("patients")} className={`${actionClass} ${outlineClass}`}>
          <Users size={14} />
          Patient Overview
        </button>
        <button
          type="button"
          onClick={() => onNavigate("intake")}
          className={`${actionClass} bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 hover:bg-slate-700 dark:hover:bg-slate-300`}
        >
          <Plus size={14} />
          Add New Patient
        </button>
        {onLogout && (
          <button type="button" onClick={onLogout} className={`${actionClass} ${outlineClass}`}>
            <LogOut size={14} />
            Sign out
          </button>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label="Toggle dark mode"
      className={`${actionClass} text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-700 dark:hover:text-slate-200`}
    >
      {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
      {theme === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}