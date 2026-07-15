import { useState } from "react";
import { PSYCH_STRATA_LOGO_URL } from "../constants";
import type { DashboardApi } from "../hooks/useDashboard";

/** Shared-password gate shown when VITE_APP_PASSWORD is configured. */
export function LoginView({ dashboard }: { dashboard: DashboardApi }) {
  const [password, setPassword] = useState("");

  return (
    <main className="min-h-screen bg-[#faf7f5] flex items-center justify-center p-8">
      <section className="bg-white rounded-2xl border border-slate-200/70 shadow-sm p-8 w-full max-w-md">
        <div className="flex items-center gap-3 mb-6">
          <img src={PSYCH_STRATA_LOGO_URL} alt="Psych-STRATA" className="h-7 w-7 object-contain" />
          <span className="text-xl font-bold tracking-tight text-slate-900">TheraPath</span>
        </div>
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Dashboard login</h1>
        <p className="text-sm text-slate-500 mb-6">Enter the shared password to access the dashboard.</p>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (dashboard.attemptLogin(password)) {
              setPassword("");
            }
          }}
        >
          <label className="flex flex-col gap-1" htmlFor="dashboard-password">
            <span className="text-xs font-medium text-slate-600">Password</span>
            <input
              id="dashboard-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </label>
          {dashboard.loginError && <p className="text-sm text-red-600">{dashboard.loginError}</p>}
          <button
            type="submit"
            className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors"
          >
            Sign in
          </button>
        </form>
      </section>
    </main>
  );
}
