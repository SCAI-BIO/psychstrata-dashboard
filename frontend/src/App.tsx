import type { ReactNode } from "react";
import { AppShell } from "./components/AppShell";
import { PatientProvider } from "./context/PatientContext";
import { useDashboard } from "./hooks/useDashboard";
import type { ResultRoute } from "./types";
import { ClinicianView } from "./views/ClinicianView";
import { IntakeView } from "./views/IntakeView";
import { LoginView } from "./views/LoginView";
import { PatientView } from "./views/PatientView";
import { ScientistView } from "./views/ScientistView";
import "./App.css";

/**
 * Top-level orchestrator: owns the container hook, provides the shared patient
 * model, gates on auth, and routes to the intake wizard or a result view. All
 * logic lives in useDashboard; every view below is presentational.
 */
function App() {
  const dashboard = useDashboard();

  return <PatientProvider value={dashboard.patientApi}>{renderRoute(dashboard)}</PatientProvider>;
}

function renderRoute(dashboard: ReturnType<typeof useDashboard>): ReactNode {
  const { authRequired, isAuthenticated, route, state } = dashboard;

  if (authRequired && !isAuthenticated) {
    return <LoginView dashboard={dashboard} />;
  }

  if (route === "intake") {
    return <IntakeView dashboard={dashboard} />;
  }

  // ── Result routes: guard on the async data lifecycle ─────────────────────
  const role: ResultRoute = route === "patient" ? "patient" : route === "scientist" ? "scientist" : "clinician";

  if (state.status === "loading") {
    return <FullScreenMessage>Loading clinical model configuration…</FullScreenMessage>;
  }

  if (state.status === "error") {
    return (
      <FullScreenMessage>
        <span className="font-semibold text-slate-900">Backend unavailable.</span> {state.message}
      </FullScreenMessage>
    );
  }

  if (state.prediction === null) {
    return (
      <AppShell role={role} onNavigate={dashboard.navigate} onLogout={authRequired ? dashboard.signOut : undefined}>
        <div className="flex flex-col items-center justify-center gap-3 h-full text-center">
          <p className="text-sm text-slate-500">No results yet. Start with the patient intake to generate a prediction.</p>
          <button
            type="button"
            onClick={() => dashboard.navigate("intake")}
            className="bg-slate-900 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors"
          >
            Go to intake
          </button>
        </div>
      </AppShell>
    );
  }

  const prediction = state.prediction;

  return (
    <AppShell role={role} onNavigate={dashboard.navigate} onLogout={authRequired ? dashboard.signOut : undefined}>
      {route === "patient" && <PatientView dashboard={dashboard} ready={state} prediction={prediction} />}
      {route === "clinician" && <ClinicianView dashboard={dashboard} ready={state} prediction={prediction} />}
      {route === "scientist" && <ScientistView dashboard={dashboard} ready={state} prediction={prediction} />}
    </AppShell>
  );
}

function FullScreenMessage({ children }: { children: ReactNode }) {
  return (
    <main className="min-h-screen bg-[#faf7f5] flex items-center justify-center p-8">
      <p className="text-sm text-slate-500 text-center max-w-md">{children}</p>
    </main>
  );
}

export default App;
