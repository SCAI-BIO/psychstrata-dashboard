import { useEffect, useState } from "react";

import { API_BASE_URL, type ApiSummary, fetchSummary } from "./api";
import "./App.css";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; summary: ApiSummary }
  | { status: "error"; message: string };

function App() {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let isMounted = true;

    fetchSummary()
      .then((summary) => {
        if (isMounted) {
          setState({ status: "ready", summary });
        }
      })
      .catch((error: unknown) => {
        if (isMounted) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "Unable to reach the backend API."
          });
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">PsychStrata Dashboard</p>
        <h1>Treatment Resistance Classifier Demo</h1>
        <p className="lede">
          Minimal React frontend connected to a FastAPI backend. This scaffold is ready for
          iterative replacement with the full dashboard experience.
        </p>

        <div className="status-card" aria-live="polite">
          {state.status === "loading" && <p>Loading backend response...</p>}

          {state.status === "error" && (
            <>
              <h2>Backend unavailable</h2>
              <p>{state.message}</p>
              <p className="meta">Configured API base URL: {API_BASE_URL}</p>
            </>
          )}

          {state.status === "ready" && (
            <>
              <h2>{state.summary.title}</h2>
              <p>{state.summary.message}</p>
              <p className="disclaimer">{state.summary.disclaimer}</p>
              <p className="meta">Configured API base URL: {API_BASE_URL}</p>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

export default App;
