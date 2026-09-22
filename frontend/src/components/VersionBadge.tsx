import { useEffect, useState } from "react";
import { fetchVersion } from "../api";

type VersionState =
  | { status: "loading" }
  | { status: "ready"; version: string }
  | { status: "unavailable" };

export function VersionBadge() {
  const [state, setState] = useState<VersionState>({ status: "loading" });

  useEffect(() => {
    let isMounted = true;
    fetchVersion()
      .then(({ version }) => {
        if (isMounted) setState({ status: "ready", version });
      })
      .catch(() => {
        if (isMounted) setState({ status: "unavailable" });
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const label = state.status === "loading"
    ? "Loading version…"
    : state.status === "ready"
      ? `Version ${state.version}`
      : "Version unavailable";

  return (
    <span
      aria-live="polite"
      className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
    >
      {label}
    </span>
  );
}
