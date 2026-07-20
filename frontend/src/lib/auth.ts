import { AUTH_HEADER_STORAGE_KEY, ROUTE_TO_PATH } from "../constants";
import type { Route } from "../types";

export function getStoredAuthHeader(): string | null {
  return sessionStorage.getItem(AUTH_HEADER_STORAGE_KEY);
}
export function persistAuthHeader(header: string): void {
  sessionStorage.setItem(AUTH_HEADER_STORAGE_KEY, header);
}
export function clearAuthHeader(): void {
  sessionStorage.removeItem(AUTH_HEADER_STORAGE_KEY);
}

/** Resolve the current route from the browser path (used on load + popstate). */
export function getInitialRoute(): Route {
  const pathname = window.location.pathname;
  if (pathname === ROUTE_TO_PATH.patient) return "patient";
  if (pathname === ROUTE_TO_PATH.clinician) return "clinician";
  if (pathname === ROUTE_TO_PATH.scientist) return "scientist";
  return "intake";
}
