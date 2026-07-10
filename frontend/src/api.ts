const DEFAULT_API_BASE_URL = "http://localhost:8000";

export type ApiSummary = {
  title: string;
  message: string;
  disclaimer: string;
};

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, "");

export async function fetchSummary(): Promise<ApiSummary> {
  const response = await fetch(`${API_BASE_URL}/api/summary`);

  if (!response.ok) {
    throw new Error(`Backend API request failed with status ${response.status}`);
  }

  return response.json() as Promise<ApiSummary>;
}
