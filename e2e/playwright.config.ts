import { defineConfig, devices } from "@playwright/test";

/**
 * System tests: run Playwright against a fully built docker-compose stack.
 *
 * The default base URL targets the production-like compose frontend
 * (http://localhost:3000, served by nginx). For local development against the
 * Vite dev server, override with E2E_BASE_URL=http://localhost:5173.
 */
const E2E_BASE_URL = process.env.E2E_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["html", { outputFolder: "playwright-report", open: "never" }]],
  // The stack performs slow one-off computations (e.g. t-SNE embedding) on
  // first load, so be generous with per-assertion and per-test timeouts.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: E2E_BASE_URL,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  outputDir: "test-results",
});
