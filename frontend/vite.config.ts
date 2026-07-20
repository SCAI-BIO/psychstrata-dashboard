import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

// Minimal typing for the one Node global we read here; avoids depending on
// @types/node (which isn't a direct dependency) just for the config file.
declare const process: { env: Record<string, string | undefined> };

// Backend target for local dev only. In production the app uses relative
// /api paths that are proxied by nginx to the backend service.
// Override via VITE_DEV_PROXY_TARGET (e.g. http://backend:8000 inside Docker);
// defaults to localhost for running `pnpm dev` directly on the host.
const DEV_BACKEND_TARGET =
  process.env.VITE_DEV_PROXY_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: DEV_BACKEND_TARGET,
        changeOrigin: true
      }
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts"
  }
});