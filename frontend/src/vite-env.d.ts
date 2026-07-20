/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

// The minified Plotly bundle ships no type declarations; it's only handed to
// react-plotly.js's factory (which accepts an opaque plotly object).
declare module "plotly.js-dist-min";
