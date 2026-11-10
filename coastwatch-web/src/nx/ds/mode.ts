/**
 * Chart modes (spec section 4): Day by default, Dusk when the system asks for a dark scheme, Night only when
 * the reader picks it. Each mode is its own token table in tokens.css; nothing is inverted.
 */
export type ChartMode = "day" | "dusk" | "night";
export const CHART_MODES: readonly ChartMode[] = ["day", "dusk", "night"];
export const CHART_MODE_KEY = "nx-chart-mode-v1";

export const CHART_MODE_LABEL: Record<ChartMode, string> = { day: "Day", dusk: "Dusk", night: "Night" };

export function isChartMode(v: unknown): v is ChartMode {
  return v === "day" || v === "dusk" || v === "night";
}

/** A stored choice wins; otherwise the system scheme picks Day or Dusk. Night is never automatic. */
export function resolveChartMode(stored: string | null | undefined, prefersDark: boolean): ChartMode {
  if (isChartMode(stored)) return stored;
  return prefersDark ? "dusk" : "day";
}

/**
 * Inline script for <head>: sets data-chart-mode on <html> before the first paint so a stored mode never
 * flashes. The shell owner decides whether to use it.
 */
export const CHART_MODE_SCRIPT = `(function(){try{var s=localStorage.getItem("${CHART_MODE_KEY}");var m=s==="day"||s==="dusk"||s==="night"?s:(matchMedia("(prefers-color-scheme: dark)").matches?"dusk":"day");document.documentElement.setAttribute("data-chart-mode",m)}catch(e){}})()`;
