/**
 * Explore's words, DOM-free (tests read them). One legend per view states the kinds (README §3: no repeated
 * caveats); each carries the mandatory words of its truth rule: chlorophyll "algae, not toxin" (T5); C-HARM the
 * event and threshold and "probability of conditions in the water, not toxin in seafood" (T9, T10), "past run"
 * when every lead is before today (T3); the outlook its typical error and "no signal" (T14, E finding 2); the
 * temperature "analysis" (T1); currents "relative speed" and "no streak is no data, not calm" (T24, T19).
 */
import { day, range } from "@/nx/ds/format";
import type { CurrentFrame, ExploreFrame, ExploreProduct } from "./model";

export const CHL_DOMAIN: [number, number] = [0.05, 50];
export const CHL_TICKS = [0.1, 1, 10];
export const CHL_STEPS = Array.from({ length: 8 }, (_, i) => `--chl-${i}`);
/** Isotherms: every whole °F, heavier every 2 °F (MUR is in °C: F = C x 1.8 + 32). */
export const ISO_F = { step: 1, index: 2, scale: 1.8, offset: 32 } as const;
/** C-HARM's display classes, cw-probability-classes-v1: ten 10-point intervals (display intervals, not risk levels, T12). */
export const PROB_STEPS = Array.from({ length: 10 }, (_, i) => `--prob-${i}`);

export const LAYER_LABEL: Record<ExploreProduct["id"], string> = { chl: "Chlorophyll", sst: "Temperature", charm: "C-HARM", outlook: "Outlook" };
export const KIND_LABEL: Record<ExploreFrame["kind"], string> = { measured: "measured", analysis: "analysis", model: "model", outlook: "outlook" };

const cap = (x: string) => x.charAt(0).toUpperCase() + x.slice(1);

/** The legend's caption: kind · source · dates, then the mandatory words. */
export function legendCaption(p: ExploreProduct, f: ExploreFrame): string {
  switch (p.id) {
    case "chl":
      return `Measured · ${f.sensor} · ${f.label === "Latest" ? `latest view, ${f.stamp}` : f.stamp}. Algae, not toxin.`;
    case "sst":
      return `${cap(KIND_LABEL[f.kind])} · ${f.sensor} · ${day(f.end)}.${f.kind === "analysis" ? " Blended from satellites and buoys, not a measurement at a point." : ""}`;
    case "charm":
      return `Model · ${p.source} · ${f.stamp}.${p.pastRun ? " Past run: every day it covers is before today." : ""} Probability of conditions in the water, not toxin in seafood. Classes are display steps, not risk levels.`;
    case "outlook":
      return `Outlook · ${p.source} · ${f.stamp}. Typical error about a factor of 2: read as higher or lower than usual, not as a value. Dotted: no signal, the usual for this time of year. Algae, not toxin.`;
  }
}

export function currentsCaption(c: CurrentFrame): string {
  return `Streaks: measured surface currents · HF radar, 2 km · ${c.mean ? c.stamp : c.stamp}. Relative speed; fainter where the radar measured the cell in fewer hours. No streak is no data, not calm.`;
}

/** The outlook's value at one significant figure (T15). */
export function aboutWords(v: number): string {
  const p = Number(v.toPrecision(1));
  return `about ${p >= 1 ? p.toFixed(0) : p}`;
}

export function frameSpanWords(f: ExploreFrame): string {
  return f.start === f.end ? day(f.start) : range(f.start, f.end);
}
