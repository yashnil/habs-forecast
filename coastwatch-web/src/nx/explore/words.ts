/**
 * Explore's words, DOM-free (tests read them). One legend per view states the kinds (README §3: no repeated
 * caveats); each carries the mandatory words of its truth rule: chlorophyll "algae, not toxin" (T5); C-HARM the
 * event and threshold and "probability of conditions in the water, not toxin in seafood" (T9, T10), "past run"
 * when every lead is before today (T3); the outlook its typical error and "no signal" (T14, E finding 2); the
 * temperature "analysis" (T1); currents "relative speed" and "no streak is no data, not calm" (T24, T19).
 */
import { day, range } from "@/nx/ds/format";
import { runKindWord } from "@/lib/modelTime";
import type { FreshnessState } from "@/lib/freshness";
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

/**
 * A layer's kind word, the same on the layer switch, the rail and "Also": "measured", "analysis", "model",
 * "outlook", or "past run (Oct 8)" once every day a model or outlook covers is before today (T3).
 */
export function layerKindWord(p: Pick<ExploreProduct, "kind" | "pastRun" | "runIssued">): string {
  if (p.kind === "measured" || p.kind === "analysis") return p.kind;
  return runKindWord(p.kind, !!p.pastRun, p.runIssued ? day(p.runIssued) : null);
}

const cap = (x: string) => x.charAt(0).toUpperCase() + x.slice(1);

/** The legend's caption: kind · source · dates, then the mandatory words. */
export function legendCaption(p: ExploreProduct, f: ExploreFrame): string {
  switch (p.id) {
    case "chl":
      return `Measured · ${f.sensor} · ${f.label === "Latest" ? `latest view, ${f.stamp}` : f.stamp}${oldMark(f.fresh)}. Algae, not toxin.`;
    case "sst":
      return `${cap(KIND_LABEL[f.kind])} · ${f.sensor} · ${day(f.end)}.${f.kind === "analysis" ? " Blended from satellites and buoys, not a measurement at a point." : ""}`;
    case "charm":
      // design-lint-allow forbidden-word: T37 requires "does not mean open or safe" within one glance of a low probability
      return `Model · ${p.source} · ${f.stamp}.${p.pastRun ? " Past run: every day it covers is before today." : ""} Probability of conditions in the water, not toxin in seafood. Classes are display steps, not risk levels; a lower probability does not mean open or safe.`;
    case "outlook":
      return `${p.pastRun ? "Outlook, past run" : "Outlook"} · ${p.source} · ${f.stamp}.${p.pastRun ? " Past run: every day it covers is before today." : ""} Typical error about a factor of 2: read as higher or lower than usual, not as a value. Dotted: no signal, the usual for this time of year. Algae, not toxin.`;
  }
}

/** T17: a frame past its own layer's current limit (from the layer's freshness policy, never hardcoded). */
export const isOld = (fresh: FreshnessState) => fresh === "stale" || fresh === "historical";
/** " (old)", set beside the date or value itself (T17). */
export const oldMark = (fresh: FreshnessState) => (isOld(fresh) ? " (old)" : "");

export function currentsCaption(c: CurrentFrame): string {
  const old = isOld(c.fresh);
  return `Streaks: measured surface currents · HF radar, 2 km · ${c.stamp}${oldMark(c.fresh)}.${old ? " Older than the radar's own freshness limit, so drawn still." : ""} Relative speed; fainter where the radar measured the cell in fewer hours. No streak is no data, not calm.`;
}

/**
 * Currents on or off when Explore opens (T25, T27, T17): off over chlorophyll (moving lines over a bloom picture
 * read as the bloom drifting), off over a model or outlook, and off when the hour on offer is past the radar's
 * stale limit. On over temperature with a fresh-enough hour. The reader may still switch them on.
 */
export function currentsByDefault(product: Pick<ExploreProduct, "id" | "kind"> | null, hour: Pick<CurrentFrame, "fresh"> | null): boolean {
  if (!product || !hour) return false;
  if (product.id === "chl" || product.kind === "model" || product.kind === "outlook") return false;
  return hour.fresh === "current" || hour.fresh === "stale";
}

/**
 * Explore's first line (the 5-second test): what this view is, in plain words, before any legend or instrument.
 * The kind and date are in it; sensor names, coverage bars and the other layers wait behind "Sources and
 * coverage".
 */
export function leadSentence(p: ExploreProduct, f: ExploreFrame): string {
  switch (p.id) {
    case "chl":
      return f.label === "Latest" && f.start !== f.end ? `Algae seen from space, ${range(f.start, f.end)}: each spot shows its latest clear view. Not toxin.` : `Algae seen from space on ${day(f.end)}. Not toxin.`;
    case "sst":
      return `Sea surface temperature for ${day(f.end)}${f.kind === "analysis" ? ", an analysis blended from satellites and buoys" : ""}.`;
    case "charm":
      return `NOAA's model chance of conditions for domoic acid in the water${p.pastRun ? `, from a past run (${p.runIssued ? day(p.runIssued) : "date unknown"}) whose days are all before today` : `, ${frameSpanWords(f)}`}. Not toxin in seafood.`;
    case "outlook":
      return `Neritic's chlorophyll outlook: one average for ${range(f.start, f.end)}${p.pastRun ? `, a past run (${p.runIssued ? day(p.runIssued) : "date unknown"}) whose days are all before today` : ""}. Algae, not toxin.`;
  }
}

/** The outlook's value at one significant figure (T15). */
export function aboutWords(v: number): string {
  const p = Number(v.toPrecision(1));
  return `about ${p >= 1 ? p.toFixed(0) : p}`;
}

export function frameSpanWords(f: ExploreFrame): string {
  return f.start === f.end ? day(f.start) : range(f.start, f.end);
}
