/**
 * Explore's words, DOM-free (tests read them). One legend per view states the kinds (README §3: no repeated
 * caveats); each carries the mandatory words of its truth rule: chlorophyll "algae, not toxin" (T5); C-HARM the
 * event and threshold and "probability of conditions in the water, not toxin in seafood" (T9, T10), "past run"
 * when every lead is before today (T3); the outlook its typical error and "no signal" (T14, E finding 2); the
 * temperature "analysis" (T1); currents "relative speed" and "no streak is no data, not calm" (T24, T19).
 */
import { day, fixed, range, sig } from "@/nx/ds/format";
import { runKindWord } from "@/lib/modelTime";
import type { FreshnessState } from "@/lib/freshness";
import type { CurrentFrame, ExploreFrame, ExploreProduct, ProductId } from "./model";
import type { GapPosition } from "./read";

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
      return `${cap(KIND_LABEL[f.kind])} · ${f.sensor} · ${day(f.end)}${oldMark(f.fresh)}.${f.kind === "analysis" ? " Blended from satellites and buoys, not a measurement at a point." : ""}`;
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

/** The legend's key for water a model or an outlook has no cell for (the footprint is the model's, not a low value). */
export const OUTSIDE_KEY = "Ruled: outside the model, no value";

/** A currents frame that is drawn still: an hour past the radar's stale limit (T17), and the 24 h mean (lead decision 7). */
export const currentsStill = (c: Pick<CurrentFrame, "fresh" | "mean">) => isOld(c.fresh) || c.mean;

/** "The radar measured 12 % of the coastal ocean this hour.", when the frame publishes its coverage. */
export function coverageWords(c: Pick<CurrentFrame, "volume" | "mean">): string | null {
  if (c.volume == null) return null;
  const p = Math.round(c.volume * 100);
  return `The radar measured ${p < 1 && c.volume > 0 ? "under 1" : p} % of the coastal ocean ${c.mean ? "over these 24 hours" : "this hour"}.`;
}

export function currentsCaption(c: CurrentFrame): string {
  const old = isOld(c.fresh);
  const still = old ? " Older than the radar's own freshness limit, so drawn still." : c.mean ? " A mean of 24 hours is not one moment of the water, so it is drawn still." : "";
  const cov = coverageWords(c);
  return `Streaks: measured surface currents · HF radar, 2 km · ${c.stamp}${oldMark(c.fresh)}.${still}${cov ? ` ${cov}` : ""} Relative speed; fainter where the radar measured the cell in fewer hours. No streak is no data, not calm.`;
}

/** What the reader is told about the streaks: nothing is claimed that is not drawn. */
export type CurrentsState = "off" | "over-model" | "static" | "loading" | "failed" | "empty" | "on";

/** The currents switch's second line. A failed or empty hour says so; it never reads "on, measured" with nothing drawn. */
export function currentsToggleWords(state: CurrentsState, c: Pick<CurrentFrame, "fresh" | "mean"> | null): string {
  switch (state) {
    case "off":
      return "off";
    case "over-model":
      return "not over a model";
    case "static":
      return "not on a still picture";
    case "loading":
      return "reading the radar";
    case "failed":
      return "unavailable: this hour could not be read";
    case "empty":
      return "no measurement in this hour";
    case "on":
      return `on, measured${c ? oldMark(c.fresh) : ""}${c?.mean ? ", 24 h mean" : ""}`;
  }
}

/** A reading's figure without its unit: chlorophyll to two figures, the outlook to one (T15), a probability as a whole percent, °F to a tenth. */
export function valueWords(id: ProductId, v: number): string {
  if (id === "chl") return sig(v, 2);
  if (id === "outlook") return aboutWords(v).replace("about ", "");
  if (id === "charm") return String(Math.round(v * 100));
  return fixed(v * 1.8 + 32, 1);
}
export const unitOf = (id: ProductId) => (id === "charm" ? "%" : id === "sst" ? "°F" : "mg/m³");

/**
 * Why a cell of a model, an outlook or an analysis has no value, from where the gap really is in the layer's own
 * grid (read.ts `gapPosition`), in the layer's own name. Never one fixed sentence for every gap.
 */
export function noValueWords(p: Pick<ExploreProduct, "id" | "kind">, pos: GapPosition | "outside-grid"): string {
  const what = p.id === "charm" ? "C-HARM" : p.id === "outlook" ? "the outlook" : p.kind === "analysis" ? "the analysis" : "this layer";
  const where =
    pos === "outside-grid"
      ? `Outside the area ${what} covers`
      : pos === "shoreward"
        ? `Between ${what}'s cells and the shore: ${what === "C-HARM" ? "the model leaves out the nearshore band and the bays" : `${what} has no cell this close to land`}`
        : pos === "seaward"
          ? `Beyond the seaward edge of ${what} at this latitude`
          : pos === "inside"
            ? `A gap inside the area ${what} covers`
            : `${what.charAt(0).toUpperCase() + what.slice(1)} has no cells at this latitude`;
  return `${where}. No value is not a low value.`;
}

/**
 * What a readout's stamp says a value's date is. A measured pixel has its own day; a model lead its valid day;
 * an outlook is ONE mean over a window, so it is dated by its issue and names the window beside its source,
 * never by the window's last day (which reads as "measured on that day").
 */
export function readingStamp(f: Pick<ExploreFrame, "kind" | "start" | "end" | "issued" | "sensor">, pixel: string | null): { date: string; dateRole: "measured" | "valid" | "issued" | undefined; source: string } {
  if (f.kind === "outlook") return { date: f.issued ?? f.end, dateRole: f.issued ? "issued" : undefined, source: `${f.sensor}, mean of ${range(f.start, f.end)}` };
  if (f.kind === "model") return { date: f.end, dateRole: "valid", source: f.issued ? `${f.sensor}, run of ${day(f.issued)}` : f.sensor };
  if (f.kind === "measured") return { date: pixel ?? f.end, dateRole: "measured", source: f.sensor };
  return { date: f.end, dateRole: undefined, source: f.sensor };
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
