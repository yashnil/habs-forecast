/**
 * Round 2 (B, D): the model and estimate layers read live from the app's own routes (lib/ocean), DOM-free so the
 * unit tests read it. They are never observations: WCOFS is model guidance (developmental), the DINEOF chlorophyll
 * is a gap-filled estimate. Near the shore, where the coarse grids are least reliable, nothing is drawn.
 */
import { cycleName, MASK_WATER, type ChlEstimateIndex, type OceanGrid, type WcofsIndex, type WcofsStep } from "@/lib/ocean/contract";
import { day, pacificYmd, time } from "@/nx/ds/format";

/** A copy of the codes with no value wherever the mask is not open water (land, or within the near-shore band). */
export function maskedCodes(g: Pick<OceanGrid, "nodata">, codes: Uint16Array, mask: Uint16Array | null): Uint16Array {
  if (!mask || mask.length !== codes.length) return codes;
  const out = codes.slice();
  for (let i = 0; i < out.length; i++) if (mask[i] !== MASK_WATER) out[i] = g.nodata;
  return out;
}

/** The same for a current field: no vector where the mask is not open water (streaks stop there). */
export function maskedVectors(u: Float32Array, v: Float32Array, mask: Uint16Array | null): { u: Float32Array; v: Float32Array } {
  if (!mask || mask.length !== u.length) return { u, v };
  const uu = u.slice(), vv = v.slice();
  for (let i = 0; i < uu.length; i++) if (mask[i] !== MASK_WATER) uu[i] = vv[i] = NaN;
  return { u: uu, v: vv };
}

/** The nowcast steps (Analysis / Estimates) and the forecast steps (Forecasts) of a WCOFS cycle, in time order. */
export function wcofsTracks(ix: WcofsIndex | null): { nowcast: WcofsStep[]; forecast: WcofsStep[] } {
  const steps = [...(ix?.steps ?? [])].sort((a, b) => a.valid_time.localeCompare(b.valid_time));
  return { nowcast: steps.filter((s) => s.track === "estimates"), forecast: steps.filter((s) => s.track === "forecasts") };
}

/** "Nov 14, 7:00 PM" for a step's valid time. */
export const stepWhen = (s: Pick<WcofsStep, "valid_time">) => `${day(s.valid_time)}, ${time(s.valid_time)}`;

/** One line of a WCOFS index's state: what it holds, or why there is nothing (never a substitute). */
export const NEEDS_CONNECTION = "Needs a connection: live model data is read when online.";
export function wcofsStateWords(ix: WcofsIndex | null, loaded: boolean, offline = false): string {
  if (offline) return NEEDS_CONNECTION;
  if (!loaded) return "Reading NOAA WCOFS…";
  if (!ix || ix.status === "unavailable" || !ix.cycle) return `NOAA WCOFS is unavailable${ix?.reason ? `: ${ix.reason}` : ""}.`;
  const v = ix.valid ? ` · valid ${stepWhen({ valid_time: ix.valid.start })} to ${stepWhen({ valid_time: ix.valid.end })}` : "";
  return `${ix.status === "expired" ? "Expired: " : ""}NOAA WCOFS model guidance (developmental) · cycle ${cycleName(ix.cycle.id)}${v}${ix.fallback ? " · the last good cycle (a newer one could not be checked)" : ""}`;
}

/** One line of the chlorophyll estimate's state: its day, and how old it is. */
export function chlEstimateStateWords(ix: ChlEstimateIndex | null, loaded: boolean, offline = false): string {
  if (offline) return NEEDS_CONNECTION;
  if (!loaded) return "Reading the gap-filled estimate…";
  if (!ix || ix.status === "unavailable" || !ix.date) return `The gap-filled estimate is unavailable${ix?.reason ? `: ${ix.reason}` : ""}.`;
  return `Gap-filled estimate (NOAA DINEOF), ${day(ix.date)}${ix.status === "stale" ? `, stale: ${ix.age_days} days old, no newer day published` : ""}. Reconstructed where satellites saw nothing, not observed.`;
}

/** Is the estimate offered at all (a day with grids)? A stale one is offered, said stale. */
export const chlEstimateUsable = (ix: ChlEstimateIndex | null) => !!ix && ix.status !== "unavailable" && !!ix.grids && !!ix.date;
export const wcofsUsable = (ix: WcofsIndex | null) => !!ix && ix.status !== "unavailable" && !!ix.cycle && ix.steps.length > 0;

// ---------------------------------------------------------------- one shared, true time axis (lead's review)

const ymd = (iso: string) => pacificYmd(iso);

/**
 * The rail's day axis widened to hold everything shown at its true time: the published frames' axis, the latest WCOFS
 * valid time (forecasts run days past today) and the estimate's day. Nothing is stretched to fill a row.
 */
export function extendAxis(axis: [string, string], times: (string | null | undefined)[], days: (string | null | undefined)[] = []): [string, string] {
  let [a, b] = axis;
  for (const t of times) if (t) {
    const d = ymd(t);
    if (d > b) b = d;
    if (d < a) a = d;
  }
  for (const d of days) if (d) {
    if (d > b) b = d;
    if (d < a) a = d;
  }
  return [a, b];
}

/** The WCOFS steps' own span as an axis (when the shared axis is too narrow for 3-hour steps). */
export function stepsAxis(steps: Pick<WcofsStep, "valid_time">[]): [string, string] | null {
  if (!steps.length) return null;
  const t = steps.map((s) => s.valid_time).sort();
  return [ymd(t[0]), ymd(t[t.length - 1])];
}

/**
 * The step to open on: the one nearest to now that the route actually serves (`probe`), tried nearest first; null when
 * none answers. Live every listed step exists, so the first probe answers; a fixture serves only a few.
 */
export async function pickServedStep<T extends Pick<WcofsStep, "id" | "valid_time">>(steps: T[], nowIso: string, probe: (s: T) => Promise<boolean>): Promise<T | null> {
  const now = Date.parse(nowIso);
  const order = [...steps].sort((a, b) => Math.abs(Date.parse(a.valid_time) - now) - Math.abs(Date.parse(b.valid_time) - now));
  for (const s of order) if (await probe(s)) return s;
  return null;
}

/** The forecast steps that play, in time order, without those known not to be served. */
export const playableSteps = <T extends Pick<WcofsStep, "id" | "valid_time" | "kind">>(steps: T[], failed: Set<string>) => steps.filter((s) => s.kind === "forecast" && !failed.has(s.id)).sort((a, b) => a.valid_time.localeCompare(b.valid_time));
