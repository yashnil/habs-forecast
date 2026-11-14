/**
 * The client contract for ocean model and estimate layers read live at request time (nothing is published to a
 * data branch). Browser-safe: types, constants and pure label helpers only.
 *
 * Families, kept visibly distinct from measurements (owner's direction D):
 * - "currents.predicted": NOAA WCOFS predicted surface currents (model guidance; HF radar is the measured layer).
 * - "sst.model":          NOAA WCOFS model sea surface temperature (depth 0 m on the model's grid, not satellite skin).
 * - "chlorophyll.estimate": a gap-filled Level-4 chlorophyll estimate (NOAA DINEOF today), reconstructed where
 *   satellites saw nothing; never an observation.
 *
 * Routes:
 *   GET /explore/model/wcofs                                  → WcofsIndex (JSON)
 *   GET /explore/model/wcofs/grid?cycle=&step=&var=u|v|sst|mask → one grid, uint16le+gzip (lib/grid loadGrid reads it)
 *   GET /explore/model/chlorophyll                            → ChlEstimateIndex (JSON)
 *   GET /explore/model/chlorophyll/grid?date=&var=chl|mask    → one grid, uint16le+gzip
 * A grid route that cannot reach upstream answers 503 with an `OceanUnavailable` JSON body.
 */
import { formatDateTimePT } from "@/lib/time";

export type OceanFamily = "currents.predicted" | "sst.model" | "chlorophyll.estimate";

/** The three time tracks of Explore (direction E): a nowcast is an analysis/estimate, not an observation. */
export type OceanTrack = "observations" | "estimates" | "forecasts";

/**
 * "current": the product's newest data is what we serve and its valid window reaches now.
 * "expired": the newest data we could find ends before now (show the last run, its valid interval and why).
 * "stale":   an estimate product whose newest day is older than its expected latency (for chlorophyll.estimate).
 * "unavailable": upstream could not be read and nothing was cached; say so, never substitute.
 */
export type OceanStatus = "current" | "expired" | "stale" | "unavailable";

/**
 * Grid geometry, compatible with the published `ValueGrid` (src/generated/schema.ts) so `loadGrid`, `cellAt`,
 * `valueOf` and the FieldLayer/FlowLayer read it unchanged: row-major, row 0 at `lat_first`, steps of either sign;
 * code → value is `code * scale_factor + add_offset` (then 10^x when transform is "log10"); `nodata` is no value.
 */
export type OceanGrid = {
  width: number;
  height: number;
  lat_first: number;
  lat_step: number;
  lon_first: number;
  lon_step: number;
  scale_factor: number;
  add_offset: number;
  nodata: number;
  transform: "none" | "log10";
  max_quantization_error: number;
  encoding: "uint16le+gzip";
  /** The grid route for this variable and time (relative to the site). */
  url: string;
};

/** Codes of every `mask` grid. */
export const MASK_LAND = 0;
export const MASK_WATER = 1;
/** Water, but within the family's near-shore distance of land: draw faded or hatched as "coarse near shore". */
export const MASK_NEAR_SHORE = 2;

export type OceanSource = {
  /** e.g. "NOAA CO-OPS West Coast Operational Forecast System (WCOFS)". */
  name: string;
  /** The upstream product or dataset id. */
  product: string;
  /** The upstream catalog or dataset page a person can open. */
  url: string;
  /** About the product (method, skill). */
  info_url: string;
  licence: string;
  developmental: boolean;
};

export type OceanUnavailable = { status: "unavailable"; reason: string; source: OceanSource };

export type WcofsVar = "u" | "v" | "sst" | "mask";

export type WcofsStep = {
  /** Upstream step id: "n003"…"n024" (nowcast) or "f003"…"f072" (forecast). */
  id: string;
  kind: "nowcast" | "forecast";
  track: Extract<OceanTrack, "estimates" | "forecasts">;
  /** Hours from the cycle time (negative for nowcast steps before it). */
  lead_h: number;
  valid_time: string;
  grids: Record<WcofsVar, OceanGrid>;
};

export type WcofsFamily = {
  family: Extract<OceanFamily, "currents.predicted" | "sst.model">;
  /** Display units of decoded values. */
  units: "m/s" | "°C";
  /** Grids to read: ["u", "v"] for currents, ["sst"] for temperature. */
  vars: WcofsVar[];
  /** The label for the default step; use `wcofsLabel` for any other step. */
  label: string;
};

export type WcofsIndex = {
  status: Exclude<OceanStatus, "stale">;
  /** Why it is expired or unavailable, or why an older cycle is served. */
  reason: string | null;
  source: OceanSource;
  /** "2026111403". Null only when unavailable. */
  cycle: { id: string; issued: string; steps_available: number; steps_expected: number } | null;
  /** First and last valid time of the served steps. */
  valid: { start: string; end: string } | null;
  /** True when a newer cycle could not be checked and the last good one is served. */
  fallback: boolean;
  /** The step to show first: the newest step valid at or before now, else the first. */
  default_step: string | null;
  steps: WcofsStep[];
  families: { "currents.predicted": WcofsFamily; "sst.model": WcofsFamily };
  near_shore_km: number;
  caveats: string[];
  generated_at: string;
};

export type ChlVar = "chl" | "mask";

export type ChlEstimateIndex = {
  status: Exclude<OceanStatus, "expired">;
  reason: string | null;
  family: "chlorophyll.estimate";
  track: "estimates";
  source: OceanSource;
  /** The estimate's calendar date (UTC day of the upstream daily product). Null only when unavailable. */
  date: string | null;
  /** Whole days between `date` and today (UTC). */
  age_days: number | null;
  units: "mg/m³";
  label: string | null;
  grids: Record<ChlVar, OceanGrid> | null;
  /** What the product carries: DINEOF has neither per-pixel uncertainty nor an observed/filled flag. */
  per_pixel_uncertainty: boolean;
  filled_flag: boolean;
  near_shore_km: number;
  caveats: string[];
  generated_at: string;
};

const UTC_DAY = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

/** "Nov 14 03z" for cycle "2026111403". */
export function cycleName(cycleId: string): string {
  const d = new Date(Date.UTC(+cycleId.slice(0, 4), +cycleId.slice(4, 6) - 1, +cycleId.slice(6, 8)));
  return `${UTC_DAY.format(d)} ${cycleId.slice(8, 10)}z`;
}

/**
 * The WCOFS label for a step: "NOAA WCOFS model guidance (developmental) · cycle Nov 14 03z · valid Nov 14, 7:00 PM
 * PST · a model forecast, not a measurement; not a bloom or toxin prediction" (nowcast steps say "nowcast").
 */
export function wcofsLabel(cycleId: string, step: Pick<WcofsStep, "kind" | "valid_time">): string {
  const what = step.kind === "nowcast" ? "a model nowcast" : "a model forecast";
  return `NOAA WCOFS model guidance (developmental) · cycle ${cycleName(cycleId)} · valid ${formatDateTimePT(step.valid_time)} · ${what}, not a measurement; not a bloom or toxin prediction`;
}

/** "Gap-filled estimate (NOAA DINEOF) · Nov 2 · reconstructed where satellites saw nothing, not observed · no per-pixel uncertainty". */
export function chlEstimateLabel(date: string, sourceShort = "NOAA DINEOF", perPixelUncertainty = false): string {
  const day = UTC_DAY.format(new Date(`${date}T12:00:00Z`));
  return `Gap-filled estimate (${sourceShort}) · ${day} · reconstructed where satellites saw nothing, not observed${perPixelUncertainty ? "" : " · no per-pixel uncertainty"}`;
}
