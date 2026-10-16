import { nearest, type Station } from "@/lib/conditions";
import type { Setting } from "@/lib/harborEvidence";

/**
 * CeNCOOS automated shore stations: sensors at a pier or a seawater intake reporting water
 * temperature, salinity and chlorophyll fluorescence every 10–15 minutes, each value with its
 * QARTOD aggregate flag (1 PASS, 2 NOT_EVALUATED, 3 SUSPECT, 4 FAIL, 9 MISSING). They cover the
 * harbours NOAA's tide-station thermistors miss: Morro Bay, Humboldt Bay, Trinidad, Bodega.
 *
 * Read from https://erddap.cencoos.org/erddap (tabledap CSV), checked live on 2026-10-16
 * (docs/coastwatch/v3/02-nearshore-sources.md). The licence: "These data may be used and
 * redistributed for free but they are not intended for legal use, since they may contain
 * inaccuracies"; publications credit CeNCOOS and NOAA. Only values whose flag is PASS are
 * reported as readings; any other flag is said as a withheld value, never shown as a number.
 *
 * Chlorophyll here is in-water fluorescence (µg/L), a different measurement from the satellite's
 * ocean-colour chlorophyll and from a laboratory's extracted chlorophyll: algae of every kind, not toxin.
 */
export type ShoreStation = Station & { dataset: string; setting: Setting; place: string };

export const SHORE_STATIONS: ShoreStation[] = [
  { id: "edu_calpoly_marine_morro", dataset: "edu_calpoly_marine_morro", name: "Morro Bay shore station", place: "the Cal Poly T-pier in Morro Bay harbour", lat: 35.37085, lon: -120.8589, setting: "sheltered-harbor" },
  { id: "edu_humboldt_humboldt", dataset: "edu_humboldt_humboldt", name: "Humboldt Bay shore station", place: "the Humboldt Bay pier", lat: 40.7775, lon: -124.19652, setting: "sheltered-harbor" },
  { id: "edu_humboldt_tdp", dataset: "edu_humboldt_tdp", name: "Trinidad shore station", place: "the Trinidad Head pier, on the open coast", lat: 41.055, lon: -124.14703, setting: "exposed-pier" },
  { id: "bodega-bay-bml_wts", dataset: "bodega-bay-bml_wts", name: "Bodega Marine Lab shore station", place: "the Bodega Marine Laboratory seawater intake, on the open coast", lat: 38.31652, lon: -123.0709, setting: "exposed-pier" },
  { id: "edu_ucsc_scwharf1", dataset: "edu_ucsc_scwharf1", name: "Santa Cruz Wharf shore station", place: "the Santa Cruz municipal wharf", lat: 36.9603, lon: -122.0203, setting: "exposed-pier" },
];

export const ERDDAP = "https://erddap.cencoos.org/erddap";
/** a shore station farther than this is another body of water (REPRESENTATIVE_KM.water_temp.near) */
export const SHORE_MAX_KM = 6;
/** the newest row must be this recent; older is "not reporting" with the last report's time */
export const SHORE_MAX_AGE_H = 3;

export type QartodFlag = 1 | 2 | 3 | 4 | 9;
export const QARTOD: Record<QartodFlag, string> = { 1: "passed", 2: "not evaluated", 3: "suspect", 4: "failed", 9: "missing" };

export type ShoreVariable = "water_temperature" | "salinity" | "chlorophyll_fluorescence";
const COLUMNS: Record<ShoreVariable, string> = {
  water_temperature: "sea_water_temperature",
  salinity: "sea_water_practical_salinity",
  chlorophyll_fluorescence: "mass_concentration_of_chlorophyll_in_sea_water",
};
export const SHORE_UNITS: Record<ShoreVariable, string> = { water_temperature: "°C", salinity: "PSU", chlorophyll_fluorescence: "µg/L" };

export type ShoreReading = { value: number; at: string };
export type ShoreObs = {
  at: string;
  readings: Partial<Record<ShoreVariable, ShoreReading>>;
  /** values present but not flagged PASS, by variable: the flag's word */
  withheld: Partial<Record<ShoreVariable, string>>;
};

/** "2026-10-16T11:10:00Z" (ERDDAP takes ISO times; seconds dropped so a fixed clock gives one URL) */
const iso = (d: Date) => `${d.toISOString().slice(0, 16)}:00Z`;

/**
 * The station's newest row at or before `now` (ERDDAP orderByMax): one small answer whether the
 * station is reporting or has been silent for months, so the caller can say since when. An empty
 * time window would be a 404, which says nothing about how long the station has been silent.
 */
export function shoreUrl(s: ShoreStation, now: Date): string {
  const cols = ["time", ...Object.values(COLUMNS).flatMap((c) => [c, `${c}_qc_agg`])].join("%2C");
  return `${ERDDAP}/tabledap/${s.dataset}.csv?${cols}&time%3C=${iso(now)}&orderByMax(%22time%22)`;
}
export const shorePage = (s: ShoreStation) => `${ERDDAP}/tabledap/${s.dataset}.html`;

/**
 * The newest row of an ERDDAP CSV (header row, units row, data rows oldest first), per variable the
 * newest value flagged PASS in that row. ERDDAP answers 404 "no matching results" for an empty
 * window; the caller says that as "not reporting".
 */
export function parseShoreCsv(text: string): ShoreObs | null {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 3) return null;
  const head = lines[0].split(",");
  const rows = lines.slice(2).map((l) => l.split(","));
  const col = (name: string) => head.indexOf(name);
  const t = col("time");
  const last = rows.filter((r) => r[t]).sort((a, b) => a[t].localeCompare(b[t])).at(-1);
  if (!last) return null;
  const out: ShoreObs = { at: last[t], readings: {}, withheld: {} };
  for (const [v, c] of Object.entries(COLUMNS) as [ShoreVariable, string][]) {
    const i = col(c);
    const q = col(`${c}_qc_agg`);
    if (i < 0) continue;
    const value = Number(last[i]);
    if (last[i] === "" || last[i] === "NaN" || !Number.isFinite(value)) continue;
    const flag = Number(q >= 0 ? last[q] : NaN) as QartodFlag;
    if (flag === 1) out.readings[v] = { value, at: last[t] };
    else out.withheld[v] = QARTOD[flag] ?? "not flagged";
  }
  return out;
}

export function shoreCandidate(lat: number, lon: number): (ShoreStation & { km: number }) | null {
  return nearest(SHORE_STATIONS, lat, lon, 1, SHORE_MAX_KM)[0] ?? null;
}
