import type { ValueGrid } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import { bearingName } from "@/lib/coast";
import { decodeCode, distanceKm, type Cell } from "@/lib/grid";

/**
 * HAB evidence for chosen waters (My Coast's "HAB conditions"), from published data only:
 * - measured: the shore stations' own samples (CalHABMAP): toxin (particulate domoic acid),
 *   Pseudo-nitzschia cells, extracted chlorophyll; each with its date, distance and recent timeline;
 * - modelled: C-HARM cells, in the waters' own cell or, when that cell has no value, the real cells
 *   within NEIGHBOURHOOD_KM (the rule the published port summaries use), never presented as the
 *   waters' own forecast;
 * - observed from space: chlorophyll (algae, not toxin).
 * Nothing is interpolated, extrapolated or combined into a risk score.
 */

export const NEIGHBOURHOOD_KM = 15;
/** a station farther than this is listed as "nearest", never as describing these waters */
export const EVIDENCE_RADIUS_KM = 60;
export const RECENT_DAYS = 30;

const DAY = 86400000;

// ---------------------------------------------------------------- measured at shore stations

export type MeasuredKind = "toxin" | "cells" | "chl";
export const MEASURED_VARIABLES: Record<MeasuredKind, string[]> = {
  toxin: ["pDA"],
  cells: ["pn_seriata", "pn_delicatissima"],
  chl: ["chl_extracted"],
};

export type StationEvidence = {
  kind: MeasuredKind;
  station: StationLite;
  km: number;
  bearing: string;
  /** the newest sample of this kind at the station (value per variable on that date) */
  date: string;
  values: { variable: string; value: number }[];
  ageDays: number;
  /** that sample's timeline at this station (last ~400 days), per variable */
  series: Record<string, [string, number | null][]>;
  /** within EVIDENCE_RADIUS_KM */
  near: boolean;
};

function newestOf(st: StationLite, vars: string[]): { date: string; values: { variable: string; value: number }[] } | null {
  let date: string | null = null;
  for (const v of vars) {
    const l = st.latest.find((x) => x.variable === v && x.value != null);
    if (l && (!date || l.date > date)) date = l.date;
    const r = st.recent?.[v]?.at(-1);
    if (r && r[1] != null && (!date || r[0] > date)) date = r[0];
  }
  if (!date) return null;
  const values: { variable: string; value: number }[] = [];
  for (const v of vars) {
    const fromRecent = st.recent?.[v]?.find((p) => p[0] === date && p[1] != null);
    const fromLatest = st.latest.find((x) => x.variable === v && x.date === date && x.value != null);
    // the published latest value is exact; the timeline keeps 3 significant digits
    const val = fromLatest?.value ?? fromRecent?.[1] ?? null;
    if (val != null) values.push({ variable: v, value: val });
  }
  return values.length ? { date, values } : null;
}

/**
 * The best measured evidence of one kind: the nearest station (within EVIDENCE_RADIUS_KM) with a
 * sample in the last RECENT_DAYS days; otherwise the nearest with any sample of that kind; otherwise
 * the nearest such station anywhere (flagged as not near). Its age always travels with it.
 */
export function measuredEvidence(stations: StationLite[] | null, lat: number, lon: number, kind: MeasuredKind, now: Date): StationEvidence | null {
  const vars = MEASURED_VARIABLES[kind];
  const cands = (stations ?? [])
    .map((st) => {
      const n = newestOf(st, vars);
      if (!n) return null;
      const km = distanceKm(lat, lon, st.lat, st.lon);
      const ageDays = Math.max(0, Math.floor((now.getTime() - Date.parse(`${n.date}T12:00:00Z`)) / DAY));
      const series: Record<string, [string, number | null][]> = {};
      for (const v of vars) if (st.recent?.[v]) series[v] = st.recent[v];
      return { kind, station: st, km, bearing: bearingName(lat, lon, st.lat, st.lon), date: n.date, values: n.values, ageDays, series, near: km <= EVIDENCE_RADIUS_KM } as StationEvidence;
    })
    .filter((x): x is StationEvidence => !!x)
    .sort((a, b) => a.km - b.km);
  const near = cands.filter((c) => c.near);
  return near.find((c) => c.ageDays <= RECENT_DAYS) ?? near[0] ?? cands[0] ?? null;
}

// ---------------------------------------------------------------- modelled (C-HARM cells)

export type Neighbourhood = {
  /** valued cells whose centres lie within NEIGHBOURHOOD_KM */
  n: number;
  median: number | null;
  min: number | null;
  max: number | null;
  /** the nearest valued cell (its centre) and its value */
  nearest: { cell: Cell; value: number; km: number } | null;
  /** the waters' own cell value, when it has one */
  here: number | null;
  /** the cells used, for the map */
  cells: { cell: Cell; value: number }[];
};

/** The nearest valued cell is looked for this far (it is named with its distance, never as the waters' value). */
export const NEAREST_SEARCH_KM = 80;

/** Real cells only: the waters' own cell, the valued cells within NEIGHBOURHOOD_KM, and the nearest valued cell within NEAREST_SEARCH_KM. */
export function neighbourhood(g: ValueGrid, codes: Uint16Array, lat: number, lon: number, radiusKm = NEIGHBOURHOOD_KM, searchKm = NEAREST_SEARCH_KM): Neighbourhood {
  const latFirst = g.lat_first;
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const r0 = Math.round((lat - latFirst) / g.lat_step);
  const c0 = Math.round((lon - lonFirst) / g.lon_step);
  const reach = Math.max(radiusKm, searchKm);
  const dR = Math.ceil(reach / 111 / Math.abs(g.lat_step)) + 1;
  const dC = Math.ceil(reach / (111 * Math.cos((lat * Math.PI) / 180)) / Math.abs(g.lon_step)) + 1;
  const cells: { cell: Cell; value: number }[] = [];
  let nearest: Neighbourhood["nearest"] = null;
  let here: number | null = null;
  for (let r = r0 - dR; r <= r0 + dR; r++) {
    if (r < 0 || r >= g.height) continue;
    for (let c = c0 - dC; c <= c0 + dC; c++) {
      if (c < 0 || c >= g.width) continue;
      const v = decodeCode(g, codes[r * g.width + c]);
      if (v == null) continue;
      const cell: Cell = { row: r, col: c, lat: latFirst + r * g.lat_step, lon: lonFirst + c * g.lon_step };
      const km = distanceKm(lat, lon, cell.lat, cell.lon);
      if (r === r0 && c === c0) here = v;
      if (km <= radiusKm) cells.push({ cell, value: v });
      if (km <= reach && (!nearest || km < nearest.km)) nearest = { cell, value: v, km };
    }
  }
  const vs = cells.map((x) => x.value).sort((a, b) => a - b);
  const median = vs.length ? (vs.length % 2 ? vs[(vs.length - 1) / 2] : (vs[vs.length / 2 - 1] + vs[vs.length / 2]) / 2) : null;
  return { n: vs.length, median, min: vs[0] ?? null, max: vs.at(-1) ?? null, nearest, here, cells };
}

// ---------------------------------------------------------------- words

export const fmtPct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

export function fmtMeasured(variable: string, v: number): string {
  if (variable === "pDA") return v === 0 ? "0 (not quantified)" : `${Number(v.toPrecision(2))} ng/mL`;
  if (variable.startsWith("pn_")) return v === 0 ? "0 cells/L (none counted)" : `${Math.round(v).toLocaleString("en-US")} cells/L`;
  if (variable === "chl_extracted") return `${Number(v.toPrecision(2))} mg/m³`;
  return String(v);
}

export function ageWords(days: number): string {
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return `${days} days ago`;
  if (days < 60) return `${Math.round(days / 7)} weeks ago`;
  if (days < 730) return `${Math.round(days / 30.4)} months ago`;
  return `${Math.floor(days / 365)} years ago`;
}

// ---------------------------------------------------------------- map

/** A circle of `km` around the waters, as a polygon (the model's neighbourhood on the map). */
export function ringPolygon(lat: number, lon: number, km = NEIGHBOURHOOD_KM, n = 72): GeoJSON.Polygon {
  const dLat = km / 111.2;
  const dLon = km / (111.2 * Math.cos((lat * Math.PI) / 180));
  const ring: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    ring.push([lon + dLon * Math.cos(a), lat + dLat * Math.sin(a)]);
  }
  return { type: "Polygon", coordinates: [ring] };
}

/** Stations within `km` of the waters that report HAB measurements, nearest first, with their newest sample's age. */
export function stationsAround(stations: StationLite[] | null, lat: number, lon: number, now: Date, km = 80) {
  const vars = [...MEASURED_VARIABLES.toxin, ...MEASURED_VARIABLES.cells];
  return (stations ?? [])
    .map((st) => {
      const n = newestOf(st, vars);
      if (!n) return null;
      const d = distanceKm(lat, lon, st.lat, st.lon);
      return { station: st, km: d, date: n.date, ageDays: Math.max(0, Math.floor((now.getTime() - Date.parse(`${n.date}T12:00:00Z`)) / DAY)) };
    })
    .filter((x): x is NonNullable<typeof x> => !!x && x.km <= km)
    .sort((a, b) => a.km - b.km);
}
