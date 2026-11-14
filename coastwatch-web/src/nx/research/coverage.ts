/**
 * Research coverage (v15.1), DOM-free: what the shore stations around a place measure, how often and how recently,
 * per variable. The audit behind it is docs/coastwatch/v15.1/research-coverage.md.
 *
 * Truth rules kept here:
 * - Every date and count is of ONE variable at ONE station, never "the station's newest visit".
 * - A station that never published a variable "does not collect it" (not "zero", not "no bloom").
 * - A distant station never stands for the place: beyond NEAR_KM the words say so, with its distance and direction.
 * - Old samples are valid records of their day and say nothing about now ("old" is a state, not an error).
 * - Sparse cadence is its own state: a station sampled every few weeks is not a weekly record with gaps.
 * - C-HARM is a model and is never part of a station's coverage; its words are separate (`modelCoverage`).
 */
import type { ObsStation } from "@/generated/observations";
import { distanceKm } from "@/lib/grid";
import type { ObsVariableId } from "@/lib/observations";
import { DAY, NEAR_KM, VAR_ORDER } from "./model";

// ---------------------------------------------------------------- one variable at one station

/**
 * The state of one variable at one station, as the page says it:
 *   recent        a value within RECENT_DAYS, sampled at a regular cadence
 *   sparse        values in the last year, but further apart than SPARSE_DAYS (or fewer than SPARSE_MIN a year)
 *   old           a valid record whose newest value is older than RECENT_DAYS (it says nothing about now)
 *   notCollected  the station has never published this variable (the source does not sample it there)
 *   unavailable   the station's last request failed: the source could not be read
 */
export type CoverageState = "recent" | "sparse" | "old" | "notCollected" | "unavailable";

export const RECENT_DAYS = 45;
/** A median gap between samples (last 365 days) wider than this is "sparse" (weekly programmes run 7 days). */
export const SPARSE_DAYS = 21;
/** Fewer values than this in the last 365 days is "sparse" whatever the gaps (a burst is not a record). */
export const SPARSE_MIN = 12;

export type VarCoverage = {
  v: ObsVariableId;
  state: CoverageState;
  /** values (quantified or reported 0) in the whole record; QC-rejected and unmeasured visits are not values */
  n: number;
  /** the oldest and newest sample with a value (ISO time as published), or null */
  first: string | null;
  last: string | null;
  /** whole days from the newest value to now, or null */
  ageDays: number | null;
  /** values in the 365 days to now */
  nYear: number;
  /** median days between values in the 365 days to now, or null with fewer than two */
  cadenceDays: number | null;
};

type St = Pick<ObsStation, "sample_times" | "series" | "status">;

/** The times (ms) of a variable's samples that carry a value, oldest first. */
export function valueTimes(st: Pick<ObsStation, "sample_times" | "series">, v: ObsVariableId): number[] {
  const s = st.series.find((x) => x.variable === v);
  if (!s) return [];
  const out: number[] = [];
  st.sample_times.forEach((time, i) => {
    const raw = s.values[i];
    if (raw == null || s.qualifiers?.[String(i)] === "rejected_negative") return;
    const t = Date.parse(time);
    if (Number.isFinite(t)) out.push(t);
  });
  return out.sort((a, b) => a - b);
}

const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

export function varCoverage(st: St, v: ObsVariableId, now: number): VarCoverage {
  const ts = valueTimes(st, v);
  const year = ts.filter((t) => t > now - 365 * DAY && t <= now + DAY);
  const gaps = year.slice(1).map((t, i) => (t - year[i]) / DAY);
  const cadenceDays = gaps.length ? median(gaps) : null;
  const last = ts.at(-1) ?? null;
  const ageDays = last == null ? null : Math.max(0, Math.floor((now - last) / DAY));
  const times = (t: number | null) => (t == null ? null : st.sample_times.find((x) => Date.parse(x) === t) ?? new Date(t).toISOString());
  let state: CoverageState;
  if (st.status === "failed" && !ts.length) state = "unavailable";
  else if (!ts.length) state = "notCollected";
  else if (ageDays! > RECENT_DAYS) state = "old";
  else if (year.length < SPARSE_MIN || (cadenceDays != null && cadenceDays > SPARSE_DAYS)) state = "sparse";
  else state = "recent";
  return { v, state, n: ts.length, first: times(ts[0] ?? null), last: times(last), ageDays, nYear: year.length, cadenceDays };
}

// ---------------------------------------------------------------- stations around a place

export type Located = Pick<ObsStation, "station_id" | "name" | "sample_times" | "series" | "status"> & { lat: number; lon: number };

const POINTS8 = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"];

/** The direction from one point to another in words ("south"), or "here" within 1 km. */
export function directionWords(lat1: number, lon1: number, lat2: number, lon2: number): string {
  if (distanceKm(lat1, lon1, lat2, lon2) < 1) return "here";
  const dy = lat2 - lat1;
  const dx = (lon2 - lon1) * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  const deg = ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360;
  return POINTS8[Math.round(deg / 45) % 8];
}

export type NearbyStation<S extends Located = Located> = {
  station: S;
  km: number;
  dir: string;
  near: boolean;
  /** every variable the programme measures somewhere, as this station holds it (notCollected included) */
  vars: VarCoverage[];
  /** the variables it has ever measured */
  measures: ObsVariableId[];
};

/** The variables at least one station has ever published (dDA is in the schema; no station publishes it). */
export function programmeVariables(stations: St[]): ObsVariableId[] {
  return VAR_ORDER.filter((v) => stations.some((s) => valueTimes(s, v).length > 0));
}

export function nearbyOf<S extends Located>(station: S, from: { lat: number; lon: number }, vars: ObsVariableId[], now: number): NearbyStation<S> {
  const km = distanceKm(from.lat, from.lon, station.lat, station.lon);
  const cov = vars.map((v) => varCoverage(station, v, now));
  return { station, km, dir: directionWords(from.lat, from.lon, station.lat, station.lon), near: km <= NEAR_KM, vars: cov, measures: cov.filter((c) => c.n > 0).map((c) => c.v) };
}

/**
 * The stations to list around a place: every station within NEAR_KM, nearest first; when fewer than `min` are,
 * the nearest `min` (each labelled with its distance; `near: false` beyond NEAR_KM).
 */
export function stationsAround<S extends Located>(stations: S[], from: { lat: number; lon: number }, now: number, min = 3, vars: ObsVariableId[] = programmeVariables(stations)): NearbyStation<S>[] {
  const all = stations.map((s) => nearbyOf(s, from, vars, now)).sort((a, b) => a.km - b.km);
  const near = all.filter((x) => x.near);
  return near.length >= min ? near : all.slice(0, min);
}

// ---------------------------------------------------------------- per variable: the nearest station that measures it

export type VariableReach<S extends Located = Located> = {
  v: ObsVariableId;
  /** the nearest station that has ever measured the variable, or null when none does */
  nearest: NearbyStation<S> | null;
  cov: VarCoverage | null;
  /** the nearest station with a RECENT value of it, when that is a different (farther) station */
  recent: NearbyStation<S> | null;
  /** stations within NEAR_KM that measure it */
  nNear: number;
};

export function reachOf<S extends Located>(stations: S[], from: { lat: number; lon: number }, v: ObsVariableId, now: number): VariableReach<S> {
  const rows = stations
    .map((s) => nearbyOf(s, from, [v], now))
    .filter((x) => x.vars[0].n > 0)
    .sort((a, b) => a.km - b.km);
  const nearest = rows[0] ?? null;
  const rec = rows.find((x) => x.vars[0].state === "recent" || x.vars[0].state === "sparse") ?? null;
  return { v, nearest, cov: nearest?.vars[0] ?? null, recent: rec && rec !== nearest && (nearest?.vars[0].state === "old") ? rec : null, nNear: rows.filter((x) => x.near).length };
}

/** One row per variable the programme measures: the place's coverage ladder. */
export function placeCoverage<S extends Located>(stations: S[], from: { lat: number; lon: number }, now: number): VariableReach<S>[] {
  return programmeVariables(stations).map((v) => reachOf(stations, from, v, now));
}

// ---------------------------------------------------------------- words

const kmWords = (km: number) => (km < 1 ? "under 1 km" : km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`);

/** "77 km south", "2.2 km northwest", "at the port". */
export function whereWords(n: Pick<NearbyStation, "km" | "dir">): string {
  return n.dir === "here" ? "at the port" : `${kmWords(n.km)} ${n.dir}`;
}

export const STATE_WORDS: Record<CoverageState, string> = {
  recent: "Regular samples",
  sparse: "Sparse cadence",
  old: "Old but valid",
  notCollected: "Not collected",
  unavailable: "Source unavailable",
};

/** A cadence in words: "weekly", "every 19 days", or null. */
export function cadenceWords(days: number | null): string | null {
  if (days == null) return null;
  if (days <= 8) return "weekly";
  if (days <= 16) return "every 2 weeks";
  return `every ${Math.round(days)} days`;
}

/**
 * The sentence for a variable around a place, never implying a distant station describes it:
 *   near:  "Trinidad Pier, 0.6 km south, measures pDA."
 *   far:   "The nearest station measuring pDA is Trinidad Pier, 77 km south; nothing nearer samples it."
 *   none:  "No CalHABMAP station measures pDA."
 */
export function reachWords(r: Pick<VariableReach, "nearest">, short: string, placeName: string | null): string {
  const n = r.nearest;
  if (!n) return `No CalHABMAP station measures ${short}.`;
  if (n.near) return `${n.station.name}, ${whereWords(n)}, measures ${short}.`;
  return `The nearest station measuring ${short} is ${n.station.name}, ${whereWords(n)}; nothing nearer${placeName ? ` ${placeName}` : ""} samples it.`;
}

// ---------------------------------------------------------------- the model, kept apart

export type ModelCoverage = {
  /** days of C-HARM history kept at the station (the pipeline keeps CHARM_HISTORY_DAYS = 180), or 0 */
  days: number;
  first: string | null;
  last: string | null;
  radiusKm: number;
};

/** What C-HARM history the station carries: a model probability of a bloom, never a measured toxin. */
export function modelCoverage(st: Pick<ObsStation, "charm">): ModelCoverage {
  const h = st.charm?.history ?? {};
  const dates = Object.values(h)
    .flatMap((xs) => (xs ?? []).filter((x) => x.value != null).map((x) => x.date))
    .sort();
  return { days: new Set(dates).size, first: dates[0] ?? null, last: dates.at(-1) ?? null, radiusKm: st.charm?.radius_km ?? 15 };
}

// ---------------------------------------------------------------- the station the page opens on, by usefulness

/** The variables that lead a station's choice: the toxin and the cells that make it (not temperature or chlorophyll). */
export const HEADLINE: ObsVariableId[] = ["pDA", "tDA", "pn_seriata", "pn_delicatissima"];
/** "Recent" for the default station: a sample within a year. */
export const USEFUL_DAYS = 365;
/** Beyond 60 km the band reaches this far past the nearest station (a farther, newer record can still be chosen). */
export const BAND_KM = 60;
/** Recent stations within this much of the nearest recent one count as equally near: the newest of them is chosen. */
export const TIE_KM = 10;
/** …and a farther one of them wins only when its newest sample is this much newer (a week apart is the same cadence). */
export const TIE_DAYS = 30;

export type UsefulPick<S extends Located = Located> = {
  station: S;
  km: number;
  dir: string;
  /** its newest sample of the leading variables (ISO time) */
  last: string;
  /** the nearer station that measures them and was passed over (its record older), or null */
  passed: { station: S; km: number; dir: string; last: string } | null;
};

/**
 * The station a place opens on: not simply the nearest, but the nearest with a recent sample of what matters (the
 * asked variable, else pDA, tDA and Pseudo-nitzschia): Moss Landing opens on Santa Cruz Wharf (pDA Oct 28, 25 km)
 * rather than Monterey Wharf (24 km, pDA last 2022). Among stations with a sample in the last USEFUL_DAYS inside the
 * band (60 km, or the nearest + 60 km where none is that close), the nearest wins; ones within TIE_KM of it are told
 * apart only when the farther one's sample is clearly newer (TIE_DAYS). With no recent sample in the band, the newest record in it. The page names the nearer
 * station passed over, so both stay one press away.
 */
export function usefulStation<S extends Located>(stations: S[], from: { lat: number; lon: number }, asked: ObsVariableId | null, now: number): UsefulPick<S> | null {
  const score = (vars: ObsVariableId[]) =>
    stations
      .map((s) => {
        const t = Math.max(-Infinity, ...vars.map((v) => valueTimes(s, v).at(-1) ?? -Infinity));
        return { s, km: distanceKm(from.lat, from.lon, s.lat, s.lon), t };
      })
      .filter((r) => Number.isFinite(r.t))
      .sort((a, b) => a.km - b.km);
  let rows = score(asked ? [asked] : HEADLINE);
  if (!rows.length && !asked) rows = score(VAR_ORDER);
  if (!rows.length) return null;
  const reach = rows[0].km <= NEAR_KM ? NEAR_KM : rows[0].km + BAND_KM;
  const band = rows.filter((r) => r.km <= reach);
  const recent = band.filter((r) => now - r.t <= USEFUL_DAYS * DAY);
  const newest = (xs: typeof rows) => [...xs].sort((a, b) => b.t - a.t || a.km - b.km)[0];
  // equally near (within TIE_KM): the nearest, unless a farther one is clearly newer (by more than TIE_DAYS)
  const tied = (xs: typeof rows) => xs.filter((r) => r.km <= xs[0].km + TIE_KM).reduce((best, r) => (r.t > best.t + TIE_DAYS * DAY ? r : best));
  const pick = recent.length ? tied(recent) : newest(band);
  const iso = (r: (typeof rows)[number]) => r.s.sample_times.find((x) => Date.parse(x) === r.t) ?? new Date(r.t).toISOString();
  const near = rows[0];
  const passedOver = near !== pick && near.km < pick.km - 0.05 && near.t < pick.t;
  return {
    station: pick.s,
    km: pick.km,
    dir: directionWords(from.lat, from.lon, pick.s.lat, pick.s.lon),
    last: iso(pick),
    passed: passedOver ? { station: near.s, km: near.km, dir: directionWords(from.lat, from.lon, near.s.lat, near.s.lon), last: iso(near) } : null,
  };
}
