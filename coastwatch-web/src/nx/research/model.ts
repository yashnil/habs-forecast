/**
 * Research, DOM-free: what the page reads from its address, which station and window it opens on, and the
 * record as strips on one time axis. Truth rules (docs/coastwatch/v12/research/E-science-and-perf.md §2):
 * T7 a reported 0 has its own mark, never 0 on a log axis; T19/T20 a gap is a gap and a visit without a
 * measurement is never a zero; T2 the model is hollow and its own strip; T11 C-HARM is "median of cells
 * within 15 km", never "at the station"; T15 precision follows error.
 */
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { distanceKm } from "@/lib/grid";
import type { ObsVariableId } from "@/lib/observations";

export const DAY = 86_400_000;
export const DEFAULT_STATION = "HABs-SantaCruzWharf";
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;

// ---------------------------------------------------------------- variables

/** Strips top to bottom: toxin, then the cells that make it, then algae and water. */
export const VAR_ORDER: ObsVariableId[] = ["pDA", "dDA", "tDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"];

/**
 * A `var` (or v11 `variable`) from a link: a station quantity, a C-HARM variable read as its station
 * counterpart, or an Explore layer read as the nearest thing a shore station measures.
 */
export function mapVar(v: string | null | undefined): ObsVariableId | null {
  if (!v) return null;
  const k = v.trim();
  if ((VAR_ORDER as string[]).includes(k)) return k as ObsVariableId;
  const l = k.toLowerCase();
  if (l === "pda" || l === "particulate_domoic" || l === "cellular_domoic" || l === "domoic") return "pDA";
  if (l === "pseudo_nitzschia" || l === "pn" || l === "seriata") return "pn_seriata";
  if (l === "delicatissima") return "pn_delicatissima";
  if (/^(chl|chlorophyll|chl_extracted|olci300|viirs750|multi|multisensor|chl_estimate|outlook)/.test(l)) return "chl_extracted";
  if (l === "temp" || l === "sst" || l === "temperature") return "temp";
  return null;
}

/** The C-HARM quantity that belongs beside a station quantity (v11's linked model panel). */
export function modelVarFor(v: ObsVariableId): "particulate_domoic" | "pseudo_nitzschia" {
  return v === "pn_seriata" || v === "pn_delicatissima" ? "pseudo_nitzschia" : "particulate_domoic";
}

// ---------------------------------------------------------------- the address

export type RangeId = "90d" | "1y" | "3y" | "all";
export const RANGES: { id: RangeId; label: string; days: number }[] = [
  { id: "90d", label: "90 d", days: 90 },
  { id: "1y", label: "1 y", days: 365 },
  { id: "3y", label: "3 y", days: 3 * 365 },
  { id: "all", label: "All", days: Infinity },
];

/** A link's day as the chart's instant (noon UTC, as sample days are read), or null. */
export const parseDay = (v: string | null | undefined) => (v && DAY_RE.test(v) && Number.isFinite(Date.parse(`${v}T12:00:00Z`)) ? Date.parse(`${v}T12:00:00Z`) : null);
export const dayKey = (t: number) => new Date(t).toISOString().slice(0, 10);

export type ResearchQuery = {
  station: string | null;
  place: string | null;
  v: ObsVariableId | null;
  /** the variable as asked, when it could not be read (said on the page, never silently swapped) */
  askedVar: string | null;
  from: number | null;
  to: number | null;
  range: RangeId | null;
  day: number | null;
  /** v11's `from=/path`: where Back goes */
  back: string | null;
  /** v11 section hashes (#r-provenance …) name a margin row to open */
  section: Section | null;
};

export type Section = "station" | "window" | "samples" | "checks" | "source" | "model" | "export" | "grid";
const HASH_SECTION: Record<string, Section> = { "r-stats": "window", "r-tools": "grid", "r-provenance": "checks", "r-lineage": "source", "r-exports": "export", "r-record": "samples" };

type Q = { get(k: string): string | null };

/**
 * v12's parameters are `station`, `from`, `to`, `var` (Explore links with exactly these). v11's still resolve:
 * `variable` → var, `start`/`end` → from/to, `range` (90d, 1y, 3y, all), `day` (a pinned day), `place`
 * (the nearest station), `from=/path` (Back), and `#r-…` section hashes.
 */
export function parseQuery(q: Q, hash = ""): ResearchQuery {
  const fromRaw = q.get("from");
  const path = (v: string | null) => (v && v.startsWith("/") && !v.startsWith("//") ? v : null);
  const back = path(q.get("back")) ?? path(fromRaw);
  const varRaw = q.get("var") ?? q.get("variable");
  const range = q.get("range");
  return {
    station: q.get("station"),
    place: q.get("place"),
    v: mapVar(varRaw),
    askedVar: varRaw && !mapVar(varRaw) ? varRaw : null,
    from: parseDay(fromRaw) ?? parseDay(q.get("start")),
    to: parseDay(q.get("to")) ?? parseDay(q.get("end")),
    range: RANGES.some((r) => r.id === range) ? (range as RangeId) : null,
    day: parseDay(q.get("day")),
    back,
    section: HASH_SECTION[hash.replace(/^#/, "")] ?? null,
  };
}

/** The canonical v12 address for a view (`place` and Back are kept as context). */
export function queryFor(s: { station: string; v: ObsVariableId; win: [number, number]; place?: string | null; back?: string | null; day?: number | null }): string {
  const q = new URLSearchParams();
  if (s.place) q.set("place", s.place);
  q.set("station", s.station);
  q.set("var", s.v);
  q.set("from", dayKey(s.win[0]));
  q.set("to", dayKey(s.win[1]));
  if (s.day != null) q.set("day", dayKey(s.day));
  if (s.back) q.set("back", s.back);
  return q.toString().replace(/%2C/g, ",");
}

// ---------------------------------------------------------------- the station

export type Located = ObsStation & { lat: number; lon: number };
export const located = (ds: Pick<ObservationDataset, "stations">): Located[] => ds.stations.filter((s): s is Located => s.lat != null && s.lon != null);

/** North to south, as the coast is read. */
export const northToSouth = <S extends { lat: number }>(xs: S[]) => [...xs].sort((a, b) => b.lat - a.lat);

/** A station named in a link: its id, its id without `HABs-`, or its name. */
export function findStation<S extends { station_id: string; name: string }>(stations: S[], id: string | null): S | null {
  if (!id) return null;
  const l = id.toLowerCase();
  return stations.find((s) => s.station_id === id) ?? stations.find((s) => s.station_id.toLowerCase() === `habs-${l}` || s.name.toLowerCase() === l) ?? null;
}

/** The station nearest a place: one whose published nearest harbour is that port first, else the nearest by distance (v11). */
export function nearestStation<S extends { station_id: string; lat: number; lon: number; nearest_port_code?: number | null; nearest_port_km?: number | null }>(stations: S[], lat: number, lon: number, key: string | null): { station: S; km: number } | null {
  const pc = key?.startsWith("port-") ? Number(key.slice(5)) : null;
  const own = pc == null ? [] : stations.filter((s) => s.nearest_port_code === pc).sort((a, b) => (a.nearest_port_km ?? 99) - (b.nearest_port_km ?? 99));
  let best: { station: S; km: number } | null = null;
  for (const s of own.length ? own.slice(0, 1) : stations) {
    const km = distanceKm(lat, lon, s.lat, s.lon);
    if (!best || km < best.km) best = { station: s, km };
  }
  return best;
}

/** Beyond this a "nearest station" says it is not near (research C §7 problem 10). */
export const NEAR_KM = 60;

export type Arrival = { kind: "place"; name: string; km: number; near: boolean } | null;

export type Choice = { station: Located; arrival: Arrival };

/**
 * Which station the page opens on: the link's station; else the station nearest the link's place (and the
 * page says so, with the distance); else the one nearest the reader's port; else Santa Cruz Wharf.
 */
export function chooseStation(stations: Located[], q: Pick<ResearchQuery, "station">, place: { key: string; name: string; lat: number; lon: number } | null, port: { key: string; lat: number; lon: number } | null): Choice {
  const asked = findStation(stations, q.station);
  if (asked) {
    const km = place ? distanceKm(place.lat, place.lon, asked.lat, asked.lon) : null;
    const near = place ? nearestStation(stations, place.lat, place.lon, place.key) : null;
    return { station: asked, arrival: place && km != null && near?.station.station_id === asked.station_id ? { kind: "place", name: place.name, km, near: km <= NEAR_KM } : null };
  }
  if (place) {
    const n = nearestStation(stations, place.lat, place.lon, place.key);
    if (n) return { station: n.station, arrival: { kind: "place", name: place.name, km: n.km, near: n.km <= NEAR_KM } };
  }
  if (port) {
    const n = nearestStation(stations, port.lat, port.lon, port.key);
    if (n) return { station: n.station, arrival: null };
  }
  return { station: findStation(stations, DEFAULT_STATION) ?? stations[0], arrival: null };
}

// ---------------------------------------------------------------- the record

export type Q11 = "reported_zero" | "rejected_negative" | "flag_high";

/**
 * One sample of one quantity. `cls`: value (on the axis), zero (a reported 0: its own lane, T7), rejected
 * (QC rejected: no value, a cross in the lane), flagged (a value kept as published but flagged high by QC).
 */
export type RecPoint = { t: number; time: string; i: number; v: number | null; cls: "value" | "zero" | "rejected"; flagged: boolean; /** model days: cells with a value */ n?: number };

export type StripKind = "measured" | "model";

export type Strip = {
  id: string;
  v: ObsVariableId | null;
  kind: StripKind;
  label: string;
  /** words after the label: unit and what the number is of */
  unit: string;
  unitShort: string;
  scale: "log" | "linear";
  domain: [number, number];
  points: RecPoint[];
  /** a reference line on its own scale (C-HARM's own thresholds; T6: never a seafood level on water) */
  threshold?: { value: number; label: string };
  /** model days with fewer than 60 % of the usual cells (drawn open, not joined) */
  partial?: Set<number>;
  /** the model's history starts here; before it nothing was kept (hatched) */
  historyStart?: number | null;
  /** samples further apart than this are not joined (a gap is a gap) */
  joinDays: number;
};

export const UNIT: Record<ObsVariableId, string> = { pDA: "ng/mL", dDA: "ng/mL", tDA: "ng/mL", pn_seriata: "cells/L", pn_delicatissima: "cells/L", chl_extracted: "mg/m³", temp: "°C" };

export const LABEL: Record<ObsVariableId, string> = {
  pDA: "Particulate domoic acid",
  dDA: "Dissolved domoic acid",
  tDA: "Total domoic acid",
  pn_seriata: "Pseudo-nitzschia, seriata group",
  pn_delicatissima: "Pseudo-nitzschia, delicatissima group",
  chl_extracted: "Chlorophyll-a, extracted",
  temp: "Water temperature",
};

/** What the unit is of, said once on the strip (T5: chlorophyll is algae, not toxin; T6: seawater, not seafood). */
const UNIT_WORDS: Record<ObsVariableId, string> = {
  pDA: "ng/mL in seawater",
  dDA: "ng/mL in seawater",
  tDA: "ng/mL in seawater",
  pn_seriata: "cells/L, larger cells",
  pn_delicatissima: "cells/L, smaller cells",
  chl_extracted: "mg/m³, algae, not toxin",
  temp: "°C at sampling",
};

/** C-HARM's own thresholds (its `threshold_text`): references on the water scale, not safety limits. */
const REF: Partial<Record<ObsVariableId, { value: number; label: string }>> = {
  pDA: { value: 0.5, label: "C-HARM threshold, 500 ng/L" },
  pn_seriata: { value: 10_000, label: "C-HARM threshold, 10,000 cells/L (all Pseudo-nitzschia)" },
  pn_delicatissima: { value: 10_000, label: "C-HARM threshold, 10,000 cells/L (all Pseudo-nitzschia)" },
};

const LOG_VARS = new Set<ObsVariableId>(["pDA", "dDA", "tDA", "pn_seriata", "pn_delicatissima", "chl_extracted"]);

/** One quantity's samples at a station, classified. Visits that did not measure it are left out (not zero). */
export function pointsOf(st: Pick<ObsStation, "sample_times" | "series">, v: ObsVariableId): RecPoint[] {
  const s = st.series.find((x) => x.variable === v);
  if (!s) return [];
  const log = LOG_VARS.has(v);
  const out: RecPoint[] = [];
  st.sample_times.forEach((time, i) => {
    const raw = s.values[i];
    const q = s.qualifiers?.[String(i)] as Q11 | undefined;
    if (raw == null && q !== "rejected_negative") return;
    const t = Date.parse(time);
    if (!Number.isFinite(t)) return;
    if (q === "rejected_negative") out.push({ t, time, i, v: null, cls: "rejected", flagged: false });
    else if (q === "reported_zero" || (log && raw === 0)) out.push({ t, time, i, v: 0, cls: "zero", flagged: false });
    else out.push({ t, time, i, v: raw as number, cls: "value", flagged: q === "flag_high" });
  });
  return out;
}

/** Whole decades around the positive values; at least two decades so a quiet record never fills the strip. */
export function logDomainOf(values: number[]): [number, number] {
  const pos = values.filter((v) => v > 0 && Number.isFinite(v));
  if (!pos.length) return [0.01, 1];
  let lo = Math.floor(Math.log10(Math.min(...pos)));
  let hi = Math.ceil(Math.log10(Math.max(...pos)));
  while (hi - lo < 2) {
    hi += 1;
    if (hi - lo < 2) lo -= 1;
  }
  return [Number(`1e${lo}`), Number(`1e${hi}`)];
}

/** A linear domain in whole steps of `step` around the values, at least `minSpan` wide. */
export function linearDomainOf(values: number[], step = 2, minSpan = 4): [number, number] {
  const v = values.filter(Number.isFinite);
  if (!v.length) return [0, minSpan];
  let lo = Math.floor(Math.min(...v) / step) * step;
  let hi = Math.ceil(Math.max(...v) / step) * step;
  if (hi - lo < minSpan) {
    const pad = (minSpan - (hi - lo)) / 2;
    lo = Math.floor((lo - pad) / step) * step;
    hi = lo + Math.max(minSpan, Math.ceil((hi + pad - lo) / step) * step);
  }
  return [lo, hi];
}

export type ModelPoint = { date: string; value: number | null; n: number };

/**
 * The station's strips. Axes span the station's whole record and never rescale as the window moves (a quiet
 * year cannot look like a bloom); the two Pseudo-nitzschia groups share one axis. The model strip follows the
 * focused quantity (P(pDA > 500 ng/L) beside the toxin, P(Pseudo-nitzschia > 10,000 cells/L) beside the cells).
 */
export function buildStrips(st: ObsStation, focus: ObsVariableId): Strip[] {
  const vars = VAR_ORDER.filter((v) => pointsOf(st, v).some((p) => p.cls !== "rejected"));
  const pts = new Map(vars.map((v) => [v, pointsOf(st, v)] as const));
  const pnAll = (["pn_seriata", "pn_delicatissima"] as const).flatMap((v) => (pts.get(v) ?? []).filter((p) => p.cls === "value").map((p) => p.v as number));
  const strips: Strip[] = vars.map((v) => {
    const p = pts.get(v)!;
    const vals = p.filter((x) => x.cls === "value").map((x) => x.v as number);
    const scale = LOG_VARS.has(v) ? "log" : "linear";
    const domain = scale === "log" ? logDomainOf(v.startsWith("pn_") ? pnAll : vals) : linearDomainOf(vals);
    const ref = REF[v];
    return {
      id: v,
      v,
      kind: "measured",
      label: LABEL[v],
      unit: UNIT_WORDS[v],
      unitShort: UNIT[v],
      scale,
      domain,
      points: p,
      threshold: ref && ref.value >= domain[0] && ref.value <= domain[1] ? ref : undefined,
      joinDays: 21,
    };
  });
  const mv = modelVarFor(focus);
  const hist = (st.charm?.history?.[mv] ?? []) as ModelPoint[];
  const valued = hist.filter((h) => h.value != null);
  if (valued.length) {
    const ns = valued.map((h) => h.n).sort((a, b) => a - b);
    const usual = ns[Math.floor(ns.length / 2)] ?? 0;
    const points: RecPoint[] = valued.map((h, i) => ({ t: Date.parse(`${h.date}T12:00:00Z`), time: h.date, i, v: (h.value as number) * 100, cls: "value", flagged: false, n: h.n }));
    const partial = new Set(valued.flatMap((h, i) => (usual > 0 && h.n < 0.6 * usual ? [points[i].t] : [])));
    strips.push({
      id: "model",
      v: null,
      kind: "model",
      label: mv === "particulate_domoic" ? "C-HARM P(pDA > 500 ng/L)" : "C-HARM P(Pseudo-nitzschia > 10,000 cells/L)",
      unit: `%, median of model cells within ${st.charm?.radius_km ?? 15} km`,
      unitShort: "%",
      scale: "linear",
      domain: [0, 100],
      points,
      partial,
      historyStart: points[0]?.t ?? null,
      joinDays: 1.5,
    });
  }
  return strips;
}

/** Every sampling visit of the station that measured at least one strip's quantity, in time order. */
export function visitsOf(strips: Strip[]): number[] {
  const ts = new Set<number>();
  for (const s of strips) if (s.kind === "measured") for (const p of s.points) ts.add(p.t);
  return [...ts].sort((a, b) => a - b);
}

/** The whole record's span: the first sample to today (so the record's end shows how old it is). */
export function extentOf(strips: Strip[], now: number): [number, number] {
  const ts = strips.flatMap((s) => s.points.map((p) => p.t));
  if (!ts.length) return [now - 365 * DAY, now];
  return [Math.min(...ts) - 4 * DAY, Math.max(now, Math.max(...ts) + 4 * DAY)];
}

// ---------------------------------------------------------------- the window

export const MIN_WINDOW_DAYS = 30;

/** A window inside the extent, never narrower than the brush allows. */
export function clampWin([a, b]: [number, number], [lo, hi]: [number, number], minDays = MIN_WINDOW_DAYS): [number, number] {
  const w = Math.min(hi - lo, Math.max(minDays * DAY, b - a));
  const start = Math.min(Math.max(a, lo), hi - w);
  return [start, start + w];
}

export function presetWindow(range: RangeId, extent: [number, number]): [number, number] {
  const days = RANGES.find((r) => r.id === range)!.days;
  return Number.isFinite(days) ? clampWin([extent[1] - days * DAY, extent[1]], extent) : extent;
}

/** Which preset a window is, if any (its end at the record's end and its length a preset's). */
export function presetOf(win: [number, number], extent: [number, number]): RangeId | null {
  for (const r of RANGES) {
    const p = presetWindow(r.id, extent);
    if (Math.abs(p[0] - win[0]) < DAY && Math.abs(p[1] - win[1]) < DAY) return r.id;
  }
  return null;
}

/**
 * The window the page opens on. A link's `from`/`to` (clamped; a single day or a short span becomes 90 days
 * around it, with that day pinned); else a preset; else a pinned day's year; else the last three years, or
 * the whole record when the focused quantity was not measured in them (never an empty frame on arrival).
 */
export function openingWindow(o: { extent: [number, number]; from: number | null; to: number | null; range: RangeId | null; day: number | null; focusTimes: number[] }): [number, number] {
  const { extent } = o;
  if (o.from != null || o.to != null) {
    const a = o.from ?? (o.to as number) - 365 * DAY;
    const b = o.to ?? Math.min(extent[1], (o.from as number) + 365 * DAY);
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    if (hi - lo < MIN_WINDOW_DAYS * DAY) {
      const mid = (lo + hi) / 2;
      return clampWin([mid - 45 * DAY, mid + 45 * DAY], extent);
    }
    return clampWin([lo, hi], extent);
  }
  if (o.range) return presetWindow(o.range, extent);
  const three = presetWindow("3y", extent);
  if (o.day != null && (o.day < three[0] || o.day > three[1])) return clampWin([o.day - 182 * DAY, o.day + 183 * DAY], extent);
  if (o.focusTimes.length && !o.focusTimes.some((t) => t >= three[0])) return extent;
  return three;
}

export const inWin = (t: number, [a, b]: [number, number]) => t >= a && t <= b;

// ---------------------------------------------------------------- words for a value (T15: precision follows error)

const sig2 = (v: number) => {
  if (v === 0) return "0";
  if (Math.abs(v) >= 100) return Math.round(v).toLocaleString("en-US");
  const p = Number(v.toPrecision(2));
  const d = Math.max(0, 1 - Math.floor(Math.log10(Math.abs(p))));
  return p.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};

/** A figure at its honest precision: two significant figures measured, one decimal °C, whole percent model. */
export function figure(strip: Pick<Strip, "kind" | "v">, v: number): string {
  if (strip.kind === "model") return `${Math.round(v)}`;
  if (strip.v === "temp") return v.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return sig2(v);
}

/** A sample in words, with its unit: "0.20 ng/mL", "reported 0", "rejected by QC". */
export function valueWords(strip: Pick<Strip, "kind" | "v" | "unitShort">, p: RecPoint, withUnit = true): string {
  if (p.cls === "zero") return "reported 0";
  if (p.cls === "rejected") return "rejected by QC";
  const u = !withUnit ? "" : strip.unitShort === "%" ? " %" : ` ${strip.unitShort}`;
  return `${figure(strip, p.v as number)}${u}${p.flagged ? ", flagged high" : ""}`;
}

// ---------------------------------------------------------------- reading the window

export type WindowStats = { n: number; zeros: number; rejected: number; flagged: number; median: number | null; max: RecPoint | null };

/** What the window holds for one quantity. T8: the highest is always printed with the median, and its date. */
export function windowStats(points: RecPoint[], win: [number, number]): WindowStats {
  const w = points.filter((p) => inWin(p.t, win));
  const vals = w.filter((p) => p.cls === "value");
  const sorted = vals.map((p) => p.v as number).sort((a, b) => a - b);
  const median = sorted.length ? (sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2) : null;
  const max = vals.reduce<RecPoint | null>((m, p) => (m == null || (p.v as number) > (m.v as number) ? p : m), null);
  return { n: vals.length, zeros: w.filter((p) => p.cls === "zero").length, rejected: w.filter((p) => p.cls === "rejected").length, flagged: vals.filter((p) => p.flagged).length, median, max };
}

export type SortKey = "date" | "value";
export type SortDir = "asc" | "desc";

/**
 * The dated log: one row per sample of the focused quantity in the window. Sorted by value, a reported 0 is
 * below every measured value (it is at or below an unpublished limit, not a number) and a rejected sample last.
 */
export function logRows(points: RecPoint[], win: [number, number], key: SortKey = "date", dir: SortDir = "desc"): RecPoint[] {
  const rank = (p: RecPoint) => (p.cls === "value" ? 2 : p.cls === "zero" ? 1 : 0);
  const rows = points.filter((p) => inWin(p.t, win));
  const s = dir === "asc" ? 1 : -1;
  return rows.sort((a, b) => {
    if (key === "value") {
      const r = rank(a) - rank(b);
      if (r) return r * s;
      if (a.cls === "value" && b.cls === "value" && a.v !== b.v) return ((a.v as number) - (b.v as number)) * s;
    }
    return (a.t - b.t) * (key === "date" ? s : -1);
  });
}

/** The crosshair's reading for one strip at a visit: the sample on that day (±1 day), or nothing. */
export function readingAt(strip: Strip, t: number): RecPoint | null {
  let best: RecPoint | null = null;
  for (const p of strip.points) {
    const d = Math.abs(p.t - t);
    if (d <= 1.01 * DAY && (!best || d < Math.abs(best.t - t))) best = p;
  }
  return best;
}

/** Runs of on-axis values to join with a line: broken by a reported 0, a rejected sample, a partial model day, or a gap. */
export function runsOf(strip: Strip, win: [number, number]): RecPoint[][] {
  const out: RecPoint[][] = [];
  let run: RecPoint[] = [];
  let last: number | null = null;
  const pad = strip.joinDays * DAY;
  for (const p of strip.points) {
    if (p.t < win[0] - pad || p.t > win[1] + pad) continue;
    const broken = p.cls !== "value" || strip.partial?.has(p.t) || (strip.scale === "log" && (p.v as number) <= 0);
    if (broken || (last != null && p.t - last > pad)) {
      if (run.length) out.push(run);
      run = [];
    }
    if (!broken) run.push(p);
    last = p.t;
  }
  if (run.length) out.push(run);
  return out;
}

/** Decades of a log domain, and which of them carry a label at this height (labels never collide). */
export function logTicksFor([lo, hi]: [number, number], plotH: number): { v: number; label: boolean }[] {
  const e0 = Math.round(Math.log10(lo));
  const e1 = Math.round(Math.log10(hi));
  const n = e1 - e0;
  const every = Math.max(1, Math.ceil(18 / Math.max(1, plotH / Math.max(1, n))));
  const out: { v: number; label: boolean }[] = [];
  for (let e = e0; e <= e1; e++) out.push({ v: Number(`1e${e}`), label: (e1 - e) % every === 0 });
  return out;
}

export function linearTicksFor([lo, hi]: [number, number], plotH: number): { v: number; label: boolean }[] {
  const span = hi - lo;
  const room = Math.max(1, Math.floor(plotH / 20));
  const steps = [1, 2, 5, 10, 20, 25, 50, 100];
  const step = steps.find((s) => span / s <= room) ?? span;
  const out: { v: number; label: boolean }[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push({ v, label: true });
  return out;
}

/** A tick label: 0.0001 … 100, 1,000, 10K, 1M; no scientific notation. */
export function tickLabel(v: number): string {
  if (v <= 0 || !Number.isInteger(Math.log10(v))) return v < 0 ? `−${Math.abs(v).toLocaleString("en-US")}` : v.toLocaleString("en-US");
  if (v >= 1e6) return `${Math.round(v / 1e6).toLocaleString("en-US")}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3).toLocaleString("en-US")}K`;
  if (v >= 1) return Math.round(v).toLocaleString("en-US");
  return v.toFixed(Math.max(0, -Math.floor(Math.log10(v) + 1e-9)));
}
