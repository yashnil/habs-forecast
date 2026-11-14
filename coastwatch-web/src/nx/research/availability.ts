/**
 * Research, per-variable truth (v13), DOM-free. A station is visited more often than any one quantity is
 * measured: Monterey Wharf is sampled weekly and has published temperature only since Aug 2025 (no pDA since
 * 2022-08-31, checked at SCCOOS ERDDAP 2026-11-12). So nothing here is said of "the station's samples":
 * every count, date and default is of ONE variable, and a visit that did not measure it is counted as such,
 * never as a zero and never as a sample of it.
 *
 * States of a visit for a variable (the brief's data states, as far as the source distinguishes them):
 *   quantified  a published value above 0
 *   zero        a reported 0: at or below an unpublished laboratory limit, not absent (CalHABMAP publishes no ND flag)
 *   rejected    a negative value, rejected by Neritic's QC: no value
 *   absent      the visit has no value for this variable (NaN upstream): not measured, or not yet published
 */
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { distanceKm } from "@/lib/grid";
import { dtf, stableFormat } from "@/lib/time";
import { lastMeasured, SOURCE_COLUMN, sourceRowUrl, type Measured, type ObsVariableId } from "@/lib/observations";
import { clampWin, DAY, inWin, LABEL, pointsOf, UNIT, VAR_ORDER, valueWords, type Located, type Q11, type RecPoint, type Strip } from "./model";

// ---------------------------------------------------------------- availability

export type VisitState = "quantified" | "zero" | "rejected" | "absent";

/** One sampling visit, read for one variable (the availability strip draws one tick per visit). */
export type VisitTick = { t: number; time: string; i: number; state: VisitState };

const stateOf = (p: RecPoint | undefined): VisitState => (!p ? "absent" : p.cls === "value" ? "quantified" : p.cls);

/** Every visit of the station (optionally inside a window), with what it holds for one variable. */
export function visitTicks(st: Pick<ObsStation, "sample_times" | "series">, v: ObsVariableId, win?: [number, number]): VisitTick[] {
  const byIndex = new Map(pointsOf(st, v).map((p) => [p.i, p]));
  const out: VisitTick[] = [];
  st.sample_times.forEach((time, i) => {
    const t = Date.parse(time);
    if (!Number.isFinite(t) || (win && !inWin(t, win))) return;
    out.push({ t, time, i, state: stateOf(byIndex.get(i)) });
  });
  return out;
}

export type Availability = {
  v: ObsVariableId;
  /** sampling visits of the station inside the window, whatever they measured */
  visits: number;
  /** of those: a published value above 0 */
  quantified: number;
  /** of those: a reported 0 */
  zeros: number;
  /** of those: rejected by QC (no value) */
  rejected: number;
  /** of the quantified: flagged high, kept as published */
  flagged: number;
  /** of those: the visit has no value for this variable */
  absent: number;
  /** quantified + zeros: the samples of this variable in the window */
  withValue: number;
  /** the newest sample of this variable with a value in the whole record (may be outside the window), or null: never measured in Neritic's copy */
  last: RecPoint | null;
  /** the newest one inside the window */
  lastInWindow: RecPoint | null;
  /** the oldest sample of this variable with a value in the record */
  first: RecPoint | null;
  /** samples of this variable with a value in the whole record */
  total: number;
};

const hasValue = (p: RecPoint) => p.cls !== "rejected";

/** The newest sample that carries a value (a reported 0 counts; a rejected one does not). */
export function lastWithValue(points: RecPoint[], before = Infinity): RecPoint | null {
  for (let i = points.length - 1; i >= 0; i--) if (hasValue(points[i]) && points[i].t <= before) return points[i];
  return null;
}

/** What a window holds of one variable, against the station's visits in it. */
export function availabilityOf(st: Pick<ObsStation, "sample_times" | "series">, v: ObsVariableId, win: [number, number]): Availability {
  const ticks = visitTicks(st, v, win);
  const pts = pointsOf(st, v);
  const inside = pts.filter((p) => inWin(p.t, win));
  const n = (s: VisitState) => ticks.filter((k) => k.state === s).length;
  const valued = pts.filter(hasValue);
  return {
    v,
    visits: ticks.length,
    quantified: n("quantified"),
    zeros: n("zero"),
    rejected: n("rejected"),
    flagged: inside.filter((p) => p.flagged).length,
    absent: n("absent"),
    withValue: n("quantified") + n("zero"),
    last: valued.at(-1) ?? null,
    lastInWindow: lastWithValue(inside),
    first: valued[0] ?? null,
    total: valued.length,
  };
}

/** The variables this station has ever published a value for, in reading order (toxin, cells, algae, water). */
export function measuredVariables(st: Pick<ObsStation, "sample_times" | "series">): ObsVariableId[] {
  return VAR_ORDER.filter((v) => pointsOf(st, v).some(hasValue));
}

/** Availability of every variable the station has ever measured: the variable rail's rows. */
export function availabilityAll(st: Pick<ObsStation, "sample_times" | "series">, win: [number, number]): Availability[] {
  return measuredVariables(st).map((v) => availabilityOf(st, v, win));
}

/**
 * The variables of a visit that the same visits measured in the window, when the selected one was not: what the
 * visits DID record ("temperature only"). Empty when the window has no visit.
 */
export function measuredInstead(all: Availability[], v: ObsVariableId): ObsVariableId[] {
  return all.filter((a) => a.v !== v && a.withValue > 0).map((a) => a.v);
}

// ---------------------------------------------------------------- the quantity in focus

/**
 * The variable the page opens on: the link's `var` when this station has ever measured it (whatever the window
 * holds: an empty range is said, not dodged), else pDA, else the first it measures. `askedMissing` is the
 * link's variable when the station has never published it (said on the page, never silently swapped).
 */
export function chooseFocus(st: Pick<ObsStation, "sample_times" | "series">, asked: ObsVariableId | null): { focus: ObsVariableId; askedMissing: ObsVariableId | null; measured: ObsVariableId[] } {
  const measured = measuredVariables(st);
  if (asked && measured.includes(asked)) return { focus: asked, askedMissing: null, measured };
  return { focus: measured.find((v) => v === "pDA") ?? measured[0] ?? "pDA", askedMissing: asked, measured };
}

// ---------------------------------------------------------------- the latest sample

export const LATEST_SPAN_DAYS = 365;
/** The window "Latest sample" opens leaves this much after the sample, so its mark is not on the frame's edge. */
export const LATEST_TAIL_DAYS = 30;

/**
 * "Jump to the latest sample" of a variable: a year ending a month after its newest sample with a value, with
 * that sample's day pinned. Null when the variable has no sample with a value (nothing to jump to).
 */
export function latestWindow(points: RecPoint[], extent: [number, number], spanDays = LATEST_SPAN_DAYS): { win: [number, number]; day: number; point: RecPoint } | null {
  const point = lastWithValue(points);
  if (!point) return null;
  const end = point.t + LATEST_TAIL_DAYS * DAY;
  return { win: clampWin([end - spanDays * DAY, end], extent), day: point.t, point };
}

/**
 * Where the crosshair rests before the reader points: the newest sample OF THE SELECTED VARIABLE in the window
 * (a pinned day wins when it is in the window). Never the newest visit of any variable: under pDA at Monterey
 * that is a temperature-only day. Null when the window holds no sample of the variable: no crosshair.
 */
export function defaultCursor(focusPoints: RecPoint[], win: [number, number], pinned: number | null = null): number | null {
  if (pinned != null && inWin(pinned, win)) {
    const near = focusPoints.find((p) => Math.abs(p.t - pinned) < 1.01 * DAY);
    return near?.t ?? pinned;
  }
  return lastWithValue(focusPoints.filter((p) => inWin(p.t, win)))?.t ?? null;
}

/** The selected variable's samples in order: Previous / Next step through these, not through every visit. */
export function stepSample(points: RecPoint[], from: number | null, dir: -1 | 1, win?: [number, number]): RecPoint | null {
  const pts = win ? points.filter((p) => inWin(p.t, win)) : points;
  if (!pts.length) return null;
  if (from == null) return dir > 0 ? pts[0] : pts[pts.length - 1];
  if (dir > 0) return pts.find((p) => p.t > from + 1000) ?? null;
  for (let i = pts.length - 1; i >= 0; i--) if (pts[i].t < from - 1000) return pts[i];
  return null;
}

// ---------------------------------------------------------------- other stations

/** A station's newest sample of a variable, for the station list (the visit date is context, not the headline). */
export type StationLatest<S> = { station: S; last: Measured | null; lastVisit: string | null };

export function stationLatest<S extends Pick<ObsStation, "sample_times" | "series">>(stations: S[], v: ObsVariableId): StationLatest<S>[] {
  return stations.map((station) => ({ station, last: lastMeasured(station, v), lastVisit: station.sample_times.at(-1) ?? null }));
}

/** "Recent" for a suggested neighbour: a sample of the variable within this many days. */
export const RECENT_DAYS = 90;

export type Neighbour = { station: Located; km: number; last: Measured; ageDays: number };

/**
 * The nearest OTHER station whose newest sample of the variable is recent: what an empty range offers ("Santa
 * Cruz Wharf, 41 km, measured pDA on Oct 28"). Computed from the data; null when no other station has one.
 * It is a different pier: the page must say the distance and never show its values as this station's.
 */
export function nearestWithRecent(stations: Located[], from: Pick<Located, "station_id" | "lat" | "lon">, v: ObsVariableId, now: number, withinDays = RECENT_DAYS): Neighbour | null {
  let best: Neighbour | null = null;
  for (const station of stations) {
    if (station.station_id === from.station_id) continue;
    const last = lastMeasured(station, v);
    if (!last) continue;
    const ageDays = Math.floor((now - Date.parse(last.time)) / DAY);
    if (ageDays > withinDays || ageDays < -1) continue;
    const km = distanceKm(from.lat, from.lon, station.lat, station.lon);
    if (!best || km < best.km) best = { station, km, last, ageDays };
  }
  return best;
}

// ---------------------------------------------------------------- an empty range

export type EmptyRange = {
  v: ObsVariableId;
  /** never: no sample with a value in Neritic's copy; before / after: the record has samples, none in this window */
  kind: "never" | "before" | "after" | "gap";
  /** the newest sample with a value before the window's end (what "last measured" means for this window); null when the window lies before the first */
  last: RecPoint | null;
  /** the first sample after the window, when the window lies before or between samples */
  next: RecPoint | null;
  visits: number;
  /** what the window's visits did measure */
  instead: ObsVariableId[];
};

/**
 * Why a window holds no sample of the selected variable, in terms the page can say and act on. Null when it
 * holds one (a rejected sample alone is not a value: the range is still empty of values, and `rejected` says so
 * in the availability).
 */
export function emptyRange(st: Pick<ObsStation, "sample_times" | "series">, v: ObsVariableId, win: [number, number]): EmptyRange | null {
  const all = availabilityAll(st, win);
  const a = all.find((x) => x.v === v) ?? availabilityOf(st, v, win);
  if (a.withValue > 0) return null;
  const pts = pointsOf(st, v).filter(hasValue);
  const before = lastWithValue(pts, win[1]);
  const next = pts.find((p) => p.t > win[1]) ?? null;
  const kind = !pts.length ? "never" : before && next ? "gap" : before ? "after" : "before";
  return { v, kind, last: before, next, visits: a.visits, instead: measuredInstead(all, v) };
}

// ---------------------------------------------------------------- the model panel

/**
 * The model panel is collapsed (a header line saying so, the reader can open it) when the window holds no
 * measured value of the selected variable: C-HARM must not be the only "toxin" picture beside an empty
 * measured chart (brief guardrail: a C-HARM probability is not an observed harbour toxin value).
 */
export function modelCollapsed(focusPoints: RecPoint[], win: [number, number]): boolean {
  return !focusPoints.some((p) => hasValue(p) && inWin(p.t, win));
}

// ---------------------------------------------------------------- the legend

export type LegendId = "measured" | "zero" | "rejected" | "flagged" | "model" | "partial" | "threshold" | "now";

/**
 * One legend for the view: only the marks actually drawn. A strip with nothing in the window is folded (not
 * drawn), so its marks, its threshold and a collapsed model have no legend entry.
 */
export function legendFor(strips: Strip[], win: [number, number], drawn: (strip: Strip) => boolean = (x) => x.points.some((p) => inWin(p.t, win))): { id: LegendId; words: string }[] {
  const shown = strips.filter(drawn);
  const pts = shown.filter((x) => x.kind === "measured").flatMap((x) => x.points.filter((p) => inWin(p.t, win)));
  const m = shown.find((x) => x.kind === "model");
  const out: { id: LegendId; words: string }[] = [];
  if (pts.some((p) => p.cls === "value")) out.push({ id: "measured", words: "Measured" });
  if (pts.some((p) => p.cls === "zero")) out.push({ id: "zero", words: "Reported 0" });
  if (pts.some((p) => p.cls === "rejected")) out.push({ id: "rejected", words: "Rejected by QC" });
  if (pts.some((p) => p.flagged)) out.push({ id: "flagged", words: "Flagged high, kept" });
  if (m) out.push({ id: "model", words: "Model, hatched where none kept" });
  if (m?.partial && [...m.partial].some((t) => inWin(t, win))) out.push({ id: "partial", words: "Model day, few cells" });
  const th = shown.filter((x) => x.threshold);
  if (th.length) out.push({ id: "threshold", words: th.some((x) => /^pn_/.test(x.id)) ? "C-HARM threshold (Pseudo-nitzschia: all species)" : "C-HARM threshold" });
  if (shown.length) out.push({ id: "now", words: "Today" });
  return out;
}

// ---------------------------------------------------------------- counts for the export line

/** Rows the window's export holds, by variable: "49 rows: water temperature 49" (never "49 samples" of everything). */
export function exportCounts(strips: Strip[], win: [number, number]): { rows: number; by: { v: ObsVariableId; n: number }[] } {
  const by = strips.filter((x) => x.kind === "measured" && x.v).map((x) => ({ v: x.v as ObsVariableId, n: x.points.filter((p) => inWin(p.t, win)).length }));
  return { rows: by.reduce((a, x) => a + x.n, 0), by };
}

// ---------------------------------------------------------------- where Neritic's copy starts

export type CopyStart = {
  /** the earliest sample date the pipeline asked the source for (the dataset's `window_start`) */
  windowStart: string;
  /** this station's first sample in the copy (UTC date), or null */
  firstSample: string | null;
  /**
   * The copy may be cut: its first sample falls within two months of the requested start, so earlier samples
   * may exist at the source. (With the 2014 window, Monterey's first is Jan 8, 2014 and the source goes back
   * to 2005; with the 2005 window the first, Jun 3, 2005, is the source's own first.)
   */
  mayBeCut: boolean;
  /** the station's whole record at the source */
  sourceUrl: string;
};

export function copyStart(ds: Pick<ObservationDataset, "window_start">, st: Pick<ObsStation, "sample_times" | "source_url">): CopyStart {
  const firstSample = st.sample_times[0]?.slice(0, 10) ?? null;
  const w = Date.parse(`${ds.window_start}T00:00:00Z`);
  const f = firstSample ? Date.parse(`${firstSample}T00:00:00Z`) : NaN;
  return { windowStart: ds.window_start, firstSample, mayBeCut: Number.isFinite(f) && Number.isFinite(w) && f - w < 62 * DAY, sourceUrl: st.source_url };
}

// ---------------------------------------------------------------- one sample, inspected

export type SampleState = "measured" | "reported_zero" | "rejected" | "flagged_high";

const STATE_WORDS: Record<SampleState, string> = {
  measured: "Measured, as published",
  reported_zero: "Reported 0: at or below a laboratory limit the source does not publish. Not quantified, not absent",
  rejected: "Rejected by Neritic's QC: the source published a negative value. No value is shown or counted",
  flagged_high: "Above Neritic's plausibility ceiling: kept exactly as published and flagged for review",
};

export type CoMeasured = { v: ObsVariableId; label: string; unit: string; state: VisitState; /** exact published number, or null */ value: number | null; words: string };

export type SampleRecord = {
  station: { id: string; name: string; lat: number | null; lon: number | null; locationCode: string | null; nearestPort: string | null; nearestPortKm: number | null };
  variable: { id: ObsVariableId; label: string; unit: string; sourceColumn: string | null; method: string | null; detectionLimitNote: string | null; zeroPolicy: string | null };
  /** the sample time exactly as published (UTC, ISO-8601) */
  time: string;
  /** "2022-08-31 14:45 UTC" */
  utc: string;
  /** "Aug 31, 2022, 7:45 AM PDT" (the station's own clock) */
  local: string;
  /** the number exactly as in Neritic's copy (7 significant digits of the source's float32), null when rejected */
  value: number | null;
  /** the same, as text: nothing rounded ("0.05220236"), "" when rejected */
  valueText: string;
  /** at reading precision, with its unit ("0.052 ng/mL", "reported 0", "rejected by QC") */
  words: string;
  state: SampleState;
  stateWords: string;
  /** the pipeline's qualifier, as exported */
  qualifier: Q11 | null;
  depthM: number | null;
  /**
   * The source's SampleID is not in Neritic's copy (the published schema has no field for it); the source row
   * link shows it. Always null here so the page says "at the source", never an invented id.
   */
  sampleId: null;
  /** every other variable the station measures, as this same visit recorded it ("not measured" included) */
  others: CoMeasured[];
  /** this sample's own row on SCCOOS ERDDAP */
  sourceRowUrl: string | null;
  /** the station's dataset page */
  sourceUrl: string;
  retrievedAt: string | null;
  citation: string;
  /** the station's QC checks (station-level: the source publishes no per-sample flags) */
  qc: { name: string; passed: boolean; detail: string }[];
  /** the sample's index in the station's `sample_times` */
  index: number;
};

const pad = (n: number) => String(n).padStart(2, "0");
export function utcWords(time: string): string {
  const d = new Date(time);
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}
export function localWords(time: string): string {
  return stableFormat(dtf("en-US", { timeZone: "America/Los_Angeles", year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }), new Date(time));
}

const stripLike = (v: ObsVariableId) => ({ kind: "measured" as const, v, unitShort: UNIT[v] });

/**
 * Everything Neritic holds about one sample of one variable, for the inspector. `i` is the sample's index in
 * the station's `sample_times` (a RecPoint's `i`). Null when that visit has nothing for the variable: a visit
 * that did not measure it cannot be inspected as a sample of it.
 */
export function inspectSample(ds: Pick<ObservationDataset, "variables" | "provenance">, st: ObsStation, v: ObsVariableId, i: number): SampleRecord | null {
  const p = pointsOf(st, v).find((x) => x.i === i);
  if (!p) return null;
  const meta = ds.variables.find((x) => x.id === v);
  const state: SampleState = p.cls === "zero" ? "reported_zero" : p.cls === "rejected" ? "rejected" : p.flagged ? "flagged_high" : "measured";
  const qualifier: Q11 | null = state === "reported_zero" ? "reported_zero" : state === "rejected" ? "rejected_negative" : state === "flagged_high" ? "flag_high" : null;
  const others: CoMeasured[] = measuredVariables(st)
    .filter((o) => o !== v)
    .map((o) => {
      const q = pointsOf(st, o).find((x) => x.i === i);
      return { v: o, label: LABEL[o], unit: UNIT[o], state: stateOf(q), value: q && q.cls !== "rejected" ? q.v : null, words: q ? valueWords(stripLike(o), q) : "not measured in this sample" };
    });
  const cite = ds.provenance.citation ?? "California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP), via SCCOOS ERDDAP.";
  return {
    station: { id: st.station_id, name: st.name, lat: st.lat, lon: st.lon, locationCode: st.location_code ?? null, nearestPort: st.nearest_port_name ?? null, nearestPortKm: st.nearest_port_km ?? null },
    variable: { id: v, label: LABEL[v], unit: UNIT[v], sourceColumn: meta?.source_variable ?? SOURCE_COLUMN[v], method: meta?.method ?? null, detectionLimitNote: meta?.detection_limit_note ?? null, zeroPolicy: meta?.zero_policy ?? null },
    time: p.time,
    utc: utcWords(p.time),
    local: localWords(p.time),
    value: p.cls === "rejected" ? null : p.v,
    valueText: p.cls === "rejected" || p.v == null ? "" : String(p.v),
    words: valueWords(stripLike(v), p),
    state,
    stateWords: STATE_WORDS[state],
    qualifier,
    depthM: st.depths_m[i] ?? null,
    sampleId: null,
    others,
    sourceRowUrl: sourceRowUrl(st, p.time),
    sourceUrl: st.source_url,
    retrievedAt: st.retrieved_at ?? null,
    citation: `${cite} ${st.name} (${st.station_id}), ${LABEL[v].toLowerCase()}, sample of ${utcWords(p.time)}.`,
    qc: st.qc.map((q) => ({ name: q.name, passed: q.passed, detail: q.detail })),
    index: i,
  };
}
