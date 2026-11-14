import type { ForecastPeriod, ForecastSnap, OperatingLimits } from "./types";
import { directionSequence, keyLabel, NWS_TZ, seasRangeWords, statedWinds, tideMs, wavesWords, windStatedWords, zonedIso, type StatedWind } from "./forecast";

/**
 * Departure windows: each forecast period set beside the reader's own limits, the predicted tide and NWS
 * advisories. Descriptive only. A crossing says which of the reader's limits a stated number exceeds and by what
 * number ("stated up to 25 kt; your limit 20 kt"); nothing here ranks periods or calls one good or bad. A period with
 * no forecast, or one already over, is its own state, never a period without crossings. Personal limits are the
 * reader's own thresholds, not certified or recommended ones.
 */

export type TideEvent = { at: string; ft: number; type: "H" | "L" };
/** An NWS alert as the sea answer or a snapshot carries it; `onset` null = in effect from issue (already active). */
export type AdvisoryIn = { event: string; headline: string; url: string; onset?: string | null; ends: string | null };

export type Crossing = {
  limit: "wind" | "gust" | "seas";
  /** the forecast's stated number that exceeds the limit */
  stated: number;
  limitValue: number;
  unit: "kt" | "ft";
  /** "Wind stated up to 25 kt; your limit 20 kt" */
  words: string;
};

export type CellState = "reported" | "expired" | "unavailable";

export type WindowCell = {
  key: string;
  label: string;
  date: string | null;
  night: boolean;
  start: string | null;
  end: string | null;
  /** "reported": NWS states this period. "expired": its derived interval is over. "unavailable": no forecast covers it */
  state: CellState;
  period: ForecastPeriod | null;
  winds: StatedWind[];
  crossings: Crossing[];
  /** limits the reader set that the forecast states nothing to compare with ("no seas stated") */
  notStated: ("wind" | "seas")[];
  tides: TideEvent[];
  advisories: AdvisoryIn[];
  /** the trip's departure date falls in this period's calendar day */
  departure: boolean;
};

const addDays = (date: string, n: number) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 86400000).toISOString().slice(0, 10);

/** Which of the reader's limits a period's stated numbers exceed (strictly: equal to the limit is not over it). */
export function crossings(p: Pick<ForecastPeriod, "windHi" | "gustKt" | "seasHi">, limits: OperatingLimits): Crossing[] {
  const out: Crossing[] = [];
  if (limits.windKt != null) {
    if (p.windHi != null && p.windHi > limits.windKt) out.push({ limit: "wind", stated: p.windHi, limitValue: limits.windKt, unit: "kt", words: `Wind stated up to ${p.windHi} kt; your limit ${limits.windKt} kt` });
    if (p.gustKt != null && p.gustKt > limits.windKt) out.push({ limit: "gust", stated: p.gustKt, limitValue: limits.windKt, unit: "kt", words: `Gusts stated to ${p.gustKt} kt; your wind limit ${limits.windKt} kt` });
  }
  if (limits.seasFt != null && p.seasHi != null && p.seasHi > limits.seasFt) out.push({ limit: "seas", stated: p.seasHi, limitValue: limits.seasFt, unit: "ft", words: `Seas stated up to ${p.seasHi} ft; your limit ${limits.seasFt} ft` });
  return out;
}

const overlaps = (aStart: number, aEnd: number, bStart: number, bEnd: number) => aStart < bEnd && bStart < aEnd;

function bounds(date: string, night: boolean, tz: string): [string, string] {
  return night ? [zonedIso(date, 18, 0, tz), zonedIso(addDays(date, 1), 6, 0, tz)] : [zonedIso(date, 6, 0, tz), zonedIso(date, 18, 0, tz)];
}

/**
 * The cells, in time order: every forecast period, plus an "unavailable" cell for each half-day missing between
 * them and for the departure date when the forecast does not reach it. Undated periods are kept at the end as
 * unavailable for comparison (their time is unknown).
 */
export function windowCells(
  f: ForecastSnap | null,
  limits: OperatingLimits,
  tides: TideEvent[] | null | undefined,
  alerts: AdvisoryIn[] | null | undefined,
  now: string,
  opts: { departDate?: string | null; timeZone?: string } = {},
): WindowCell[] {
  const tz = opts.timeZone ?? NWS_TZ;
  const nowMs = Date.parse(now);
  const dated = (f?.periods ?? []).filter((p) => p.date && p.start && p.end);
  const byKey = new Map(dated.map((p) => [p.key, p]));
  const keys: string[] = [];
  if (dated.length) {
    const first = dated[0];
    const last = dated[dated.length - 1];
    let d = first.date!;
    let night = first.night;
    const lastKey = last.key;
    for (let guard = 0; guard < 40; guard++) {
      keys.push(`${d}|${night ? "night" : "day"}`);
      if (keys.at(-1) === lastKey) break;
      if (night) d = addDays(d, 1);
      night = !night;
    }
  }
  const dep = opts.departDate ?? null;
  if (dep) for (const half of ["day", "night"]) if (!keys.includes(`${dep}|${half}`)) keys.push(`${dep}|${half}`);
  keys.sort();
  const tideList = (tides ?? []).map((t) => ({ t, ms: tideMs(t.at, tz) })).filter((x) => Number.isFinite(x.ms));
  const cells: WindowCell[] = keys.map((key) => {
    const [date, half] = key.split("|");
    const night = half === "night";
    const p = byKey.get(key) ?? null;
    const [start, end] = p ? [p.start!, p.end!] : bounds(date, night, tz);
    const s = Date.parse(start);
    const e = Date.parse(end);
    const state: CellState = e <= nowMs ? "expired" : p ? "reported" : "unavailable";
    const winds = p ? statedWinds(p.text) : [];
    const notStated: ("wind" | "seas")[] = [];
    if (p && limits.windKt != null && p.windHi == null) notStated.push("wind");
    if (p && limits.seasFt != null && p.seasHi == null) notStated.push("seas");
    return {
      key,
      label: keyLabel(key),
      date,
      night,
      start,
      end,
      state,
      period: p,
      winds,
      crossings: p && state === "reported" ? crossings(p, limits) : [],
      notStated,
      tides: tideList.filter((x) => x.ms >= s && x.ms < e).map((x) => x.t),
      advisories: (alerts ?? []).filter((a) => {
        const on = a.onset ? Date.parse(a.onset) : -Infinity;
        const off = a.ends ? Date.parse(a.ends) : Infinity;
        return overlaps(on, off, s, e);
      }),
      departure: !!dep && date === dep,
    };
  });
  for (const p of (f?.periods ?? []).filter((x) => !x.date || !x.start)) {
    cells.push({ key: p.key, label: p.name, date: null, night: p.night, start: null, end: null, state: "unavailable", period: p, winds: statedWinds(p.text), crossings: [], notStated: [], tides: [], advisories: [], departure: false });
  }
  return cells;
}

// ---------------------------------------------------------------- comparing two windows

export type WindowField = "state" | "windDir" | "wind" | "gust" | "seas" | "waves" | "advisories" | "tides" | "limits";
export type WindowDiff = { field: WindowField; label: string; a: string; b: string };

const NOT_STATED = "not stated";
const LABEL: Record<WindowField, string> = { state: "Forecast", windDir: "Wind direction", wind: "Wind speed", gust: "Gusts", seas: "Seas", waves: "Wave detail", advisories: "NWS advisories", tides: "Tide", limits: "Your limits" };
const STATE_WORDS: Record<CellState, string> = { reported: "NWS forecast", expired: "period over", unavailable: "no forecast for this period" };

const range = (lo: number | null, hi: number | null, unit: string) => (hi == null ? NOT_STATED : lo != null && lo !== hi ? `${lo}–${hi} ${unit}` : `${hi} ${unit}`);
/** CO-OPS wall time "2026-10-13 12:03" → "12:03 PM", read as written (never through the browser's zone). */
export function tideClockOf(at: string): string {
  const m = /[ T](\d{2}):(\d{2})/.exec(at);
  if (!m) return at;
  const h = Number(m[1]);
  return `${h % 12 === 0 ? 12 : h % 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
}
const tideWordsOf = (ts: TideEvent[]) => (ts.length ? ts.map((t) => `${t.type === "H" ? "high" : "low"} ${t.ft.toFixed(1)} ft ${tideClockOf(t.at)}`).join(", ") : "none in this period");

/** One cell's reported conditions as comparable words. "not stated" is its own value, never 0. */
export function cellFacts(c: WindowCell): Record<WindowField, string> {
  const p = c.state === "reported" ? c.period : null;
  return {
    state: STATE_WORDS[c.state],
    windDir: p ? directionSequence(p.text).join(" → ") || NOT_STATED : "—",
    wind: p ? range(p.windLo, p.windHi, "kt") : "—",
    gust: p ? (p.gustKt != null ? `${p.gustKt} kt` : NOT_STATED) : "—",
    seas: p ? (seasRangeWords(p) ?? NOT_STATED) : "—",
    waves: p ? (wavesWords(p) ?? NOT_STATED) : "—",
    advisories: c.advisories.length ? c.advisories.map((a) => a.event).join(", ") : "none",
    tides: tideWordsOf(c.tides),
    limits: c.state !== "reported" ? "not compared" : c.crossings.length ? c.crossings.map((x) => x.words).join("; ") : c.notStated.length ? `${c.notStated.join(" and ")} not stated` : "no stated value over your limits",
  };
}

/** Exactly which reported conditions differ between two windows, in reading order. Identical facts are left out. */
export function compareWindows(a: WindowCell, b: WindowCell): WindowDiff[] {
  const fa = cellFacts(a);
  const fb = cellFacts(b);
  return (Object.keys(LABEL) as WindowField[]).filter((k) => fa[k] !== fb[k]).map((k) => ({ field: k, label: LABEL[k], a: fa[k], b: fb[k] }));
}

/** The period's wind as stated, changes kept ("N 5–10 kt, then W 10–20 kt gusts 25 kt in the afternoon"). */
export const cellWindWords = (c: WindowCell) => (c.period ? windStatedWords(c.period.text) : null);
