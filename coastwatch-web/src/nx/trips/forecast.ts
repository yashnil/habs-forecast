import type { CoastSea } from "@/nx/coast/brief";
import { periodNumbers } from "@/pilot/today/marine";
import type { ForecastPeriod, ForecastSnap } from "./types";

/**
 * The NWS coastal waters (zone) forecast as a trip keeps it: each period's wind, gusts, seas and wave
 * components read from NWS's own sentence, and a valid interval derived from the period's name.
 *
 * Why derived: the API's zone forecast (`/zones/forecast/{zoneId}/forecast`, OpenAPI schema `ZoneForecast`,
 * https://api.weather.gov/openapi.json) gives only `updated` ("the time this zone forecast product was published")
 * and periods of `{ number, name, detailedForecast }` — no start or end. Point (gridpoint) forecasts carry
 * startTime/endTime; zone forecasts do not. NWS forecast periods run in local time, day 6 AM–6 PM and night
 * 6 PM–6 AM (the convention of the NWS text products, NWS Directive 10-310 for the Coastal Waters Forecast; period
 * names "Rest of Today", "This Afternoon", "Tonight", "Overnight", then weekday names). So the interval here is a
 * derivation from name + issuance, and every surface that shows it says so (`DERIVED_NOTE`). Marine wording handled:
 * "NW wind 5 to 10 kt, becoming W 10 to 20 kt with gusts to 25 kt in the afternoon", "Wind W 10 kt...becoming N
 * after midnight", "Light winds", "Wind variable less than 10 kt", "Seas 3 to 5 ft", "Seas around 8 ft",
 * "Wave Detail: NW 2 ft at 6 seconds, W 3 ft at 10 seconds and S 2 ft at 17 seconds" (recorded in
 * tests/conditions-fixtures.json). Nothing is invented: a number the sentence does not state stays null.
 */

export const NWS_TZ = "America/Los_Angeles";
export const DERIVED_NOTE = "Valid times derived from NWS period names";

const DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
const DIR = `(${DIRS.join("|")})`;

/** Compass point → degrees the wind blows FROM (for an arrow); null for "light"/"variable"/unknown. */
export function compassDeg(dir: string | null): number | null {
  const i = dir ? DIRS.indexOf(dir.toUpperCase()) : -1;
  return i < 0 ? null : i * 22.5;
}

/** One wind as NWS stated it, in order: the first, then each "becoming / backing to / easing to …". */
export type StatedWind = {
  dir: string | null;
  lo: number | null;
  hi: number | null;
  light: boolean;
  variable: boolean;
  /** "less than 10 kt": the upper value is stated, no lower one */
  lessThan: boolean;
  gust: number | null;
  /** the verb that introduced a change ("becoming", "backing to"); null for the first */
  verb: string | null;
  /** NWS's own timing words ("in the afternoon", "after midnight"); null when none */
  when: string | null;
};

const VERB = /^(becoming|backing to|veering to|easing to|increasing to|rising to|shifting to|diminishing to|decreasing to|subsiding to|then)\s+/i;
const WHEN = /\b(early this evening|late this evening|this evening|after midnight|late tonight|overnight|in the morning|in the afternoon|in the evening|late in the afternoon|early in the morning|late)\b\.?$/i;

function segment(raw: string, first: boolean): StatedWind | null {
  let t = raw.trim().replace(/\.$/, "");
  let verb: string | null = null;
  const v = VERB.exec(t);
  if (v) {
    verb = v[1].toLowerCase();
    t = t.slice(v[0].length);
  } else if (!first) return null;
  const gm = /\s*with gusts (?:up )?to (\d+) kt/i.exec(t);
  const gust = gm ? Number(gm[1]) : null;
  if (gm) t = t.replace(gm[0], "");
  const wm = WHEN.exec(t);
  const when = wm ? wm[1].toLowerCase() : null;
  const dm = new RegExp(`(?:^|\\s)${DIR}(?=\\s|$)`).exec(t);
  const dir = dm ? dm[1].toUpperCase() : null;
  const light = /\blight\b/i.test(t);
  const variable = /\bvariable\b/i.test(t);
  const lt = /less than (\d+)\s*kt/i.exec(t);
  const rg = /(\d+) to (\d+)\s*kt/i.exec(t);
  const one = /(?:around|near|up to|to)?\s*(\d+)\s*kt/i.exec(t);
  let lo: number | null = null;
  let hi: number | null = null;
  if (lt) hi = Number(lt[1]);
  else if (rg) [lo, hi] = [Number(rg[1]), Number(rg[2])];
  else if (one) lo = hi = Number(one[1]);
  if (dir == null && hi == null && !light && !variable && gust == null) return null;
  return { dir, lo, hi, light, variable, lessThan: !!lt, gust, verb, when };
}

/** The winds stated in a period, in NWS's order. Empty when the text states none. */
export function statedWinds(text: string): StatedWind[] {
  const t = text.replace(/\s+/g, " ");
  const sentences = t.split(/(?<=\.)\s+(?=[A-Z])/);
  const out: StatedWind[] = [];
  for (const s of sentences) {
    if (/^(seas|wave detail|swell|mixed swell|combined seas|a chance|a slight chance|showers|rain|patchy|areas of)/i.test(s)) continue;
    if (!/\bwinds?\b/i.test(s) && !/\bgusts?\b/i.test(s)) continue;
    const parts = s.split(/,\s*|\.\.\.+\s*|\s+(?=then\s)/i);
    parts.forEach((p, i) => {
      const w = segment(p, i === 0 && out.length === 0);
      if (w) out.push(w);
      else if (i === 0 && out.length > 0) {
        // a later wind sentence ("Gusts up to 25 kt.") without its own wind: keep only a gust it states
        const g = /gusts? (?:up )?to (\d+)\s*kt/i.exec(p);
        if (g) out[out.length - 1].gust = Math.max(out[out.length - 1].gust ?? 0, Number(g[1]));
      }
    });
  }
  return out;
}

/** Wave components as NWS states them ("Wave Detail: NW 2 ft at 6 seconds, W 3 ft at 10 seconds and S 2 ft at 17 seconds"). */
export function waveComponents(text: string): ForecastPeriod["waves"] {
  const t = text.replace(/\s+/g, " ");
  const detail = /wave detail:([^.]*(?:\.\d[^.]*)*)/i.exec(t);
  if (detail) {
    return [...detail[1].matchAll(new RegExp(`${DIR}\\s+(\\d+)\\s*(?:ft|feet|foot)(?:\\s+at\\s+(\\d+)\\s*seconds)?`, "gi"))].map((m) => ({ dir: m[1].toUpperCase(), ft: Number(m[2]), periodS: m[3] ? Number(m[3]) : null }));
  }
  const sw = periodNumbers(t).swell;
  return sw ? [{ dir: sw.dir, ft: sw.ft, periodS: sw.s }] : [];
}

/** Seas: the first stated range, and its top raised by a stated later build ("building to 8 to 10 ft"). */
function seas(text: string): { lo: number | null; hi: number | null } {
  const p = periodNumbers(text);
  if (p.seasHi == null) return { lo: null, hi: null };
  let hi = p.seasHi;
  const s = /\b(?:combined\s+)?seas?\b[^.]*/i.exec(text.replace(/\s+/g, " "));
  if (s) for (const m of s[0].matchAll(/\b(?:building|increasing|rising|subsiding|decreasing|lowering) to (?:around )?(\d+)(?: to (\d+))?\s*(?:ft|feet)/gi)) hi = Math.max(hi, Number(m[2] ?? m[1]));
  return { lo: p.seasLo, hi };
}

// ---------------------------------------------------------------- time

const addDays = (date: string, n: number) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 86400000).toISOString().slice(0, 10);

/** The zone's offset from UTC (ms) at an instant. */
function offsetMs(utc: number, timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).formatToParts(new Date(utc));
  const g = (k: string) => Number(parts.find((p) => p.type === k)?.value);
  return Date.UTC(g("year"), g("month") - 1, g("day"), g("hour") % 24, g("minute"), g("second")) - utc;
}

/** A local wall time ("2026-10-13", 18, 0) in a zone → UTC ISO. */
export function zonedIso(date: string, hour: number, minute: number, timeZone: string): string {
  const guess = Date.UTC(Number(date.slice(0, 4)), Number(date.slice(5, 7)) - 1, Number(date.slice(8, 10)), hour, minute);
  let t = guess - offsetMs(guess, timeZone);
  t = guess - offsetMs(t, timeZone);
  return new Date(t).toISOString();
}

/** The local calendar date and hour of an instant in a zone. */
export function localParts(iso: string, timeZone: string): { date: string; hour: number; minute: number } {
  const t = Date.parse(iso);
  const l = new Date(t + offsetMs(t, timeZone));
  return { date: l.toISOString().slice(0, 10), hour: l.getUTCHours(), minute: l.getUTCMinutes() };
}

/**
 * A CO-OPS tide time is wall-clock text in the station's zone ("2026-10-13 12:03"); an ISO string with an offset
 * or Z is an instant. Either → epoch ms.
 */
export function tideMs(at: string, timeZone = NWS_TZ): number {
  if (/[zZ]|[+-]\d{2}:?\d{2}$/.test(at)) return Date.parse(at);
  const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})/.exec(at);
  return m ? Date.parse(zonedIso(m[1], Number(m[2]), Number(m[3]), timeZone)) : NaN;
}

const WEEKDAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const clock = (iso: string, timeZone: string) => {
  const { hour, minute } = localParts(iso, timeZone);
  return `${hour % 12 === 0 ? 12 : hour % 12}${minute ? `:${String(minute).padStart(2, "0")}` : ""} ${hour < 12 ? "AM" : "PM"}`;
};

/** "Tue 13, night" — the derived label of a period key. */
export function keyLabel(key: string): string {
  const [date, half] = key.split("|");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return key.replace(/^undated\|/, "");
  return `${WEEKDAY[new Date(`${date}T12:00:00Z`).getUTCDay()]} ${Number(date.slice(8))}, ${half === "night" ? "night" : "day"}`;
}

/** "Tue 13 night, 6 PM–6 AM" (derived; say DERIVED_NOTE near it). */
export function periodLabel(p: Pick<ForecastPeriod, "key" | "start" | "end">, timeZone = NWS_TZ): string {
  const base = keyLabel(p.key);
  return p.start && p.end ? `${base.replace(", ", " ")}, ${clock(p.start, timeZone)}–${clock(p.end, timeZone)}` : base;
}

// ---------------------------------------------------------------- the forecast

const NIGHT_NAME = /night|tonight|overnight|evening/i;

/**
 * CoastSea (the /coast/data?part=sea answer) → the forecast a trip keeps. Periods keep NWS's sentence and name; their
 * date comes from lib/nwsPeriods (via the sea window) and their interval is derived here: day 06:00–18:00, night
 * 18:00–06:00 local, the first period starting no earlier than the issuance. "Overnight" in a bulletin issued after
 * midnight belongs to the previous evening's night (so it compares with that night's "Tonight").
 */
export function forecastFromSea(sea: CoastSea | null | undefined, opts: { timeZone?: string } = {}): ForecastSnap {
  const tz = opts.timeZone ?? NWS_TZ;
  const v = sea?.sea;
  if (!v || v.state !== "ok") return { zone: v?.zone ?? null, issued: v?.issued ?? null, url: v?.marineUrl ?? null, periods: [] };
  const issued = v.issued && Number.isFinite(Date.parse(v.issued)) ? v.issued : null;
  const issueLocal = issued ? localParts(issued, tz) : null;
  const seen = new Set<string>();
  const periods: ForecastPeriod[] = [];
  for (const [i, lane] of v.lanes.entries()) {
    const night = lane.night ?? NIGHT_NAME.test(lane.name);
    let date = lane.date;
    if (i === 0 && date && night && issueLocal && issueLocal.hour < 6 && date === issueLocal.date && /^(overnight|late tonight|rest of tonight|tonight)/i.test(lane.name.trim())) date = addDays(date, -1);
    const key = date ? `${date}|${night ? "night" : "day"}` : `undated|${lane.name}`;
    if (seen.has(key)) continue;
    seen.add(key);
    let start: string | null = null;
    let end: string | null = null;
    if (date) {
      start = zonedIso(date, night ? 18 : 6, 0, tz);
      end = night ? zonedIso(addDays(date, 1), 6, 0, tz) : zonedIso(date, 18, 0, tz);
      if (issued && Date.parse(issued) > Date.parse(start) && Date.parse(issued) < Date.parse(end)) start = new Date(Date.parse(issued)).toISOString();
    }
    const winds = statedWinds(lane.text);
    const pn = periodNumbers(lane.text);
    const his = winds.map((w) => w.hi).filter((x): x is number => x != null);
    // a "light" or "less than 10 kt" wind states no lower number: the period's lower bound is then not stated
    const noLow = winds.some((w) => w.lessThan || (w.light && w.hi == null) || (w.variable && w.hi == null));
    const los = noLow ? [] : winds.map((w) => w.lo).filter((x): x is number => x != null);
    const gusts = [...winds.map((w) => w.gust), pn.gust].filter((x): x is number => x != null);
    const first = winds[0] ?? null;
    const sz = seas(lane.text);
    periods.push({
      key,
      name: lane.name,
      date,
      night,
      start,
      end,
      text: lane.text,
      windDir: first ? (first.dir ?? (first.variable ? "variable" : first.light ? "light" : null)) : null,
      windLo: los.length ? Math.min(...los) : null,
      windHi: his.length ? Math.max(...his) : null,
      gustKt: gusts.length ? Math.max(...gusts) : null,
      seasLo: sz.lo,
      seasHi: sz.hi,
      waves: waveComponents(lane.text),
    });
  }
  return { zone: v.zone, issued, url: v.marineUrl, periods };
}

// ---------------------------------------------------------------- words

const spd = (lo: number | null, hi: number | null, lessThan = false) => (hi == null ? null : lessThan ? `under ${hi} kt` : lo != null && lo !== hi ? `${lo}–${hi} kt` : `${hi} kt`);

/** "W 10–15 kt, then NW 5–10 kt late this evening, then N after midnight" — the period's wind as stated, changes kept. */
export function windStatedWords(text: string): string | null {
  const ws = statedWinds(text);
  if (!ws.length) return null;
  return ws
    .map((w, i) => [i ? "then" : null, w.dir ?? (w.variable ? "variable" : w.light ? "light" : null), spd(w.lo, w.hi, w.lessThan), w.gust != null ? `gusts ${w.gust} kt` : null, w.when].filter(Boolean).join(" "))
    .join(", ");
}

/** The directions stated, in order, with repeats folded: ["W", "NW", "N"]. */
export function directionSequence(text: string): string[] {
  const out: string[] = [];
  for (const w of statedWinds(text)) {
    const d = w.dir ?? (w.variable ? "variable" : w.light ? "light" : null);
    if (d && out.at(-1) !== d) out.push(d);
  }
  return out;
}

export const seasRangeWords = (p: Pick<ForecastPeriod, "seasLo" | "seasHi">) => (p.seasHi == null ? null : p.seasLo != null && p.seasLo !== p.seasHi ? `${p.seasLo}–${p.seasHi} ft` : `${p.seasHi} ft`);
export const windRangeWords = (p: Pick<ForecastPeriod, "windLo" | "windHi">) => spd(p.windLo, p.windHi);
export const wavesWords = (p: Pick<ForecastPeriod, "waves">) => (p.waves.length ? p.waves.map((w) => [w.dir, w.ft != null ? `${w.ft} ft` : null, w.periodS != null ? `at ${w.periodS} s` : null].filter(Boolean).join(" ")).join(", ") : null);
