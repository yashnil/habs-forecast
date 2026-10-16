import type { ObsStation, ObsVariableSummary, SeriesPoint } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { ageInDays, classifyDate, type Freshness } from "@/lib/freshness";
import { DAY, monthYear, summaryOf, type ObsVariableId, type Sample } from "@/lib/observations";
import { pacificToday } from "@/lib/time";

/**
 * Bloom Intelligence (M6) presentation helpers. Every chart on the page takes series computed
 * here from the published observations, so presentation cannot change scientific handling
 * (design spec §4.3). Nothing here interpolates, smooths, fills or rescales.
 */

/** The five quantities of the measurement selector, in reading order (spec §3.2 item 5). */
export const SELECTOR_VARS = ["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"] as const satisfies readonly ObsVariableId[];
export type SelectorVar = (typeof SELECTOR_VARS)[number];

export type Axis = { scale: "log" | "linear"; domain: [number, number]; edges: number[] };

/**
 * Fixed axes (spec §5.2). Log quantities run over whole decades that never rescale to the data,
 * so a quiet year cannot look like a bloom. `edges` are the heatmap bins: decades for log
 * quantities (§5.3), 2 °C steps for temperature. Display intervals, never risk levels.
 */
export const AXES: Record<ObsVariableId, Axis> = {
  pDA: { scale: "log", domain: [1e-4, 1e2], edges: [0.01, 0.1, 1, 10] },
  dDA: { scale: "log", domain: [1e-4, 1e2], edges: [0.01, 0.1, 1, 10] },
  tDA: { scale: "log", domain: [1e-4, 1e2], edges: [0.01, 0.1, 1, 10] },
  pn_seriata: { scale: "log", domain: [1, 1e7], edges: [1e3, 1e4, 1e5, 1e6] },
  pn_delicatissima: { scale: "log", domain: [1, 1e7], edges: [1e3, 1e4, 1e5, 1e6] },
  chl_extracted: { scale: "log", domain: [1e-2, 1e3], edges: [0.1, 1, 10, 100] },
  temp: { scale: "linear", domain: [5, 30], edges: [12, 14, 16, 18] },
};

/** Whole-decade ticks of a log axis. */
export function decadeTicks(domain: [number, number]): number[] {
  const out: number[] = [];
  for (let e = Math.round(Math.log10(domain[0])); e <= Math.round(Math.log10(domain[1])); e++) out.push(Number(`1e${e}`)); // exact, no 10 ** -4 drift
  return out;
}

/** A value that goes on the axis: measured, not a reported zero, not rejected, and > 0 on a log axis. */
export function onAxis(s: Sample, axis: Axis): boolean {
  return s.value != null && s.q !== "reported_zero" && s.q !== "rejected_negative" && (axis.scale === "linear" || s.value > 0);
}

export const inView = (s: { t: number }, x0: number, x1: number) => s.t >= x0 && s.t <= x1;

/** Where an on-axis value falls against the fixed domain. A value outside it is drawn as a marker
 *  at the edge (never silently clamped onto a gridline) and breaks the line. */
export function axisPosition(v: number, axis: Axis): "in" | "below" | "above" {
  return v < axis.domain[0] ? "below" : v > axis.domain[1] ? "above" : "in";
}

// ---------------------------------------------------------------- freshness card

export type SamplingStrip = {
  t0: number;
  t1: number;
  /** every sampling visit in the window (any quantity) */
  visits: number[];
  /** the newest visit ever, even before the window */
  last: number | null;
  /** whole Pacific days from the newest visit to today */
  daysSinceLast: number | null;
};

/** The 120-day sampling strip: one tick per visit, the gap since the last visit, today marked. */
export function samplingStrip(sampleTimes: string[], now: Date, days = 120): SamplingStrip {
  const t1 = now.getTime();
  const t0 = t1 - days * DAY;
  const all = sampleTimes.map((s) => Date.parse(s)).filter(Number.isFinite);
  const last = all.length ? Math.max(...all) : null;
  const lastDate = sampleTimes.length ? sampleTimes[all.indexOf(last!)].slice(0, 10) : null;
  return {
    t0,
    t1,
    visits: all.filter((t) => t >= t0 && t <= t1).sort((a, b) => a - b),
    last,
    daysSinceLast: lastDate ? ageInDays(lastDate, now) : null,
  };
}

// ---------------------------------------------------------------- selector cards

export type Headline =
  | { kind: "never" }
  | { kind: "old"; lastDate: string; value: number | null; q: ObsVariableSummary["last_qualifier"] }
  | { kind: "latest"; lastDate: string; value: number | null; q: ObsVariableSummary["last_qualifier"]; freshness: Freshness };

/**
 * What a selector card may show at headline size (spec §3.2 item 5, rule 8). A value older than
 * the stale limit is never a headline: the card says "Not measured since …" and keeps the value
 * in small text, so a years-old number cannot look current.
 */
export function headline(sm: ObsVariableSummary | null, policy: FreshnessPolicy, now: Date): Headline {
  if (!sm || sm.n_measured === 0 || !sm.last_date) return { kind: "never" };
  const f = classifyDate(policy, sm.last_date, now);
  if (f.state === "historical" || f.state === "unavailable") return { kind: "old", lastDate: sm.last_date, value: sm.last_value ?? null, q: sm.last_qualifier ?? null };
  return { kind: "latest", lastDate: sm.last_date, value: sm.last_value ?? null, q: sm.last_qualifier ?? null, freshness: f };
}

export type PreviewPoint = { t: number; v: number; zero: false; clipped: boolean } | { t: number; v: null; zero: true; clipped: false };

/**
 * A selector card's preview: the samples in the chart's period, on the same fixed axis. Reported
 * zeros are flagged to sit on the baseline as open dots; values beyond the axis are clamped to it
 * and marked; visits without a value are left out (gaps stay empty).
 */
export function previewPoints(ss: Sample[], axis: Axis, x0: number, x1: number): PreviewPoint[] {
  const out: PreviewPoint[] = [];
  for (const s of ss) {
    if (!inView(s, x0, x1) || s.value == null || s.q === "rejected_negative") continue;
    if (s.q === "reported_zero" || (axis.scale === "log" && s.value <= 0)) {
      out.push({ t: s.t, v: null, zero: true, clipped: false });
      continue;
    }
    const v = Math.min(Math.max(s.value, axis.domain[0]), axis.domain[1]);
    out.push({ t: s.t, v, zero: false, clipped: v !== s.value });
  }
  return out;
}

// ---------------------------------------------------------------- primary chart

/** Consecutive on-axis samples joined by a line; a gap longer than `gapDays` breaks it (§5.2), and
 *  so does a value outside the fixed axis (it is a marker at the edge, not a point on the line). */
export function segments(ss: Sample[], axis: Axis, x0: number, x1: number, gapDays = 21): Sample[][] {
  const out: Sample[][] = [];
  let open = false;
  for (const p of ss) {
    if (!inView(p, x0, x1) || !onAxis(p, axis)) continue;
    if (axisPosition(p.value!, axis) !== "in") {
      open = false;
      continue;
    }
    const cur = out.at(-1);
    if (open && cur && p.t - cur[cur.length - 1].t <= gapDays * DAY) cur.push(p);
    else out.push([p]);
    open = true;
  }
  return out;
}

/** The highest value on the axis in view ("Highest in view", with its date). */
export function highestInView(ss: Sample[], axis: Axis, x0: number, x1: number): Sample | null {
  let best: Sample | null = null;
  for (const s of ss) if (inView(s, x0, x1) && onAxis(s, axis) && (!best || s.value! > best.value!)) best = s;
  return best;
}

/** The chart's summary line: "Measured in 49 of 49 sampling visits in view · 10 reported 0". */
export function viewSummary(ss: Sample[], x0: number, x1: number) {
  const visits = ss.filter((s) => inView(s, x0, x1));
  const measured = visits.filter((s) => s.value != null && s.q !== "rejected_negative");
  return {
    visits: visits.length,
    measured: measured.length,
    zeros: measured.filter((s) => s.q === "reported_zero").length,
    rejected: visits.filter((s) => s.q === "rejected_negative").length,
    flagged: measured.filter((s) => s.q === "flag_high").length,
  };
}

// ---------------------------------------------------------------- season heatmap

export type HeatCell =
  | { kind: "unsampled" }
  | { kind: "zero"; n: number }
  | { kind: "bin"; bin: number; max: number; at: number; n: number };

/** Week of the year (0–52) or month (0–11) of a sample, by its Pacific calendar date. */
export function periodOf(t: number, period: "week" | "month"): { year: number; i: number } {
  const d = pacificToday(new Date(t)); // YYYY-MM-DD in California
  const [y, m, day] = d.split("-").map(Number);
  if (period === "month") return { year: y, i: m - 1 };
  const doy = (Date.UTC(y, m - 1, day) - Date.UTC(y, 0, 1)) / DAY;
  return { year: y, i: Math.min(52, Math.floor(doy / 7)) };
}

/** Index of the bin a value falls in: 0 below the first edge … edges.length at or above the last. */
export function binOf(v: number, edges: number[]): number {
  let i = 0;
  while (i < edges.length && v >= edges[i]) i++;
  return i;
}

/**
 * The calendar heatmap (spec §3.2 item 8): the highest value per week or month since the first
 * year, binned on the quantity's display edges. Periods with only reported zeros and periods
 * without a sample of this quantity have their own kinds, never a colour bin.
 */
export function heatmap(ss: Sample[], axis: Axis, period: "week" | "month"): { years: number[]; cols: number; cells: Map<string, Exclude<HeatCell, { kind: "unsampled" }>> } {
  const cells = new Map<string, Exclude<HeatCell, { kind: "unsampled" }>>();
  const years = new Set<number>();
  for (const s of ss) {
    if (s.value == null || s.q === "rejected_negative") continue;
    const { year, i } = periodOf(s.t, period);
    years.add(year);
    const key = `${year}:${i}`;
    const cur = cells.get(key);
    if (s.q === "reported_zero" || (axis.scale === "log" && s.value <= 0)) {
      if (!cur) cells.set(key, { kind: "zero", n: 1 });
      else if (cur.kind === "zero") cur.n++;
      else if (cur.kind === "bin") cur.n++;
      continue;
    }
    if (!cur || cur.kind === "zero" || (cur.kind === "bin" && s.value > cur.max)) {
      const n = cur ? cur.n + 1 : 1;
      cells.set(key, { kind: "bin", bin: binOf(s.value, axis.edges), max: s.value, at: s.t, n });
    } else if (cur.kind === "bin") cur.n++;
  }
  const ys = [...years].sort((a, b) => a - b);
  const all = ys.length ? Array.from({ length: ys[ys.length - 1] - ys[0] + 1 }, (_, k) => ys[0] + k) : [];
  return { years: all, cols: period === "week" ? 53 : 12, cells };
}

// ---------------------------------------------------------------- model band

export type ModelPoint = { t: number; date: string; v: number; n: number; partial: boolean };

/**
 * C-HARM nowcast history on the measurements' time axis (spec §3.2 item 9, §5.4). Days without a
 * value are left out (never interpolated). A day whose neighbourhood had fewer than 60 % of the
 * usual valid cells is `partial`: drawn as an open marker and not joined.
 */
export function modelOnAxis(history: SeriesPoint[] | undefined, x0: number, x1: number) {
  const pts = (history ?? []).filter((p) => p.value != null);
  const ns = pts.map((p) => p.n).sort((a, b) => a - b);
  const usual = ns.length ? ns[Math.floor(ns.length / 2)] : 0;
  const all: ModelPoint[] = pts.map((p) => ({
    t: Date.parse(`${p.date}T12:00:00Z`),
    date: p.date,
    v: p.value as number,
    n: p.n,
    partial: usual > 0 && p.n < 0.6 * usual,
  }));
  const first = all.length ? all[0].t : null;
  return {
    points: all.filter((p) => inView(p, x0, x1)),
    /** before this, Neritic kept no model history: the band hatches it */
    historyStart: first,
  };
}

/** Joined runs of full-coverage model days; a missing day or a partial day breaks the line. */
export function modelSegments(points: ModelPoint[]): ModelPoint[][] {
  const out: ModelPoint[][] = [];
  for (const p of points) {
    if (p.partial) {
      out.push([]);
      continue;
    }
    const cur = out.at(-1);
    if (cur && cur.length && p.t - cur[cur.length - 1].t <= 1.5 * DAY) cur.push(p);
    else out.push([p]);
  }
  return out.filter((s) => s.length);
}

// ---------------------------------------------------------------- station rail

const SHORT_DAY = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const shortDay = (date: string) => SHORT_DAY.format(new Date(`${date.slice(0, 10)}T12:00:00Z`));

/** Two significant figures, trailing zeros kept (0.20, not 0.2), as spec §5.2. */
const sig2 = (v: number) => (v >= 100 ? v.toFixed(0) : v.toPrecision(2));

/** One plain line about particulate domoic acid at a station (spec §3.2 item 2). Never implies absence. */
export function pdaLine(s: ObsStation, policy: FreshnessPolicy | null, now: Date | null): string {
  const sm = summaryOf(s, "pDA");
  if (!sm || sm.n_measured === 0) return "pDA not measured here";
  const h = policy && now ? headline(sm, policy, now) : null;
  if (h?.kind === "old") return `pDA last measured ${monthYear(sm.last_date!)}`;
  if (sm.last_qualifier === "reported_zero") return `pDA reported 0 · ${shortDay(sm.last_date!)}`;
  return `pDA ${sm.last_value != null ? sig2(sm.last_value) : "—"} ng/mL`;
}
