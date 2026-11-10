/**
 * Series maths, DOM-free. Truth rules: T7 (censored values printed as such and drawn as their own mark,
 * never as 0 or as the limit), T19/T20 (a gap is drawn as a gap; zero is not missing), T2 (model hollow).
 */
import { day, sig, toDate } from "../format";

/** reported0: a lab's 0 with no published limit; censored: "<2.5"; nd: "ND", not detected. */
export type Flag = "reported0" | "censored" | "nd";

export interface SeriesPoint {
  /** ISO date or date-time. */
  t: string;
  /** Null: not measured in this sample (missing, never zero). */
  v: number | null;
  /** What the source printed, kept verbatim ("<2.5", "ND"). */
  raw?: string;
  flag?: Flag;
}

export type PointClass = "value" | "censored" | "missing";

export function classify(p: SeriesPoint): PointClass {
  if (p.flag) return "censored";
  if (p.v === null || !Number.isFinite(p.v)) return "missing";
  return "value";
}

/** The words a censored point is printed with. Never "0" alone, never the limit as a value. */
export function censoredWords(p: SeriesPoint): string {
  if (p.flag === "reported0") return "reported 0";
  if (p.flag === "nd") return "ND, not detected";
  if (p.flag === "censored") return `${p.raw ?? "below the limit"}, below the limit`;
  return "";
}

export const ms = (t: string) => toDate(t).getTime();

/**
 * Runs of drawable values for a line. A run breaks at a censored sample (it has no place on the axis),
 * and wherever consecutive samples are more than `gapDays` apart (a gap is drawn as a gap).
 * Missing samples are skipped: they are not samples of this variable.
 */
export function segments(points: SeriesPoint[], gapDays = 45): SeriesPoint[][] {
  const out: SeriesPoint[][] = [];
  let run: SeriesPoint[] = [];
  let lastT: number | null = null;
  for (const p of points) {
    const c = classify(p);
    if (c === "missing") continue;
    const t = ms(p.t);
    if (c === "censored" || (lastT !== null && t - lastT > gapDays * 86_400_000)) {
      if (run.length) out.push(run);
      run = [];
    }
    if (c === "value") run.push(p);
    lastT = t;
  }
  if (run.length) out.push(run);
  return out;
}

/** Powers of ten that cover [lo, hi]. */
export function logDomain(values: number[], floor = 1e-4): [number, number] {
  const pos = values.filter((v) => v > 0);
  if (!pos.length) return [floor, 1];
  const lo = Math.max(floor, 10 ** Math.floor(Math.log10(Math.min(...pos))));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...pos)));
  return [lo, hi === lo ? lo * 10 : hi];
}

export function logTicks([lo, hi]: [number, number]): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e);
  return out;
}

/** Tick label on a log axis: 0.0001 … 100, no scientific notation. */
export function logTickLabel(v: number): string {
  if (v >= 1e6) return `${Math.round(v / 1e6).toLocaleString("en-US")}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3).toLocaleString("en-US")}K`;
  if (v >= 1) return Math.round(v).toLocaleString("en-US");
  return v.toFixed(Math.max(0, -Math.floor(Math.log10(v))));
}

/** Index of the sample nearest in time to `t` (points sorted by time). */
export function nearestIndex(points: SeriesPoint[], t: number): number {
  if (!points.length) return -1;
  let lo = 0, hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ms(points[mid].t) < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(ms(points[lo].t) - t) <= Math.abs(ms(points[hi].t) - t) ? lo : hi;
}

/** What the crosshair says about one sample: "Sep 30, 2026 · 0.20 ng/mL · measured". */
export function readoutText(p: SeriesPoint, unit: string, kindWord: string): string {
  const c = classify(p);
  const when = day(p.t, { year: true });
  if (c === "censored") return `${when} · ${censoredWords(p)} · ${kindWord}`;
  if (c === "missing") return `${when} · not measured in this sample`;
  return `${when} · ${sig(p.v as number, 2)} ${unit} · ${kindWord}`;
}

/** One sentence for assistive tech: count, span, censored count. */
export function describeSeries(points: SeriesPoint[], label: string): string {
  const samples = points.filter((p) => classify(p) !== "missing");
  if (!samples.length) return `${label}: no samples in this window.`;
  const cens = samples.filter((p) => classify(p) === "censored").length;
  const first = day(samples[0].t, { year: true });
  const last = day(samples[samples.length - 1].t, { year: true });
  return `${label}: ${samples.length} samples, ${first} to ${last}${cens ? `; ${cens} below the limit or reported 0` : ""}.`;
}

/** Samples inside a time window [a, b] in ms. */
export function within(points: SeriesPoint[], [a, b]: [number, number]): SeriesPoint[] {
  return points.filter((p) => {
    const t = ms(p.t);
    return t >= a && t <= b;
  });
}
