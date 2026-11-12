/**
 * The plot kit's maths, DOM-free (v13 README §4). Truth rules as in ds/series: a gap is drawn as a gap, a
 * reported zero and a value below detection are their own marks and never a zero on the axis, a model value is
 * hollow, and data is never smoothed or tweened.
 */
import { day } from "../format";
import { DATA_STATE_WORDS, type DataStateKind } from "../dataStates";

/** A value at three significant figures without trailing noise: 2, 6.2, 0.0013, 1,060. */
export function plain(v: number): string {
  if (!Number.isFinite(v)) return "";
  const r = Number(v.toPrecision(3));
  return `${r < 0 ? "−" : ""}${Math.abs(r).toLocaleString("en-US", { maximumFractionDigits: 6 })}`;
}

export interface PlotPoint {
  /** Position on x: ms for a time axis, a number for a linear one. */
  t: number;
  /** Null: nothing to place on the y axis (reported zero, below detection with no limit, not collected). */
  v: number | null;
  /** The reading's state: decides its mark and its word. Default "measured". */
  kind?: DataStateKind;
  /** What the source printed, verbatim ("<2.5", "ND"), said in the readout instead of a number. */
  raw?: string;
  /** Where it was measured, for a linked chart: [lon, lat]. */
  place?: [number, number];
  /** Anything the page needs back when the point is inspected (source, QC, sample id). */
  meta?: unknown;
  id?: string;
}

export type XSpec = { type?: "time" | "linear"; domain: [number, number]; /** a linear axis: its unit ("lb") and how a position is written, for ticks, the tooltip and the live region */ unit?: string; format?: (t: number) => string };
export type YSpec = { type?: "linear" | "log"; domain?: [number, number]; unit?: string; format?: (v: number) => string };

/** Kinds that have a place on the y axis when they carry a value. */
/** (`rejected` is never on the axis: a value quality control rejected is not drawn as one.) */
const ON_AXIS: DataStateKind[] = ["measured", "estimated", "forecast", "historical", "stale", "unverified", "below-detection"];
export const onAxis = (p: PlotPoint) => p.v !== null && Number.isFinite(p.v) && ON_AXIS.includes(p.kind ?? "measured");

/** The y domain from the values that have a place on the axis: padded for linear, whole decades for log. */
export function yDomainOf(points: PlotPoint[], type: "linear" | "log" = "linear", include: number[] = []): [number, number] {
  const vals = [...points.filter(onAxis).map((p) => p.v as number), ...include].filter((v) => Number.isFinite(v) && (type === "linear" || v > 0));
  if (!vals.length) return type === "log" ? [0.1, 10] : [0, 1];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  if (type === "log") return [10 ** Math.floor(Math.log10(lo) - 1e-9), 10 ** Math.ceil(Math.log10(hi) + 1e-9 + (lo === hi ? 1 : 0))];
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  return [lo >= 0 ? 0 : lo - pad, hi + pad];
}

/** Runs of joinable values: a line breaks at anything without a place on the axis and at gaps over `gapDays`. */
export function runs(points: PlotPoint[], gapMs: number): PlotPoint[][] {
  const out: PlotPoint[][] = [];
  let cur: PlotPoint[] = [];
  let last: number | null = null;
  for (const p of points) {
    if (!onAxis(p) || p.kind === "below-detection") {
      if (cur.length) out.push(cur);
      cur = [];
      last = null;
      continue;
    }
    if (last !== null && p.t - last > gapMs && cur.length) {
      out.push(cur);
      cur = [];
    }
    cur.push(p);
    last = p.t;
  }
  if (cur.length) out.push(cur);
  return out;
}

/** Index of the point nearest to `t` (points sorted by t); -1 when there are none. */
export function nearest(points: PlotPoint[], t: number): number {
  if (!points.length) return -1;
  let lo = 0, hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].t < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo].t - t) <= Math.abs(points[hi].t - t) ? lo : hi;
}

export type StepKey = "prev" | "next" | "first" | "last" | "pagePrev" | "pageNext";
/**
 * Where the keyboard goes from `from` (an index, or -1 for "nowhere yet"): one sample left or right, the first
 * or last, or a page (30 days on a time axis, a tenth of the points otherwise). Never leaves the list.
 */
export function stepIndex(points: PlotPoint[], from: number, key: StepKey, time = true): number {
  const n = points.length;
  if (!n) return -1;
  if (key === "first") return 0;
  if (key === "last") return n - 1;
  if (from < 0) return key === "prev" || key === "pagePrev" ? n - 1 : 0;
  if (key === "prev") return Math.max(0, from - 1);
  if (key === "next") return Math.min(n - 1, from + 1);
  const dir = key === "pageNext" ? 1 : -1;
  if (!time) return Math.min(n - 1, Math.max(0, from + dir * Math.max(1, Math.round(n / 10))));
  const i = nearest(points, points[from].t + dir * 30 * 86_400_000);
  return i === from ? Math.min(n - 1, Math.max(0, from + dir)) : i;
}

/** What a point says, for the tooltip, the live region and the table: "Jun 3, 2025 · 6.2 ng/mL · measured". */
export function describePoint(p: PlotPoint, o: { unit?: string; time?: boolean; format?: (v: number) => string; xFormat?: (t: number) => string; xUnit?: string } = {}): string {
  const kind = p.kind ?? "measured";
  const word = DATA_STATE_WORDS[kind].word.toLowerCase();
  const where = o.time === false ? `${(o.xFormat ?? plain)(p.t)}${o.xUnit ? ` ${o.xUnit}` : ""}` : day(p.t, { year: true });
  if (p.raw && (kind === "below-detection" || kind === "reported-zero")) return `${where} · ${p.raw} · ${word}`;
  if (p.v === null || !Number.isFinite(p.v)) return `${where} · ${word}`;
  return `${where} · ${(o.format ?? plain)(p.v)}${o.unit ? ` ${o.unit}` : ""} · ${word}`;
}

/** Round ticks for a linear axis: about `count` of them, on 1, 2, 2.5 or 5 times a power of ten. */
export function linearTicks([lo, hi]: [number, number], count = 5): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step + 0);
  return out;
}

/**
 * A window of the same width moved so it holds `t` (centred, kept inside the record): the "jump to the latest
 * sample" move. A width may be given to change it at the same time.
 */
export function windowAround(extent: [number, number], current: [number, number], t: number, width?: number): [number, number] {
  const w = Math.min(extent[1] - extent[0], Math.max(1, width ?? current[1] - current[0]));
  const a = Math.min(Math.max(t - w / 2, extent[0]), extent[1] - w);
  return [a, a + w];
}

/**
 * Break-even along a row of a catch × price grid: the first column whose value is at or above zero, per row
 * (a step line the grid draws as a contour). Null where a row never breaks even or has gaps.
 */
export function breakEvenSteps(values: Array<Array<number | null>>): Array<number | null> {
  return values.map((row) => {
    const i = row.findIndex((v) => v !== null && v >= 0);
    return i === -1 || row.some((v) => v === null) ? null : i;
  });
}
