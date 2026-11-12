/**
 * Research → the ds plot kit, DOM-free: a station's samples and the model's days as `PlotPoint`s, what each
 * says, and the availability rail's ticks. Truth rules as everywhere on this page: a reported 0 and a rejected
 * sample have no place on the y axis (their own marks on the baseline), a visit that did not measure the
 * variable is not a point at all, and the model is a different kind on a different panel.
 */
import type { DataStateKind, PlotPoint } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import type { VisitTick } from "./availability";
import { valueWords, type RecPoint, type Strip } from "./model";

/** A sample's state in the kit's vocabulary. */
export function kindOf(strip: Pick<Strip, "kind">, p: RecPoint): DataStateKind {
  if (strip.kind === "model") return "forecast";
  return p.cls === "zero" ? "reported-zero" : p.cls === "rejected" ? "rejected" : "measured";
}

/** What a sample says in one line, for the tooltip, the live region and the table. */
export function sayPoint(strip: Pick<Strip, "kind" | "v" | "unitShort" | "partial">, p: RecPoint): string {
  const when = day(p.t, { year: true });
  if (strip.kind === "model") return `${when} · ${valueWords(strip, p)} · model, not measured${strip.partial?.has(p.t) ? " · few cells that day" : ""}`;
  if (p.cls === "zero") return `${when} · reported 0 · at or below an unpublished limit, not absent`;
  if (p.cls === "rejected") return `${when} · rejected by QC · no value`;
  return `${when} · ${valueWords(strip, p)} · measured${p.flagged ? ", flagged high" : ""}`;
}

export type SamplePoint = PlotPoint & { meta: RecPoint };

/** A strip's points for a `<Plot>`: one per sample, the RecPoint kept as `meta`, the station as its place. */
export function plotPoints(strip: Strip, place?: [number, number]): SamplePoint[] {
  return strip.points.map((p) => ({
    t: p.t,
    v: p.cls === "value" ? p.v : null,
    kind: kindOf(strip, p),
    raw: p.cls === "zero" ? "reported 0" : p.cls === "rejected" ? "rejected by QC" : undefined,
    place,
    meta: p,
    id: `${strip.id}-${p.i}`,
  }));
}

/**
 * The model's days for its `<Line>`: a day with few cells is not joined (a break in its place), and the rest
 * join only day to day (the caller sets `gapDays`).
 */
export function modelLine(points: SamplePoint[], partial: ReadonlySet<number> | undefined): PlotPoint[] {
  return points.map((p) => (partial?.has(p.t) ? { t: p.t, v: null, kind: "missing" as const } : p));
}

/** "today", "12 days ago", "5 months ago", "4.2 years ago": how old a variable's last sample is. */
export function ageWords(days: number): string {
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 60) return `${days} days ago`;
  if (days < 730) return `${Math.round(days / 30.44)} months ago`;
  return `${(days / 365.25).toFixed(1)} years ago`;
}

/**
 * The availability rail's ticks as SVG path data on a 0–1000 × 0–12 box: one stroke per visit, by state.
 * `value` = a published value above 0 (tall), `zero` = reported 0 (a short stroke at the foot), `rejected`
 * (a short stroke at the head), `absent` = the station was visited and this variable was not measured (a dot
 * on the centre line). Four paths, so 900 visits cost four elements.
 */
export function tickPaths(ticks: VisitTick[], extent: [number, number]): Record<"quantified" | "zero" | "rejected" | "absent", string> {
  const span = Math.max(1, extent[1] - extent[0]);
  const out = { quantified: "", zero: "", rejected: "", absent: "" };
  for (const k of ticks) {
    const x = Math.round(((k.t - extent[0]) / span) * 10000) / 10;
    if (x < 0 || x > 1000) continue;
    if (k.state === "quantified") out.quantified += `M${x} 1V11`;
    else if (k.state === "zero") out.zero += `M${x} 7V11`;
    else if (k.state === "rejected") out.rejected += `M${x} 1V4`;
    else out.absent += `M${x} 5.5V6.5`;
  }
  return out;
}
