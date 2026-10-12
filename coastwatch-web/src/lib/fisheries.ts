import type { FisheriesDataset, SpeciesGroup, YearValue } from "@/generated/fisheries";

/**
 * Fisheries & Economic Exposure (M7) helpers. Every number is read from the published dataset:
 * nothing is interpolated, split among ports, or attributed to a cause. A missing value stays
 * missing (null), never zero.
 */

export type Dollars = "real" | "nominal";
/** What the figures measure: revenue in base-year dollars, revenue as reported, or landed weight. */
export type Unit = Dollars | "pounds";
export const UNITS: Unit[] = ["real", "nominal", "pounds"];
export type TierSet = "1" | "12";

export function fisheriesValue(v: Pick<YearValue, "dollars_real" | "dollars_nominal" | "pounds">, u: Unit): number | null {
  return u === "real" ? (v.dollars_real ?? null) : u === "nominal" ? (v.dollars_nominal ?? null) : (v.pounds ?? null);
}

export const isDollars = (u: Unit): u is Dollars => u !== "pounds";

/** The unit as a phrase for subtitles, axis titles and screen readers. */
export function unitLabel(u: Unit, baseYear: number): string {
  return u === "real" ? `${baseYear} dollars (adjusted with the CPI-U)` : u === "nominal" ? "nominal dollars (as reported each year)" : "pounds landed";
}

/** Compact value for figures, bars and tables: $49.7M, $628k, 14.2M lb. */
export function formatValue(v: number | null, u: Unit): string {
  if (v == null) return "no value";
  if (u === "pounds") {
    if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B lb`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M lb`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)}k lb`;
    return `${Math.round(v)} lb`;
  }
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}k`;
  return `$${Math.round(v)}`;
}

/** Axis tick label: whole numbers where possible ($40M, 20M lb). */
export function formatTick(v: number, u: Unit): string {
  if (v === 0) return u === "pounds" ? "0 lb" : "$0";
  const m = v >= 1e6;
  const n = m ? v / 1e6 : v / 1e3;
  const s = `${Number.isInteger(n) ? n : n.toFixed(1)}${m ? "M" : "k"}`;
  return u === "pounds" ? `${s} lb` : `$${s}`;
}

export const groupsInTier = (groups: SpeciesGroup[], t: TierSet) => groups.filter((g) => t === "12" || g.tier === 1);

export function groupValue(g: SpeciesGroup, year: number, u: Unit): number | null {
  const a = g.annual.find((x) => x.year === year);
  return a ? fisheriesValue(a, u) : null;
}

/** Sum of the selected groups for a year; null only if every group lacks a value. */
export function selectedTotal(groups: SpeciesGroup[], year: number, u: Unit): number | null {
  const vals = groups.map((g) => groupValue(g, year, u));
  return vals.every((v) => v == null) ? null : vals.reduce<number>((s, v) => s + (v ?? 0), 0);
}

/** NOAA's statewide total (all species, including the withheld row). */
export function stateTotal(ds: FisheriesDataset, year: number, u: Unit): number | null {
  const t = ds.statewide_total.find((x) => x.year === year);
  return t ? fisheriesValue(t, u) : null;
}

export function withheldValue(ds: FisheriesDataset, year: number, u: Unit): number | null {
  const w = ds.withheld.find((x) => x.year === year);
  return w ? fisheriesValue(w, u) : null;
}

/** Share of NOAA's statewide total, 0..1; null when either side is missing or the total is 0. */
export function shareOfState(part: number | null, total: number | null): number | null {
  return part == null || total == null || total <= 0 ? null : part / total;
}

export const pct = (s: number | null) => (s == null ? "—" : s < 0.01 && s > 0 ? "<1%" : `${Math.round(s * 100)}%`);

/** Years where any selected group has source rows without a published value (totals are lower bounds). */
export function rowsWithoutValue(groups: SpeciesGroup[], year: number): number {
  return groups.reduce((n, g) => n + (g.annual.find((a) => a.year === year)?.n_rows_without_value ?? 0), 0);
}

export type Span = { mean: number | null; low: { year: number; v: number } | null; high: { year: number; v: number } | null; n: number };

/** Mean, lowest and highest year of a series; years without a value are left out and counted. */
export function span(years: number[], value: (y: number) => number | null): Span {
  const pts = years.map((year) => ({ year, v: value(year) })).filter((p): p is { year: number; v: number } => p.v != null);
  if (!pts.length) return { mean: null, low: null, high: null, n: 0 };
  const low = pts.reduce((a, b) => (b.v < a.v ? b : a));
  const high = pts.reduce((a, b) => (b.v > a.v ? b : a));
  return { mean: pts.reduce((s, p) => s + p.v, 0) / pts.length, low, high, n: pts.length };
}

/** Nice axis maximum and ticks (1, 2, 2.5, 5 × 10^k) for a positive maximum. */
export function niceTicks(max: number, count = 4): number[] {
  if (!(max > 0)) return [0, 1];
  const raw = max / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw)!;
  const top = Math.ceil(max / step) * step;
  const out: number[] = [];
  for (let v = 0; v <= top + step / 2; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** A date as a fractional calendar year (2015-11-06 → 2015.85), for marking events on annual bars. */
export function yearFraction(iso: string): number {
  const [y, m, d] = iso.split("-").map(Number);
  const start = Date.UTC(y, 0, 1);
  return y + (Date.UTC(y, m - 1, d) - start) / (Date.UTC(y + 1, 0, 1) - start);
}

export const GROUP_COLOR: Record<string, string> = {
  // Tier 1 (toxin closures): blues of the history family; Tier 2 (advisories): stone. Teal, violet
  // and amber stay reserved for measured, model and official.
  dungeness_crab: "#1d4f7a",
  rock_crab: "#4a86bb",
  northern_anchovy: "#9cc0e0",
  spiny_lobster: "#6d604d",
  pacific_sardine: "#a29478",
  bivalves: "#d3c8b0",
};
export const groupColor = (id: string) => GROUP_COLOR[id] ?? "#8a96a3";
/** One strong colour per tier for single-group series (strips, detail bars), where a pale group
 *  colour would fall below 3:1 against the paper. */
export const tierInk = (tier: number) => (tier === 1 ? "#1d4f7a" : "#6d604d");

/** Short labels for chart legends and narrow tables; the full published label is used elsewhere. */
export const GROUP_SHORT: Record<string, string> = {
  dungeness_crab: "Dungeness crab",
  rock_crab: "Rock crabs",
  northern_anchovy: "Northern anchovy",
  spiny_lobster: "Spiny lobster",
  pacific_sardine: "Pacific sardine",
  bivalves: "Bivalves",
};
export const groupShort = (g: Pick<SpeciesGroup, "id" | "label">) => GROUP_SHORT[g.id] ?? g.label;

export type FisheriesView = { unit: Unit; tiers: TierSet; group: string | null; year: number };

/** The view from the link, falling back to defaults for anything unknown. */
export function parseView(ds: Pick<FisheriesDataset, "years" | "groups">, q: { unit?: string; tiers?: string; group?: string; year?: string }): FisheriesView {
  const latest = ds.years[ds.years.length - 1];
  const unit = (UNITS as string[]).includes(q.unit ?? "") ? (q.unit as Unit) : "real";
  const group = ds.groups.some((g) => g.id === q.group) ? q.group! : null;
  // a Tier 2 group in the link implies both tiers, so the chart can show it
  const tiers: TierSet = q.tiers === "all" || ds.groups.find((g) => g.id === group)?.tier === 2 ? "12" : "1";
  const y = Number(q.year);
  const year = ds.years.includes(y) ? y : latest;
  return { unit, tiers, group, year };
}

export function viewQuery(v: FisheriesView, latest: number): string {
  const q = new URLSearchParams();
  if (v.unit !== "real") q.set("unit", v.unit);
  if (v.tiers === "12") q.set("tiers", "all");
  if (v.group) q.set("group", v.group);
  if (v.year !== latest) q.set("year", String(v.year));
  return q.toString();
}
