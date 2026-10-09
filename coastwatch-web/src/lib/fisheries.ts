import type { SpeciesGroup, YearValue } from "@/generated/fisheries";

export type Dollars = "real" | "nominal";

export function fisheriesValue(v: YearValue, d: Dollars): number | null {
  return d === "real" ? (v.dollars_real ?? null) : (v.dollars_nominal ?? null);
}

/** Sum of the selected groups for a year; null only if every group lacks a value. */
export function selectedTotal(groups: SpeciesGroup[], year: number, d: Dollars): number | null {
  const vals = groups.map((g) => g.annual.find((a) => a.year === year)).map((a) => (a ? fisheriesValue(a, d) : null));
  return vals.every((v) => v == null) ? null : vals.reduce<number>((s, v) => s + (v ?? 0), 0);
}
