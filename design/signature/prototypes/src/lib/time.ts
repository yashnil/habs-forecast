import type { Layer, SourceStatus } from "../data/types";

/** The prototypes run on a fixed clock: the evening the data snapshot was published (PDT). */
export const NOW = new Date("2026-10-12T18:00:00-07:00");
const TZ = "America/Los_Angeles";

export const dayKey = (d: Date) => d.toLocaleDateString("en-CA", { timeZone: TZ });
export const TODAY = dayKey(NOW);

export function daysBetween(a: string, b: string) {
  return Math.round((Date.parse(b + "T12:00:00Z") - Date.parse(a + "T12:00:00Z")) / 86_400_000);
}
export function fmtDay(iso: string, opts: { weekday?: boolean; year?: boolean } = {}) {
  const d = new Date(iso.length === 10 ? iso + "T12:00:00Z" : iso);
  return d.toLocaleDateString("en-US", { timeZone: TZ, month: "short", day: "numeric", weekday: opts.weekday ? "short" : undefined, year: opts.year ? "numeric" : undefined });
}
/** "today", "yesterday", "3 days ago", "Sep 30" — relative words only within a week. */
export function relDay(iso: string) {
  const n = daysBetween(iso.slice(0, 10), TODAY);
  if (n === 0) return "today";
  if (n === 1) return "yesterday";
  if (n === -1) return "tomorrow";
  if (n > 1 && n < 7) return `${n} days ago`;
  if (n < -1 && n > -7) return `in ${-n} days`;
  return fmtDay(iso);
}
export function ageWords(iso: string) {
  const n = daysBetween(iso.slice(0, 10), TODAY);
  if (n <= 0) return "today";
  if (n === 1) return "1 day old";
  if (n < 60) return `${n} days old`;
  const m = Math.round(n / 30.4);
  if (m < 24) return `${m} months old`;
  return `${Math.round(n / 365)} years old`;
}

/** Freshness words settled by the redesign: Current / Not confirmed / Stale / Historical. */
export type Freshness = "Current" | "Not confirmed" | "Stale" | "Historical";
export function forecastFreshness(issued: string, src?: SourceStatus): Freshness {
  const age = daysBetween(issued, TODAY);
  if (age > 7) return "Historical";
  if (age > 1) return "Stale";
  return src && src.outcome !== "ok" ? "Not confirmed" : "Current";
}
export function layerDate(l: Layer) {
  return l.time.valid_date ?? l.time.observed_date ?? l.time.issued_date ?? "";
}
