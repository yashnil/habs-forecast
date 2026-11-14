import type { StageField } from "./types";

/**
 * The words a field's legend must always carry. Rendered by the stage from the field, not passed in, so a
 * caption cannot be written without them (design.md §3: chlorophyll is never toxin, at every width).
 */
export function fieldCaveat(field: StageField): string | null {
  if (field === "chlorophyll" || field === "outlook") return "not toxin";
  if (field === "charm") return "probability of conditions, not toxin in seafood";
  return null;
}

/** "Oct 7" or a span ("Oct 3–12", "Sep 28–Oct 2"), UTC calendar days as published (YYYY-MM-DD). */
export function shortDate(iso: string): string {
  return new Date(`${iso.slice(0, 10)}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}
export function dateSpan(oldest: string | null | undefined, newest: string | null | undefined): string {
  if (!newest && !oldest) return "date unknown";
  if (!oldest || !newest || oldest === newest) return shortDate((newest ?? oldest)!);
  const a = shortDate(oldest);
  const b = shortDate(newest);
  return a.slice(0, 3) === b.slice(0, 3) ? `${a}–${b.slice(4)}` : `${a}–${b}`;
}

/** A forecast whose every day is before today's date is a past run: it is never called an outlook. */
export function isPastRun(lastValidDate: string | null | undefined, today: string): boolean {
  return !!lastValidDate && lastValidDate.slice(0, 10) < today;
}
