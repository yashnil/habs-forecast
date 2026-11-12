/**
 * Model freshness by its valid dates, the ONE rule every surface uses (T3): a model or outlook run whose every
 * valid day is before today (Pacific) is a past run. It is never called an "outlook" or a "forecast" then, and
 * every page words it the same way: "past run (Oct 8)". Pure; `today` is the injected clock's Pacific day
 * (YYYY-MM-DD), never the machine's.
 *
 * This is separate from a layer's freshness policy (src/lib/freshness.ts, T17), which ages a product by its
 * issue or observation date: a run can be freshly issued and still cover only days that have passed (the 8-day
 * outlook's window starts before it is issued).
 */

export type RunTiming = "past" | "current" | "upcoming";

/** Where a run's valid window [from, to] sits against today. */
export function runTiming(validFrom: string | null | undefined, validTo: string | null | undefined, today: string): RunTiming {
  const to = validTo ?? validFrom;
  const from = validFrom ?? validTo;
  if (!to || !from) return "current";
  if (to.slice(0, 10) < today) return "past";
  if (from.slice(0, 10) > today) return "upcoming";
  return "current";
}

/** Every valid day before today: a past run. */
export const isPastRun = (validFrom: string | null | undefined, validTo: string | null | undefined, today: string) => runTiming(validFrom, validTo, today) === "past";

/** Every one of a run's valid days (each lead's date) before today. An empty list is not a past run. */
export const allPast = (dates: Array<string | null | undefined>, today: string) => dates.length > 0 && dates.every((d) => !!d && d.slice(0, 10) < today);

/**
 * The kind word for a model layer: "past run (Oct 8)" once its dates have passed, else its kind ("model",
 * "outlook"). `issuedWords` is the issue date already in the page's date style.
 */
export function runKindWord(kind: "model" | "outlook", past: boolean, issuedWords?: string | null): string {
  return past ? `past run${issuedWords ? ` (${issuedWords})` : ""}` : kind;
}
