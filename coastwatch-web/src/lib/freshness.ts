import type { FreshnessPolicy, SourceStatus, TimeInfo } from "@/generated/schema";

/**
 * Freshness is decided in the browser from published dates, so a stalled pipeline
 * degrades to "stale"/"historical" instead of presenting old data as current.
 */
export type FreshnessState = "current" | "stale" | "historical" | "unavailable";

export type Freshness = {
  state: FreshnessState;
  basisDate: string | null;
  ageDays: number | null;
};

const DAY_MS = 86_400_000;

/** Whole UTC calendar days from `date` (YYYY-MM-DD) to `now`. */
export function ageInDays(date: string, now: Date): number {
  const d = Date.parse(`${date}T00:00:00Z`);
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  return Math.round((today - d) / DAY_MS);
}

export function basisDateFromTime(policy: FreshnessPolicy, time: TimeInfo | null | undefined): string | null {
  if (!time) return null;
  if (policy.basis === "issued_date") return time.issued_date ?? null;
  if (policy.basis === "observed_date") return time.observed_date ?? null;
  return time.valid_date ?? null;
}

export function classifyDate(policy: FreshnessPolicy, date: string | null, now: Date): Freshness {
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return { state: "unavailable", basisDate: null, ageDays: null };
  }
  const age = ageInDays(date, now);
  const state: FreshnessState =
    age <= policy.current_max_age_days ? "current" : age <= policy.stale_max_age_days ? "stale" : "historical";
  return { state, basisDate: date, ageDays: age };
}

export function classifyTime(policy: FreshnessPolicy, time: TimeInfo | null | undefined, now: Date): Freshness {
  return classifyDate(policy, basisDateFromTime(policy, time), now);
}

export function classifySource(status: SourceStatus, now: Date): Freshness {
  const date =
    status.freshness.basis === "issued_date" ? status.latest_issued_date ?? null : status.latest_valid_date ?? null;
  return classifyDate(status.freshness, date, now);
}

export const FRESHNESS_LABEL: Record<FreshnessState, string> = {
  current: "Current",
  stale: "Stale",
  historical: "Historical — not current",
  unavailable: "Unavailable",
};
