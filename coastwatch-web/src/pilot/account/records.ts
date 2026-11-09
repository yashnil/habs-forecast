import { readJSON, writeJSON } from "./storage";

/**
 * Feedback ("Was this useful?") and trip logs, kept on this device. When signed in they are also
 * sent to the person's rows (sync.ts); the device copy is what the page shows.
 */
export const FEEDBACK_KEY = "nrt-pilot-feedback-v1";
export const TRIPS_KEY = "nrt-pilot-trips-v1";

export type FeedbackRecord = { fact_id: string; place_id: string | null; useful: boolean; note: string | null; at: string };
export type OutcomeBand = "skipped" | "poor" | "fair" | "good";
export type TripLog = {
  id: string;
  trip_date: string; // YYYY-MM-DD
  place_id: string;
  fishery: string | null;
  outcome_band: OutcomeBand | null;
  observed_conditions: string | null;
  recommendation_useful: boolean | null;
  at: string;
};

const isArr = <T,>(v: unknown): v is T[] => Array.isArray(v);

export function readFeedback(): FeedbackRecord[] {
  return readJSON<FeedbackRecord[]>(FEEDBACK_KEY, [], isArr).filter((f) => f && typeof f.fact_id === "string");
}

/** One answer per fact: a new answer replaces the earlier one. */
export function saveFeedback(r: FeedbackRecord): boolean {
  return writeJSON(FEEDBACK_KEY, [r, ...readFeedback().filter((f) => f.fact_id !== r.fact_id)].slice(0, 200));
}

export function feedbackFor(factId: string): FeedbackRecord | null {
  return readFeedback().find((f) => f.fact_id === factId) ?? null;
}

export function readTrips(): TripLog[] {
  return readJSON<TripLog[]>(TRIPS_KEY, [], isArr).filter((t) => t && typeof t.id === "string" && typeof t.trip_date === "string");
}

export function saveTrip(t: TripLog): boolean {
  return writeJSON(TRIPS_KEY, [t, ...readTrips().filter((x) => x.id !== t.id)].slice(0, 100));
}

export function deleteTrip(id: string): boolean {
  return writeJSON(TRIPS_KEY, readTrips().filter((t) => t.id !== id));
}
