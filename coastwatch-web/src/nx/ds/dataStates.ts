/**
 * Every state a reading can be in (v13 brief, "Every page"). One mark (shape first), one word and one legend
 * entry each; nothing here is ever drawn as a value, and a missing value is never drawn as zero.
 */
export type DataStateKind =
  | "measured"
  | "reported-zero"
  | "below-detection"
  | "estimated"
  | "forecast"
  | "historical"
  | "stale"
  | "missing"
  | "not-collected"
  | "unavailable"
  | "outside-coverage"
  | "unverified"
  | "offline"
  | "loading";

export const DATA_STATE_KINDS: DataStateKind[] = ["measured", "reported-zero", "below-detection", "estimated", "forecast", "historical", "stale", "missing", "not-collected", "unavailable", "outside-coverage", "unverified", "offline", "loading"];

/** The word each state carries, and the sentence a legend gives it. */
export const DATA_STATE_WORDS: Record<DataStateKind, { word: string; legend: string }> = {
  measured: { word: "Measured", legend: "Measured: a sample or instrument reading" },
  "reported-zero": { word: "Reported 0", legend: "Reported 0: the lab reported none" },
  "below-detection": { word: "Below detection", legend: "Below detection: under what the method can count, not zero" },
  estimated: { word: "Estimated", legend: "Estimated: worked out from measurements" },
  forecast: { word: "Forecast", legend: "Forecast or model: computed, not measured" },
  historical: { word: "Historical", legend: "Historical: an older record, not the present" },
  stale: { word: "Old", legend: "Old: past its usual update" },
  missing: { word: "Missing", legend: "Missing: expected, and not in the record" },
  "not-collected": { word: "Not collected", legend: "Not collected: this was not measured here then" },
  unavailable: { word: "Unavailable", legend: "Unavailable: the source could not be read" },
  "outside-coverage": { word: "Outside coverage", legend: "Outside coverage: the product does not reach here, not a low value" },
  unverified: { word: "Not verified", legend: "Official record, not verified by a person" },
  offline: { word: "Offline", legend: "Offline: a saved copy" },
  loading: { word: "Loading", legend: "Loading" },
};
