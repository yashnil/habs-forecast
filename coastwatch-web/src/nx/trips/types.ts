import type { Draft } from "@/lib/tripEconomicsStore";
import type { RefSnapshot } from "@/content/referenceValues";
import type { FisheryId } from "@/pilot/contracts";

/**
 * The trip as a first-class object (v15): the shared contract every part of the trip companion builds on.
 * Owned by the lead integrator; agents add their own types in their own modules and only import these.
 *
 * A trip is the reader's own plan for one departure: a port, a fishery, an intended date, their personal operating
 * limits, their economics (the worksheet's `Draft`, computed only by the engine of record, src/lib/tripEvaluation.ts),
 * the evidence Neritic showed when they planned (a planning snapshot that is never overwritten), later rechecks, and
 * afterwards what actually happened. It lives on this device only (localStorage "nx-trips-v1:"); nothing here is sent
 * to a server, and nothing in it is a verdict: no field says a trip is safe, legal, approved or profitable.
 */

export const TRIP_SCHEMA = 1;
export const SNAPSHOT_SCHEMA = 1;

/** Draft → Planned → Completed, or Cancelled from Draft/Planned. A recheck is an action, never a state. */
export type Lifecycle = "draft" | "planned" | "completed" | "cancelled";
export type Activity = "commercial" | "recreational";

/** The reader's own thresholds. Personal, never certified or recommended limits; blank is "none set", never 0. */
export type OperatingLimits = { windKt: number | null; seasFt: number | null };

/** Intended departure, in the port's zone. `time` null: "some time that day". */
export type Departure = { date: string; time: string | null; timeZone: string };

export type Trip = {
  schema: typeof TRIP_SCHEMA;
  id: string;
  name: string;
  /** the name was written by Neritic and never edited: it may follow the trip's details */
  autoName: boolean;
  activity: Activity;
  fishery: FisheryId;
  /** home port (a port key, "port-592") */
  placeKey: string;
  /** departure port when it differs from home; null = the home port */
  departKey: string | null;
  depart: Departure | null;
  /** optional free text, approximate by design ("outside the bay, 20–40 fm"); never coordinates */
  area: string | null;
  vesselId: string | null;
  limits: OperatingLimits;
  /** the worksheet as the engine reads it; null when the reader modelled no costs */
  economics: Draft | null;
  /** references the reader took into the worksheet (CDFW price, EIA diesel), kept with their dates */
  refs: RefSnapshot[];
  /** earlier economics, newest first, kept when the plan changes (bounded; see store) */
  history: { at: string; economics: Draft; refs: RefSnapshot[] }[];
  notes: string;
  state: Lifecycle;
  createdAt: string;
  updatedAt: string;
  /** snapshot ids: the planning snapshot (first save or an explicit refresh) and rechecks, oldest first */
  snapshots: { planning: string | null; rechecks: string[]; superseded: string[] };
  outcome: Outcome | null;
  /** the nx-trip-v1 scenario this trip was migrated from (that record is kept untouched) */
  legacyId?: string;
};

/** What happened. Every figure is the reader's own; blank is unknown, never 0. */
export type Outcome = {
  departedOn: string | null;
  returnedOn: string | null;
  /** per catch line, in pounds and $/lb (the worksheet's own units); price null for recreational */
  landed: { label: string; lb: number | null; usdPerLb: number | null }[];
  fuelGal: number | null;
  fuelUsdPerGal: number | null;
  /** bait, ice and anything else, as paid */
  costs: { label: string; usd: number | null }[];
  /** what crew were actually paid, in total */
  crewPaidUsd: number | null;
  notes: string;
  conditions: string;
  /** a landing receipt / fish ticket number, if the reader wants it here */
  landingRef: string | null;
  recordedAt: string;
  updatedAt: string;
};

// ---------------------------------------------------------------- snapshots (what Neritic showed, and when)

export type SourceKey = "season" | "whale" | "cdphTest" | "nwsForecast" | "nwsAlerts" | "buoy" | "tides";

/**
 * How one source read. "absent": it answered and nothing of that kind applies here (e.g. no CDPH test informs
 * this port). "failed": it was asked and did not answer. "notChecked": not asked (offline, or out of scope).
 * Only "ok" and "absent" are a completed check.
 */
export type SourceRead = { state: "ok" | "absent" | "failed" | "notChecked"; at: string | null; detail: string | null };

/**
 * One piece of official or measured evidence with a stable identity. `key` names *what* it is (the same notice
 * across fetches); `version` names *which version* (a hash of the fields that carry meaning, never of fetch
 * times), so a refetch of an unchanged record compares equal.
 */
export type EvidenceItem = {
  key: string;
  kind: "season" | "whale" | "cdphTest" | "advisory" | "observation";
  source: "CDFW" | "CDPH" | "NWS" | "NDBC" | "CO-OPS";
  title: string;
  /** the record's own words, as shown */
  text: string;
  url: string | null;
  /** when the source published / measured / last read it (ISO); null when the source gives none */
  asOf: string | null;
  /** where it applies, as the source states it ("Zone 3, Pigeon Pt to Lopez Pt", "PZZ535") */
  scope: string | null;
  version: string;
  fields: Record<string, string | number | boolean | null>;
  /** official records stay unverified until a person reviews them (T38) */
  verified: boolean | null;
  /** two sources disagree: said, never resolved (T40) */
  conflict: string | null;
  stale: boolean;
};

/** One NWS zone-forecast period, keyed by its valid time so only like periods are compared. */
export type ForecastPeriod = {
  /** `${date}|day` or `${date}|night` (date = the local calendar day the period starts) */
  key: string;
  name: string;
  date: string | null;
  night: boolean;
  /** derived valid interval (ISO), from the period name and the issuance; null when it cannot be derived */
  start: string | null;
  end: string | null;
  text: string;
  windDir: string | null;
  windLo: number | null;
  windHi: number | null;
  gustKt: number | null;
  seasLo: number | null;
  seasHi: number | null;
  /** swell/wave components as NWS states them ("W 6 ft at 11 seconds") */
  waves: { dir: string | null; ft: number | null; periodS: number | null }[];
};

export type ForecastSnap = { zone: string | null; issued: string | null; url: string | null; periods: ForecastPeriod[] };

export type Snapshot = {
  schema: typeof SNAPSHOT_SCHEMA;
  id: string;
  tripId: string;
  kind: "planning" | "recheck";
  /** device clock when taken */
  takenAt: string;
  /** the server's clock in its answers (a fixture pins it); null offline */
  serverNow: string | null;
  /** "live": built from fresh answers. "saved": offline, rebuilt from the newest copy on this device (never a live check) */
  mode: "live" | "saved";
  /** for "saved": when that copy was originally fetched */
  savedFrom: string | null;
  sources: Record<SourceKey, SourceRead>;
  evidence: EvidenceItem[];
  forecast: ForecastSnap | null;
  tides: { station: string | null; url: string | null; events: { at: string; ft: number; type: "H" | "L" }[] } | null;
  /** the trip's assumptions as they stood */
  assumptions: { limits: OperatingLimits; depart: Departure | null; economics: Draft | null };
  /** what the evidence does not cover, in plain words */
  limitations: string[];
};

// ---------------------------------------------------------------- what changed

export type ChangeCategory = "official" | "toxin" | "forecast" | "advisory" | "observation" | "source";
export type ChangeKind = "new" | "revised" | "removed" | "conflict" | "unavailable" | "available" | "stale" | "expired";

export type Change = {
  /** stable: the same change found twice has the same id */
  id: string;
  /** 1 most important */
  priority: 1 | 2 | 3;
  category: ChangeCategory;
  kind: ChangeKind;
  /** one descriptive sentence, sourced, never a recommendation */
  title: string;
  detail: string | null;
  before: string | null;
  after: string | null;
  where: string | null;
  /** when the change happened at the source (ISO), if known */
  when: string | null;
  url: string | null;
  /** the forecast period it concerns, when one */
  periodKey: string | null;
  evidenceKey: string | null;
};

/**
 * A recheck against the planning snapshot. `complete` only when it was live and every *required* source answered
 * (snapshot.ts REQUIRED_SOURCES: the official records, the CDPH test, NWS's forecast and advisories). An optional
 * source that failed (the buoy's reading, the tide predictions) is still listed, marked optional, and never hidden.
 */
export type Recheck = {
  at: string;
  mode: Snapshot["mode"];
  complete: boolean;
  changed: Change[];
  same: { key: string; title: string }[];
  unchecked: { source: SourceKey; why: string; required: boolean }[];
  /** records that answered but are old (a CDPH test from weeks ago, a season record not re-read lately): said, not hidden */
  stale: { key: string; title: string; asOf: string | null }[];
  verify: { label: string; url: string }[];
  baseId: string;
  snapshotId: string;
};
