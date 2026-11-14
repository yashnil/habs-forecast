/**
 * Neritic private pilot (2026-11): the shared contracts every pilot track builds against.
 * Owner of this file: the integrator. Tracks may ADD optional fields in their own PR notes, never
 * rename or remove. See docs/coastwatch/pilot/README.md.
 *
 * Hierarchy (non-negotiable): official status → measured tests/observations → forecasts → advice.
 * Never green "safe": the calmest state is "no_restriction_on_record". Chlorophyll is never toxin.
 * Observations are never forecasts. Curated official records stay `pending_human_review` until a
 * person runs the review; the web must say "Not verified" for them.
 */

/* ------------------------------------------------------------------ sidecar 1: biotoxin tests */
/** `${dataBase}/biotoxin/index.json` — CDPH domoic-acid results, parsed from CDPH's PDFs.
 *  Published OUTSIDE the manifest (the deployed apps reject unknown manifest fields). */
export type BiotoxinIndex = {
  version: 1;
  generated_at: string; // ISO UTC
  source: { name: string; url: string; as_of: string | null; fetched_at: string; sha256: string };
  /** CDPH action levels quoted as published, ppm (e.g. crab viscera 30, meat 20; anchovy 20) */
  action_levels: { species: string; tissue: string; ppm: number }[];
  sites: BiotoxinSite[];
  /** rows the parser could not read; the web never invents them */
  parse_warnings: string[];
};
export type BiotoxinSite = {
  id: string; // stable slug, e.g. "bma-g-monterey-bay-dungeness"
  bma: string | null; // "G"
  bma_label: string | null; // "Monterey Bay"
  port_label: string | null;
  site_label: string; // CDPH collection site, as written
  species: string; // "Dungeness crab", "Rock crab", "Northern anchovy", ...
  /** approximate site location if CDPH names a known place; null otherwise (never guessed) */
  lat: number | null;
  lon: number | null;
  /** nearest pilot/port place ids this site informs (e.g. ["port-550","port-592"]) */
  informs_places: string[];
  sets: BiotoxinSet[]; // newest first
};
export type BiotoxinSet = {
  date: string; // collection date, Pacific YYYY-MM-DD
  kind: "routine" | "retest" | "unknown";
  tissue: "viscera" | "meat" | "whole" | "unknown";
  values_ppm: (number | null)[]; // individual samples as published (e.g. 6 crabs); null = "ND"/unreadable
  nd_count: number; // samples reported below detection
  avg_ppm: number | null;
  max_ppm: number | null;
  pct_over_action: number | null; // as published
  action_level_ppm: number | null;
};

/* --------------------------------------------- sidecar 2: fisheries status (curated, reviewed) */
/** `${dataBase}/fisheries-status/index.json` — season, closure, restriction and whale (RAMP) status,
 *  transcribed from CDFW/CDPH pages into PR-reviewed YAML. Never auto-expires; never auto-verifies. */
export type FisheriesStatusIndex = {
  version: 1;
  generated_at: string;
  records: FisheryStatus[];
  ramp: RampState | null;
};
export type FisheryId =
  | "dungeness_commercial"
  | "dungeness_recreational"
  | "rock_crab"
  | "salmon_commercial"
  | "groundfish"
  | "anchovy_sardine"
  | "market_squid"
  | "bivalve_shellfish"
  | "spiny_lobster"
  | "other";
export type StatusLevel =
  | "closed" // red
  | "delayed" // red
  | "restricted" // amber: open with an official restriction (depth, gear, consumption, area)
  | "watch" // amber: an official process is pending (assessment due, retest, advisory)
  | "no_restriction_on_record"; // neutral — NOT "safe", NOT "open is guaranteed"
export type FisheryStatus = {
  id: string; // stable slug
  fishery: FisheryId;
  label: string; // "Commercial Dungeness crab"
  level: StatusLevel;
  cause: "whale_entanglement" | "domoic_acid" | "psp" | "regulation" | "season" | "quality" | "other";
  headline: string; // one plain sentence, <= 110 chars
  detail: string; // what it means / what is next
  area_label: string; // "Oregon border to Point Conception (Zones 1–5)"
  /** pilot place ids this record applies to; [] = statewide, applies everywhere */
  applies_to_places: string[];
  effective: string | null; // YYYY-MM-DD
  until: string | null; // "at least" dates are written in `detail`, never auto-expire
  next_review: string | null; // e.g. next RAMP assessment
  source_name: string; // "CDFW"
  source_url: string;
  checked_at: string; // when a human/assistant last read the source
  review_state: "pending_human_review" | "human_verified";
};
export type RampState = {
  season: string; // "2026-27"
  zones: { zone: string; label: string; action: string; depth_limit_fathoms: number | null; gear_reduction_pct: number | null }[];
  last_assessment: string | null;
  next_assessment: string | null;
  source_url: string;
  review_state: "pending_human_review" | "human_verified";
};

/* --------------------------------------------------------------- the Today model (port facts) */
/** A single statement the Today page shows and that alerts and Ask Neritic reuse. Built by
 *  src/pilot/facts.ts (briefing track) from published data only. */
export type FactKind = "status" | "test" | "sea" | "weather" | "tide" | "bloom_forecast" | "bloom_outlook" | "observation" | "whale" | "notice" | "data_age";
export type Severity = "red" | "amber" | "neutral" | "info";
export type PortFact = {
  id: string; // stable across refreshes for the same subject (used for "what changed" + alert dedupe)
  place_id: string; // "port-592"
  kind: FactKind;
  severity: Severity;
  headline: string; // plain language first
  value?: string | null; // the number, with units
  detail?: string | null;
  /** what this is, epistemically */
  basis: "official" | "measured" | "forecast" | "outlook" | "analysis" | "derived";
  verified: boolean | null; // official records only; null for non-official
  source: { name: string; url: string | null };
  as_of: string | null; // ISO or YYYY-MM-DD of the underlying data
  valid_for?: { from: string; to: string } | null; // forecasts/outlooks
};
export type PortToday = { place_id: string; label: string; generated_at: string; data_generated_at: string | null; facts: PortFact[] };

/** Pilot ports (first wave). Ids match lib/coast place ids (`port-<port_code>`). */
export const PILOT_PORTS = [
  { place_id: "port-592", label: "Moss Landing", nws_zone: "PZZ535", tide_station: "9413450", buoy: "46042" },
  { place_id: "port-550", label: "Monterey", nws_zone: "PZZ535", tide_station: "9413450", buoy: "46042" },
  { place_id: "port-593", label: "Santa Cruz", nws_zone: "PZZ535", tide_station: "9413745", buoy: "46042" },
  { place_id: "port-452", label: "Princeton / Half Moon Bay", nws_zone: "PZZ545", tide_station: "9414131", buoy: "46012" },
] as const;

/* ----------------------------------------------------------------- readiness (honest labels) */
export type Readiness = "live" | "demo" | "not_configured";
export type PilotFeature = "accounts" | "alerts" | "web_push" | "ask" | "sms";
