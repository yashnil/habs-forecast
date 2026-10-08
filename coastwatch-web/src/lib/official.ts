import type { SourceStatus } from "@/generated/schema";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import { ageInDays } from "@/lib/freshness";

/**
 * Whether the official-notices registry may be presented as verified. Computed in the
 * browser from dates, so stale reviews lose their verified status even if the pipeline stops.
 * Anything other than "verified" must be shown as NOT verified.
 */
export type VerificationState = "verified" | "aging" | "unverified" | "unavailable";

export type Verification = {
  state: VerificationState;
  reasons: string[];
  reviewedAt: string | null;
  reviewedBy: string | null;
  reviewAgeDays: number | null;
  lastCheckedAt: string | null;
};

const WATCH_STALE_DAYS = 2;

export function officialVerification(ds: OfficialDataset | null, status: SourceStatus | null, now: Date): Verification {
  if (!ds) {
    return {
      state: "unavailable",
      reasons: ["Official notices could not be loaded."],
      reviewedAt: null,
      reviewedBy: null,
      reviewAgeDays: null,
      lastCheckedAt: null,
    };
  }
  const review = ds.registry.review;
  const reasons: string[] = [];
  const age = ageInDays(review.reviewed_at.slice(0, 10), now);
  if (review.status !== "human_verified") reasons.push("Records have not been checked by a person against the official sources yet.");
  if (age > ds.policy.aging_max_age_days) reasons.push(`Last review was ${age} days ago (more than ${ds.policy.aging_max_age_days}).`);
  for (const w of ds.watch) {
    if (!w.ok) reasons.push(`Could not read ${labelFor(ds, w.source_id)} at the last check.`);
    else if (w.new_items?.length) reasons.push(`${labelFor(ds, w.source_id)} lists items not yet reviewed: ${w.new_items.join(", ")}.`);
    else if (w.matches_review === false) reasons.push(`${labelFor(ds, w.source_id)} has changed since the last review.`);
    else if (w.matches_review == null) reasons.push(`${labelFor(ds, w.source_id)} has no reviewed fingerprint to compare against.`);
  }
  if (ds.conflicts.length) reasons.push("Some records contradict each other.");
  const lastCheckedAt = ds.watch.map((w) => w.checked_at).sort().at(-1) ?? null;
  if (lastCheckedAt && ageInDays(lastCheckedAt.slice(0, 10), now) > WATCH_STALE_DAYS) {
    reasons.push(`Official pages were last checked ${ageInDays(lastCheckedAt.slice(0, 10), now)} days ago.`);
  }
  if (status?.outcome === "failed") reasons.push("The latest update of official notices failed.");
  let state: VerificationState = "unverified";
  if (reasons.length === 0) state = age <= ds.policy.verified_max_age_days ? "verified" : "aging";
  return { state, reasons, reviewedAt: review.reviewed_at, reviewedBy: review.reviewed_by, reviewAgeDays: age, lastCheckedAt };
}

function labelFor(ds: OfficialDataset, id: string): string {
  return ds.registry.watched_sources.find((s) => s.id === id)?.label ?? id;
}

/** Flags that must always be visible next to a record (never hidden behind "details"). */
export function criticalFlags(r: OfficialRecord, now: Date): string[] {
  const today = now.toISOString().slice(0, 10);
  if (r.status === "active" && r.expected_end_date && r.expected_end_date < today) {
    return [`Expected end date (${r.expected_end_date}) has passed, but no lifting notice is recorded. Treat as in effect until the agency confirms.`];
  }
  return [];
}

/** All flags for a record: critical ones plus documented uncertainties. */
export function recordFlags(r: OfficialRecord, now: Date): string[] {
  const flags = criticalFlags(r, now);
  if (!r.effective_date && r.effective_date_note) flags.push(r.effective_date_note);
  for (const u of r.uncertainties ?? []) flags.push(u);
  return flags;
}

export const ACTION_LABEL: Record<OfficialRecord["action"], string> = {
  fishery_closure: "Fishery closure",
  take_restriction: "Take restriction",
  consumption_advisory: "Health advisory",
  quarantine: "Quarantine",
  special_advisory: "Special advisory",
  reopening: "Reopening",
  advisory_lifted: "Advisory lifted",
};

export const FISHERY_LABEL: Record<OfficialRecord["fishery"], string> = {
  commercial: "Commercial fishery",
  recreational: "Recreational fishery",
  commercial_and_recreational: "Commercial and recreational",
  consumption: "Eating / consumption",
  sport_harvest: "Sport harvest",
};

// ---------------------------------------------------------------- geometry hit-testing
type Ring = number[][];

function inRing(x: number, y: number, ring: Ring): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function inPolygon(x: number, y: number, rings: Ring[]): boolean {
  if (!inRing(x, y, rings[0])) return false;
  return !rings.slice(1).some((h) => inRing(x, y, h));
}

export function pointInGeometry(lon: number, lat: number, g: { type: string; coordinates: unknown }): boolean {
  if (g.type === "Polygon") return inPolygon(lon, lat, g.coordinates as Ring[]);
  if (g.type === "MultiPolygon") return (g.coordinates as Ring[][]).some((p) => inPolygon(lon, lat, p));
  return false;
}

export type FeatureHit = { recordIds: string[]; basis: string; note: string; kind: string };

/** Drawn official areas containing a point (drawn areas are approximate; wording controls). */
export function officialAreasAt(ds: OfficialDataset | null, lon: number, lat: number): FeatureHit[] {
  if (!ds) return [];
  const features = (ds.geometry as { features?: Array<{ geometry: { type: string; coordinates: unknown }; properties: { record_ids: string[]; basis: string; note: string; kind: string } }> }).features ?? [];
  return features
    .filter((f) => pointInGeometry(lon, lat, f.geometry))
    .map((f) => ({ recordIds: f.properties.record_ids, basis: f.properties.basis, note: f.properties.note, kind: f.properties.kind }));
}

/** Official notices that may apply at a point: statewide, drawn official polygons that
 * contain it, and latitude-defined notices whose official latitudes include it. */
export function officialAt(ds: OfficialDataset | null, lon: number, lat: number): { record: OfficialRecord; reason: string }[] {
  if (!ds) return [];
  const out: { record: OfficialRecord; reason: string }[] = [];
  const active = activeRecords(ds);
  const polyHits = new Set(officialAreasAt(ds, lon, lat).flatMap((h) => h.recordIds));
  for (const r of active) {
    if (r.area.type === "statewide") out.push({ record: r, reason: "statewide" });
    else if (polyHits.has(r.id)) out.push({ record: r, reason: r.area.type === "county" ? "inside the county outline" : "inside the official area" });
    else if (r.area.type === "lat_band" && r.area.lat_south != null && r.area.lat_north != null && lat >= r.area.lat_south && lat <= r.area.lat_north)
      out.push({ record: r, reason: "between the notice's official latitudes" });
  }
  return out;
}

export function activeRecords(ds: OfficialDataset | null): OfficialRecord[] {
  return ds ? ds.registry.records.filter((r) => r.status === "active") : [];
}
