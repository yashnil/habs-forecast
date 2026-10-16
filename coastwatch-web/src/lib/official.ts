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
  const disputed = disputes(ds);
  for (const d of disputed) reasons.push(`${d.agency} release ${d.release} appears to lift a notice the list still shows as active; awaiting a person's review.`);
  if (ds.conflicts.length > disputed.length) reasons.push("Some records contradict each other.");
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

// ---------------------------------------------------------------- source discrepancies

/**
 * The pipeline publishes, among `conflicts`, agency releases that appear to lift a record the
 * registry still lists as active (pipeline: official.reported_lifts). Until a person reviews the
 * release and updates the registry, such a record is "disputed": never presented as confirmed in
 * effect, never counted as a current notice, and always shown with the release that disputes it.
 */
export const DISCREPANCY_PREFIX = "Agency source discrepancy:";
export type Dispute = { recordId: string; agency: string; release: string; title: string; url: string };

export function disputes(ds: OfficialDataset | null): Dispute[] {
  if (!ds) return [];
  const out: Dispute[] = [];
  for (const c of ds.conflicts) {
    if (!c.startsWith(DISCREPANCY_PREFIX)) continue;
    const m = c.match(/^Agency source discrepancy: (\w+) release (\S+) \("(.*)", (https:\/\/\S+)\) appears to lift active record ([a-z0-9-]+)\./);
    if (m) out.push({ agency: m[1], release: m[2], title: m[3], url: m[4], recordId: m[5] });
  }
  return out;
}

export function disputeOf(ds: OfficialDataset | null, id: string): Dispute | null {
  return disputes(ds).find((d) => d.recordId === id) ?? null;
}

/** Plain-language line for a disputed record. */
export function disputeText(d: Dispute): string {
  return `${d.agency} release ${d.release} (“${d.title}”) appears to lift this notice. Neritic has not reviewed it yet, so it is not shown as in effect. Read the release.`;
}

/** Active records that no published discrepancy disputes: the notices shown as current. */
export function currentRecords(ds: OfficialDataset | null): OfficialRecord[] {
  const ids = new Set(disputes(ds).map((d) => d.recordId));
  return activeRecords(ds).filter((r) => !ids.has(r.id));
}

/** Active records that an agency release appears to lift (awaiting review). */
export function disputedRecords(ds: OfficialDataset | null): { record: OfficialRecord; dispute: Dispute }[] {
  return disputes(ds).flatMap((d) => {
    const record = activeRecords(ds).find((r) => r.id === d.recordId);
    return record ? [{ record, dispute: d }] : [];
  });
}

/** Flags that must always be visible next to a record (never hidden behind "details"). */
export function criticalFlags(r: OfficialRecord, now: Date, dispute: Dispute | null = null): string[] {
  if (dispute) return [disputeText(dispute)];
  const today = now.toISOString().slice(0, 10);
  if (r.status === "active" && r.expected_end_date && r.expected_end_date < today) {
    return [`Expected end date (${r.expected_end_date}) has passed, but no lifting notice is recorded. Treat as in effect until the agency confirms.`];
  }
  return [];
}

/** All flags for a record: critical ones plus documented uncertainties. */
export function recordFlags(r: OfficialRecord, now: Date, dispute: Dispute | null = null): string[] {
  const flags = criticalFlags(r, now, dispute);
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
