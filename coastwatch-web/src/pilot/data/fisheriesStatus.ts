import type { FisheriesStatusIndex, FisheryId, FisheryStatus, RampState, StatusLevel } from "@/pilot/contracts";
import { cachedLoader, dayOrNull, isNum, isObj, isStr, numOrNull, oneOf, strList, strOrNull } from "./validate";

/**
 * `${dataBase}/fisheries-status/index.json`: season, closure, restriction and whale-entanglement
 * (RAMP) status transcribed from CDFW/CDPH pages into reviewed YAML (data/curated/fisheries-status;
 * pipeline/coastwatch_pipeline/sources/fisheries_status.py). Nothing here expires on its own.
 *
 * Review: a record is "human_verified" only if the pipeline says so AND names the reviewer; anything
 * else is treated as pending, so the web shows "Not verified". Missing or invalid file -> null; a
 * malformed record is dropped, never half-shown.
 */

export type FisheryStatusExt = FisheryStatus & { reviewed_by?: string | null; reviewed_at?: string | null; other_sources?: string[] };
export type RampZoneExt = RampState["zones"][number] & { lat_north?: number | null; lat_south?: number | null; applies_to_places?: string[] };
export type RampStateExt = Omit<RampState, "zones"> & { zones: RampZoneExt[]; next_assessment_window?: string | null; checked_at?: string | null; zone_source_url?: string | null; reviewed_by?: string | null };
export type FisheriesStatusIndexExt = Omit<FisheriesStatusIndex, "records" | "ramp"> & { records: FisheryStatusExt[]; ramp: RampStateExt | null };

const FISHERIES: readonly FisheryId[] = ["dungeness_commercial", "dungeness_recreational", "rock_crab", "salmon_commercial", "groundfish", "anchovy_sardine", "market_squid", "bivalve_shellfish", "spiny_lobster", "other"];
const LEVELS: readonly StatusLevel[] = ["closed", "delayed", "restricted", "watch", "no_restriction_on_record"];
const CAUSES = ["whale_entanglement", "domoic_acid", "psp", "regulation", "season", "quality", "other"] as const;
const places = (x: unknown): x is string[] => Array.isArray(x) && (x.length === 0 || strList(x));

/** "human_verified" needs a named reviewer; otherwise pending (never upgraded by the web). */
const reviewState = (state: unknown, by: unknown): FisheryStatus["review_state"] => (state === "human_verified" && isStr(by) ? "human_verified" : "pending_human_review");

function record(x: unknown): FisheryStatusExt | null {
  if (!isObj(x) || !isStr(x.id) || !oneOf(x.fishery, FISHERIES) || !isStr(x.label) || !oneOf(x.level, LEVELS) || !oneOf(x.cause, CAUSES)) return null;
  if (!isStr(x.headline) || typeof x.detail !== "string" || !isStr(x.area_label) || !places(x.applies_to_places)) return null;
  if (!dayOrNull(x.effective) || !dayOrNull(x.until) || !dayOrNull(x.next_review) || !isStr(x.source_name) || !isStr(x.source_url) || !isStr(x.checked_at)) return null;
  if (x.review_state !== "pending_human_review" && x.review_state !== "human_verified") return null;
  return {
    id: x.id, fishery: x.fishery, label: x.label, level: x.level, cause: x.cause, headline: x.headline, detail: x.detail,
    area_label: x.area_label, applies_to_places: x.applies_to_places, effective: x.effective, until: x.until, next_review: x.next_review,
    source_name: x.source_name, source_url: x.source_url, checked_at: x.checked_at,
    review_state: reviewState(x.review_state, x.reviewed_by),
    reviewed_by: isStr(x.reviewed_by) ? x.reviewed_by : null,
    reviewed_at: isStr(x.reviewed_at) ? x.reviewed_at : null,
    other_sources: Array.isArray(x.other_sources) ? x.other_sources.filter(isStr) : [],
  };
}

function ramp(x: unknown): RampStateExt | null {
  if (!isObj(x) || !isStr(x.season) || !Array.isArray(x.zones) || !dayOrNull(x.last_assessment) || !dayOrNull(x.next_assessment) || !isStr(x.source_url)) return null;
  const zones = x.zones
    .filter((z): z is Record<string, unknown> => isObj(z) && isStr(z.zone) && isStr(z.label) && isStr(z.action) && numOrNull(z.depth_limit_fathoms) && numOrNull(z.gear_reduction_pct))
    .map((z) => ({
      zone: z.zone as string, label: z.label as string, action: z.action as string,
      depth_limit_fathoms: z.depth_limit_fathoms as number | null, gear_reduction_pct: z.gear_reduction_pct as number | null,
      lat_north: isNum(z.lat_north) ? z.lat_north : null, lat_south: isNum(z.lat_south) ? z.lat_south : null,
      applies_to_places: places(z.applies_to_places) ? z.applies_to_places : [],
    }));
  return {
    season: x.season, zones, last_assessment: x.last_assessment, next_assessment: x.next_assessment, source_url: x.source_url,
    review_state: reviewState(x.review_state, x.reviewed_by),
    next_assessment_window: strOrNull(x.next_assessment_window) ? x.next_assessment_window : null,
    checked_at: isStr(x.checked_at) ? x.checked_at : null,
    zone_source_url: isStr(x.zone_source_url) ? x.zone_source_url : null,
    reviewed_by: isStr(x.reviewed_by) ? x.reviewed_by : null,
  };
}

/** The index as published, or null when it is not a version-1 FisheriesStatusIndex. */
export function parseFisheriesStatus(j: unknown): FisheriesStatusIndexExt | null {
  if (!isObj(j) || j.version !== 1 || !isStr(j.generated_at) || !Array.isArray(j.records)) return null;
  return {
    version: 1,
    generated_at: j.generated_at,
    records: j.records.map(record).filter((r): r is FisheryStatusExt => !!r),
    ramp: j.ramp == null ? null : ramp(j.ramp),
  };
}

/** `${dataBase}/fisheries-status/index.json`, fetched once per base; never throws (failure -> null). */
export const loadFisheriesStatus = cachedLoader("fisheries-status/index.json", parseFisheriesStatus);

/** Records that apply at a pilot place: the place's own plus statewide ([]). */
export const recordsForPlace = (idx: FisheriesStatusIndexExt | null | undefined, placeId: string): FisheryStatusExt[] =>
  (idx?.records ?? []).filter((r) => r.applies_to_places.length === 0 || r.applies_to_places.includes(placeId));

/** The RAMP zone a pilot place sits in, if the index maps it. */
export const rampZoneForPlace = (idx: FisheriesStatusIndexExt | null | undefined, placeId: string): RampZoneExt | null =>
  idx?.ramp?.zones.find((z) => z.applies_to_places?.includes(placeId)) ?? null;

/** Words for review state: the web must say "Not verified" for every pending record. */
export const reviewWords = (r: { review_state: FisheryStatus["review_state"] }) => (r.review_state === "human_verified" ? "Verified by a person" : "Not verified");
