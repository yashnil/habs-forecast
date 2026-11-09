import type { BiotoxinIndex, FisheriesStatusIndex } from "@/pilot/contracts";

/**
 * TEMPORARY (Today track): minimal checks of the two pilot sidecars, used until the data track's
 * loaders (src/pilot/data/**, branch pilot/data) are merged. Anything malformed is null (the
 * page shows that section as unavailable), never half-shown.
 */
const isObj = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x);
const isStr = (x: unknown): x is string => typeof x === "string";

export function parseBiotoxin(raw: unknown): BiotoxinIndex | null {
  if (!isObj(raw) || raw.version !== 1 || !Array.isArray(raw.sites) || !isObj(raw.source) || !Array.isArray(raw.action_levels)) return null;
  const sites = raw.sites.filter((s): s is BiotoxinIndex["sites"][number] => isObj(s) && isStr(s.id) && isStr(s.species) && Array.isArray(s.sets) && Array.isArray(s.informs_places));
  return { ...(raw as unknown as BiotoxinIndex), sites: sites.map((s) => ({ ...s, sets: s.sets.filter((x) => isObj(x) && isStr(x.date) && Array.isArray(x.values_ppm)) })) };
}

const LEVELS = new Set(["closed", "delayed", "restricted", "watch", "no_restriction_on_record"]);
export function parseFisheriesStatus(raw: unknown): FisheriesStatusIndex | null {
  if (!isObj(raw) || raw.version !== 1 || !Array.isArray(raw.records)) return null;
  const records = raw.records.filter((r): r is FisheriesStatusIndex["records"][number] => isObj(r) && isStr(r.id) && isStr(r.fishery) && LEVELS.has(r.level as string) && isStr(r.headline) && isStr(r.checked_at) && Array.isArray(r.applies_to_places));
  const ramp = isObj(raw.ramp) && Array.isArray(raw.ramp.zones) ? (raw.ramp as unknown as FisheriesStatusIndex["ramp"]) : null;
  return { version: 1, generated_at: isStr(raw.generated_at) ? raw.generated_at : "", records, ramp };
}
