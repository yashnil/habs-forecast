import { createTripStore, type KV, type ScenarioRecord } from "@/lib/tripEconomicsStore";
import { audienceOf, tripFishery } from "@/nx/coast/fisheries";
import type { FisheryId } from "@/pilot/contracts";
import { blankTrip, type TripsStore } from "./store";
import type { Trip } from "./types";

/**
 * Saved worksheet scenarios (nx-trip-v1, My Coast before v15) copied into trips. Conservative by design:
 * - copy only: the nx-trip-v1 keys are read, never written or removed, so rolling back to the earlier version finds
 *   every scenario where it was;
 * - once: each scenario copied is recorded (nx-trips-v1:migrated, ids only) and any trip already carrying its
 *   `legacyId` counts too, so it is never copied twice, even after the reader deletes the copy;
 * - a scenario becomes a Draft trip (it never had a date), its name, port, fishery, worksheet and references kept;
 * - a legacy store written by a newer version, or one whose list does not parse, is skipped and said.
 */

export const MIGRATED_KEY = "migrated";

export type MigrationReport = {
  /** what happened to the old store as a whole */
  legacy: "absent" | "read" | "newer" | "corrupt" | "unavailable";
  copied: { legacyId: string; tripId: string; name: string }[];
  /** already copied before (by marker or by a trip's legacyId) */
  already: number;
  skipped: { legacyId: string | null; why: string }[];
  /** the trips store could not be written (full, or a newer version is in charge) */
  failed: "quota" | "readOnly" | "unavailable" | null;
};

/** Scenarios saved by the v4 worksheet named a species, not a fishery. Only exact counterparts are mapped. */
const SPECIES_TO_FISHERY: Record<string, FisheryId> = {
  dungeness: "dungeness_commercial",
  salmon: "salmon_commercial",
  "salmon-troll": "salmon_commercial",
  groundfish: "groundfish",
  "groundfish-hl": "groundfish",
  squid: "market_squid",
  "rock-crab": "rock_crab",
  lobster: "spiny_lobster",
};

export function legacyFishery(species: string): FisheryId | null {
  if (tripFishery(species)) return species as FisheryId;
  return SPECIES_TO_FISHERY[species] ?? null;
}

export function scenarioToTrip(s: ScenarioRecord, id: string, fishery: FisheryId): Trip {
  return blankTrip({
    id,
    fishery,
    placeKey: s.placeKey,
    now: s.createdAt,
    name: s.name,
    autoName: s.autoName === true,
    activity: audienceOf(tripFishery(fishery)),
    economics: s.draft,
    refs: s.refs,
    state: "draft",
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    legacyId: s.id,
  });
}

function readMarker(store: TripsStore): string[] {
  const r = store.readKey(MIGRATED_KEY);
  const p = r.parsed as { ids?: unknown } | null;
  return p && Array.isArray(p.ids) ? p.ids.filter((x): x is string => typeof x === "string") : [];
}

/** Copy every legacy scenario not yet copied. Safe to run on every page load. */
export function migrateLegacy(kv: KV | null, store: TripsStore, clock: () => Date = () => new Date()): MigrationReport {
  const report: MigrationReport = { legacy: "absent", copied: [], already: 0, skipped: [], failed: null };
  if (!kv) return { ...report, legacy: "unavailable" };
  const legacy = createTripStore(kv, clock).load();
  if (legacy.status.readOnly) return { ...report, legacy: "newer" };
  if (legacy.status.corrupt.includes("scenarios")) return { ...report, legacy: "corrupt" };
  if (!legacy.scenarios.length) return { ...report, legacy: legacy.status.invalidRecords ? "read" : "absent", skipped: legacy.status.invalidRecords ? [{ legacyId: null, why: `${legacy.status.invalidRecords} saved scenario(s) could not be read and stay in the old store` }] : [] };
  report.legacy = "read";
  if (legacy.status.invalidRecords) report.skipped.push({ legacyId: null, why: `${legacy.status.invalidRecords} saved scenario(s) could not be read and stay in the old store` });

  const now = store.load();
  if (!now.status.available) return { ...report, failed: "unavailable" };
  if (now.status.readOnly) return { ...report, failed: "readOnly" };
  const done = new Set([...readMarker(store), ...now.trips.map((t) => t.legacyId).filter((x): x is string => !!x)]);
  const taken = new Set(now.trips.map((t) => t.id));
  const marked: string[] = [...done];

  // oldest first, so the trips list keeps the scenarios' own order by updatedAt
  for (const s of [...legacy.scenarios].reverse()) {
    if (done.has(s.id)) {
      report.already += 1;
      continue;
    }
    const fishery = legacyFishery(s.species);
    if (!fishery) {
      report.skipped.push({ legacyId: s.id, why: s.species ? `its fishery ("${s.species}") is not one a trip can be planned for` : "it has no fishery" });
      continue;
    }
    let id = `t-legacy-${s.id}`.slice(0, 60);
    while (taken.has(id)) id = `${id.slice(0, 54)}-${Math.random().toString(36).slice(2, 6)}`;
    const r = store.restore(scenarioToTrip(s, id, fishery));
    if (!r.ok) {
      report.failed = r.reason === "readOnly" ? "readOnly" : r.reason === "unavailable" ? "unavailable" : "quota";
      break;
    }
    taken.add(id);
    marked.push(s.id);
    report.copied.push({ legacyId: s.id, tripId: id, name: s.name });
  }
  if (report.copied.length) {
    const w = store.writeKey(MIGRATED_KEY, { schema: 1, ids: marked, at: clock().toISOString() });
    if (!w.ok) report.failed = report.failed ?? "quota";
  }
  return report;
}
