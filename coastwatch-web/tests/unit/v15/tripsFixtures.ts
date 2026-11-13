import type { KV } from "@/lib/tripEconomicsStore";
import { tripFishery } from "@/nx/coast/fisheries";
import { emptyWorkspace, toDraft, type Workspace } from "@/nx/coast/workspace";
import { blankTrip } from "@/nx/trips/store";
import type { Snapshot, Trip } from "@/nx/trips/types";

/** A Map-backed Storage. `quota` (bytes over all values) makes setItem throw like a full browser storage. */
export function memory(o: { quota?: number } = {}): KV & { map: Map<string, string>; quota: number | undefined } {
  const map = new Map<string, string>();
  const size = () => [...map.values()].reduce((a, v) => a + v.length, 0);
  const kv = {
    map,
    quota: o.quota,
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => {
      const next = size() - (map.get(k)?.length ?? 0) + String(v).length;
      if (kv.quota !== undefined && next > kv.quota) throw new DOMException("full", "QuotaExceededError");
      map.set(k, String(v));
    },
    removeItem: (k: string) => void map.delete(k),
    key: (i: number) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
  return kv;
}

export const crab = tripFishery("dungeness_commercial")!;
export const sport = tripFishery("dungeness_recreational")!;
export const MOSS = "port-592";

/** The owner's test trip: 400 lb at $9, 30 gal at $8.24, bait and ice $100, 1 crew on 20 % after costs. */
export const ownerWs = (o: Partial<Workspace> = {}): Workspace => ({
  ...emptyWorkspace(crab),
  catch: [{ label: "Dungeness crab", ref: "dungeness", amount: "400", price: "9" }],
  fuelGal: "30",
  fuelPrice: "8.24",
  fuelPayer: "shared",
  costs: [{ label: "Bait and ice", usd: "100", payer: "shared" }],
  feesPct: "0",
  crew: "1",
  crewKind: "afterExpenses",
  share: "20",
  ...o,
});

export const clockAt = (iso: string) => {
  let t = Date.parse(iso);
  return () => new Date((t += 1000));
};

export function ownerTrip(o: Partial<Trip> = {}): Trip {
  return blankTrip({ id: "t-owner", fishery: "dungeness_commercial", placeKey: MOSS, now: "2026-11-01T08:00:00.000Z", name: "Moss Landing crab", economics: toDraft(ownerWs(), crab), state: "planned", depart: { date: "2026-11-20", time: "05:00", timeZone: "America/Los_Angeles" }, ...o });
}

export function snap(id: string, tripId: string, kind: Snapshot["kind"] = "recheck", takenAt = "2026-11-02T08:00:00.000Z"): Snapshot {
  const read = { state: "ok" as const, at: takenAt, detail: null };
  return {
    schema: 1,
    id,
    tripId,
    kind,
    takenAt,
    serverNow: takenAt,
    mode: "live",
    savedFrom: null,
    sources: { season: read, whale: read, cdphTest: read, nwsForecast: read, nwsAlerts: read, buoy: read, tides: read },
    evidence: [],
    forecast: null,
    tides: null,
    assumptions: { limits: { windKt: 20, seasFt: 8 }, depart: null, economics: null },
    limitations: [],
  };
}
