import { describe, expect, it } from "vitest";
import { createTripStore, NS } from "@/lib/tripEconomicsStore";
import { toDraft } from "@/nx/coast/workspace";
import { migrateLegacy } from "@/nx/trips/migrate";
import { planVsActual } from "@/nx/trips/outcome";
import { createTripsStore, TRIPS_NS } from "@/nx/trips/store";
import { clockAt, crab, memory, MOSS, ownerWs } from "./tripsFixtures";

const T0 = "2026-11-13T08:00:00.000Z";

function seeded() {
  const kv = memory();
  const legacy = createTripStore(kv, clockAt("2026-11-01T08:00:00.000Z"));
  legacy.saveScenario({ name: "Moss crab, 400 lb at $9", autoName: true, placeKey: MOSS, species: "dungeness_commercial", draft: toDraft(ownerWs(), crab), refs: [] });
  legacy.saveScenario({ name: "Old v4 salmon", placeKey: "port-550", species: "salmon", draft: toDraft(ownerWs(), crab), refs: [] });
  legacy.saveScenario({ name: "No fishery", placeKey: MOSS, species: "", draft: toDraft(ownerWs(), crab), refs: [] });
  const legacyKeys = () => new Map([...kv.map].filter(([k]) => k.startsWith(NS)));
  return { kv, legacyKeys };
}

describe("legacy scenarios copied into trips", () => {
  it("copies each scenario as a Draft trip (name, port, fishery, economics, references kept, legacyId set)", () => {
    const { kv } = seeded();
    const store = createTripsStore(kv, clockAt(T0));
    const r = migrateLegacy(kv, store, clockAt(T0));
    expect(r.legacy).toBe("read");
    expect(r.copied).toHaveLength(2);
    expect(r.skipped).toEqual([{ legacyId: expect.any(String), why: "it has no fishery" }]);
    const trips = store.list();
    const moss = trips.find((t) => t.name === "Moss crab, 400 lb at $9")!;
    expect(moss).toMatchObject({ state: "draft", depart: null, fishery: "dungeness_commercial", placeKey: MOSS, activity: "commercial", autoName: true });
    expect(moss.legacyId).toMatch(/^s-/);
    expect(planVsActual(moss).planned.result.value).toBeCloseTo(2602.24, 9);
    expect(trips.find((t) => t.name === "Old v4 salmon")?.fishery).toBe("salmon_commercial");
  });

  it("never writes the nx-trip-v1 keys", () => {
    const { kv, legacyKeys } = seeded();
    const before = legacyKeys();
    migrateLegacy(kv, createTripsStore(kv, clockAt(T0)), clockAt(T0));
    expect(legacyKeys()).toEqual(before);
  });

  it("is idempotent: a second run copies nothing, even after the reader deleted a copied trip", () => {
    const { kv } = seeded();
    const store = createTripsStore(kv, clockAt(T0));
    migrateLegacy(kv, store, clockAt(T0));
    const again = migrateLegacy(kv, store, clockAt(T0));
    expect(again.copied).toEqual([]);
    expect(again.already).toBe(2);
    const first = store.list()[0];
    store.remove(first.id);
    expect(migrateLegacy(kv, store, clockAt(T0)).copied).toEqual([]);
    expect(store.list()).toHaveLength(1);
    // the marker survives "delete all trips", so deleted trips do not return from the old store
    store.deleteAll();
    expect(migrateLegacy(kv, store, clockAt(T0)).copied).toEqual([]);
    expect(store.list()).toEqual([]);
  });

  it("a trip carrying the legacyId counts as copied even without the marker", () => {
    const { kv } = seeded();
    const store = createTripsStore(kv, clockAt(T0));
    migrateLegacy(kv, store, clockAt(T0));
    kv.map.delete(`${TRIPS_NS}migrated`);
    expect(migrateLegacy(kv, store, clockAt(T0)).copied).toEqual([]);
  });

  it("a newer or corrupt legacy store is skipped and said; nothing is written", () => {
    for (const [text, want] of [
      [JSON.stringify([{ schema: 9, id: "s-x" }]), "newer"],
      ["{broken", "corrupt"],
    ] as const) {
      const kv = memory();
      kv.map.set(`${NS}scenarios`, text);
      const store = createTripsStore(kv, clockAt(T0));
      const r = migrateLegacy(kv, store, clockAt(T0));
      expect(r.legacy).toBe(want);
      expect(r.copied).toEqual([]);
      expect([...kv.map.keys()]).toEqual([`${NS}scenarios`]);
      expect(kv.map.get(`${NS}scenarios`)).toBe(text);
    }
  });

  it("no legacy store: nothing to do", () => {
    const kv = memory();
    expect(migrateLegacy(kv, createTripsStore(kv), clockAt(T0)).legacy).toBe("absent");
    expect(kv.map.size).toBe(0);
  });

  it("a full trips store stops the copy and says so; the next run finishes it", () => {
    const { kv } = seeded();
    const store = createTripsStore(kv, clockAt(T0));
    const used = [...kv.map.values()].reduce((a, x) => a + x.length, 0);
    kv.quota = used + 10;
    const r = migrateLegacy(kv, store, clockAt(T0));
    expect(r.failed).toBe("quota");
    expect(r.copied).toEqual([]);
    kv.quota = undefined;
    expect(migrateLegacy(kv, store, clockAt(T0)).copied).toHaveLength(2);
  });
});
