import { describe, expect, it } from "vitest";
import { copyName, duplicateTrip, groupTrips, MIN_FOR_COMPARISON, ownRecords, renameTrip, searchTrips } from "@/nx/trips/history";
import { createTripsStore } from "@/nx/trips/store";
import type { Outcome, Trip } from "@/nx/trips/types";
import { clockAt, memory, ownerTrip, snap } from "./tripsFixtures";

const NOW = new Date("2026-11-13T18:00:00.000Z"); // 10:00 in California
const day = (date: string) => ({ date, time: null, timeZone: "America/Los_Angeles" });
const ports: Record<string, string> = { "port-592": "Moss Landing", "port-550": "Monterey", "port-600": "Half Moon Bay" };
const portName = (k: string) => ports[k] ?? null;

const landed = (lb: number, usdPerLb = 9, crew = 650.56): Outcome => ({ departedOn: null, returnedOn: null, landed: [{ label: "Dungeness crab", lb, usdPerLb }], fuelGal: 30, fuelUsdPerGal: 8.24, costs: [{ label: "Bait and ice", usd: 100 }], crewPaidUsd: crew, notes: "", conditions: "", landingRef: null, recordedAt: "2026-11-01T00:00:00.000Z", updatedAt: "2026-11-01T00:00:00.000Z" });

describe("grouping", () => {
  it("upcoming (soonest first), planned with the date passed, drafts, completed, cancelled", () => {
    const trips: Trip[] = [
      ownerTrip({ id: "a", state: "planned", depart: day("2026-11-20") }),
      ownerTrip({ id: "b", state: "planned", depart: day("2026-11-13") }),
      ownerTrip({ id: "c", state: "planned", depart: day("2026-11-12") }),
      ownerTrip({ id: "d", state: "draft", depart: null }),
      ownerTrip({ id: "e", state: "completed", outcome: landed(400) }),
      ownerTrip({ id: "f", state: "cancelled" }),
      ownerTrip({ id: "g", state: "planned", depart: null }),
    ];
    const g = groupTrips(trips, NOW);
    expect(g.map((x) => [x.id, x.trips.map((t) => t.id)])).toEqual([
      ["upcoming", ["b", "a", "g"]],
      ["pastDue", ["c"]],
      ["drafts", ["d"]],
      ["completed", ["e"]],
      ["cancelled", ["f"]],
    ]);
  });

  it("today is the port's calendar day, not UTC's", () => {
    // 2026-11-14T02:00Z is still 13 November in California
    const g = groupTrips([ownerTrip({ id: "x", state: "planned", depart: day("2026-11-13") })], new Date("2026-11-14T02:00:00.000Z"));
    expect(g[0].id).toBe("upcoming");
  });
});

describe("search, duplicate, rename", () => {
  const trips = [ownerTrip({ id: "a", name: "Crab run" }), ownerTrip({ id: "b", name: "Squid", fishery: "market_squid", placeKey: "port-550" }), ownerTrip({ id: "c", name: "Other", placeKey: "port-600", departKey: "port-592" })];
  it("matches port names (home or departure), fishery and name, accents and case aside", () => {
    expect(searchTrips(trips, "moss", portName).map((t) => t.id)).toEqual(["a", "c"]);
    expect(searchTrips(trips, "squid monterey", portName).map((t) => t.id)).toEqual(["b"]);
    expect(searchTrips(trips, "DUNGENESS", portName).map((t) => t.id)).toEqual(["a", "c"]);
    expect(searchTrips(trips, "  ", portName)).toHaveLength(3);
  });

  it("duplicate: a new id, a draft with no date, outcome, snapshots or history", () => {
    const s = createTripsStore(memory(), clockAt("2026-11-13T08:00:00.000Z"));
    s.put(ownerTrip({ state: "completed", outcome: landed(300), legacyId: "s-1" }));
    s.putSnapshot(snap("p1", "t-owner", "planning"));
    const r = duplicateTrip(s, "t-owner", NOW);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.trip.id).not.toBe("t-owner");
    expect(r.trip).toMatchObject({ state: "draft", depart: null, outcome: null, history: [], snapshots: { planning: null, rechecks: [], superseded: [] }, name: "Moss Landing crab (copy)" });
    expect(r.trip.legacyId).toBeUndefined();
    expect(r.trip.economics).toEqual(ownerTrip().economics);
    expect(s.get("t-owner")?.snapshots.planning).toBe("p1");
    expect(copyName("Moss Landing crab (copy)", ["Moss Landing crab (copy)"])).toBe("Moss Landing crab (copy 2)");
  });

  it("rename: the reader's name is kept and no longer automatic; a blank keeps the old name", () => {
    const s = createTripsStore(memory(), clockAt("2026-11-13T08:00:00.000Z"));
    s.put(ownerTrip({ autoName: true }));
    const r = renameTrip(s, "t-owner", "  Opening day  ");
    expect(r.ok && r.trip.name).toBe("Opening day");
    expect(r.ok && r.trip.autoName).toBe(false);
    expect(renameTrip(s, "t-owner", " ").ok && s.get("t-owner")?.name).toBe("Opening day");
  });
});

describe("own-records comparisons", () => {
  const done = (id: string, lb: number) => ownerTrip({ id, state: "completed", outcome: landed(lb) });
  it("nothing below three completed trips with recorded figures", () => {
    const r = ownRecords([done("a", 300), done("b", 500), ownerTrip({ id: "c" })]);
    expect(MIN_FOR_COMPARISON).toBe(3);
    expect(r.enough).toBe(false);
    expect(r.net).toBeNull();
    expect(r.catch).toBeNull();
    expect(r.completed).toBe(2);
  });

  it("a trip without the figures does not count toward the three", () => {
    const partial = ownerTrip({ id: "p", state: "completed", outcome: { ...landed(400), crewPaidUsd: null } });
    const r = ownRecords([done("a", 300), done("b", 500), partial]);
    expect(r.net).toBeNull();
    expect(r.catch?.count).toBe(3);
  });

  it("from three: the median planned-against-actual difference, said with its count, nothing predictive", () => {
    const r = ownRecords([done("a", 300), done("b", 400), done("c", 500), done("d", 450)]);
    expect(r.catch?.count).toBe(4);
    expect(r.catch?.median).toBe(25);
    expect(r.catch?.words).toBe("Across your 4 completed trips with both figures recorded, the middle catch landed was 25 lb above plan.");
    expect(r.net?.count).toBe(4);
    expect(r.net?.words).toMatch(/^Across your 4 completed trips .* net to boat \(not profit\) was \$\d/);
    expect(JSON.stringify(r)).not.toMatch(/expect|predict|likely|will /i);
  });
});
