import { describe, expect, it } from "vitest";
import { briefing, cell, departurePeriods } from "@/nx/trips/ui/briefingModel";
import { tripGeo, num } from "@/nx/trips/ui/tripGeo";
import { readTrip, tripSnapshots } from "@/nx/trips/ui/localTrips";
import type { Snapshot } from "@/nx/trips/types";
import { clone, gates, sea, snap as evSnap, trip as evTrip } from "./evidenceFixtures";
import { memory, ownerTrip, snap } from "./tripsFixtures";

const MOSS = { key: "port-592", name: "Moss Landing", lon: -121.788, lat: 36.805 };
const VERDICT = /\b(safe|unsafe|legal|allowed|all clear|approved|profit)\b/i;

describe("the trip's chart: what is drawn where", () => {
  it("places the port, its RAMP zone and only the records the snapshot gives a position for", () => {
    const s = evSnap();
    const g = tripGeo(evTrip, s, MOSS);
    expect(g.zone?.zone).toBe("3"); // the whale record names the zone the agency put the port in
    expect(g.test).toBeNull(); // the fixture's test site has no position: listed, never drawn at a guess
    expect(g.nwsZone).toBe("PZZ535");
    expect(g.advisories).toEqual(["Small Craft Advisory"]);
    const [[w, south], [e, north]] = g.bounds;
    expect(w).toBeLessThan(MOSS.lon);
    expect(e).toBeGreaterThan(MOSS.lon);
    expect(south).toBeLessThan(MOSS.lat);
    expect(north).toBeGreaterThan(MOSS.lat);
  });

  it("draws a test site and a buoy when their positions are in the evidence, and refuses one far from the port", () => {
    const s = clone(evSnap());
    const t = s.evidence.find((x) => x.kind === "cdphTest")!;
    t.fields = { ...t.fields, lat: 36.8, lon: -121.9 };
    const b = s.evidence.find((x) => x.kind === "observation")!;
    b.fields = { ...b.fields, lat: 36.759, lon: -121.95 };
    const g = tripGeo(evTrip, s, MOSS);
    expect(g.test).toMatchObject({ lat: 36.8, lon: -121.9, max: 9.8, action: 30 });
    expect(g.buoy).toMatchObject({ id: "46042", waveFt: 5.2 });
    b.fields = { ...b.fields, lat: 40.1, lon: -124.3 };
    expect(tripGeo(evTrip, s, MOSS).buoy).toBeNull();
  });

  it("draws no whale zone for a fishery RAMP does not govern, and works with no snapshot", () => {
    const g = tripGeo({ ...evTrip, fishery: "rock_crab" }, null, MOSS);
    expect(g.zone).toBeNull();
    expect(g.takenAt).toBeNull();
    expect(num({ a: "4.5", b: null }, "b", "a")).toBe(4.5);
  });
});

describe("the departure briefing", () => {
  it("reads the regression economics through the engine: net to boat $2,602.24, break-even 39 lb", () => {
    const b = briefing(ownerTrip(), { planning: null, latest: null });
    expect(b.economics).toMatchObject({ audience: "commercial", net: "$2,602.24", gross: "$3,600.00" });
    expect(b.economics && b.economics.audience === "commercial" ? b.economics.breakEven : null).toMatch(/^39 lb/);
    expect(b.basis).toBeNull();
    expect(b.periodsWhy).toMatch(/No snapshot/);
  });

  it("states what the evidence rests on: a complete live recheck, or an incomplete one naming what was not checked", () => {
    const planning = evSnap();
    const b = briefing({ ...evTrip, snapshots: { planning: planning.id, rechecks: [], superseded: [] } }, { planning, latest: null });
    expect(b.basis).toMatchObject({ kind: "planning", complete: true, unchecked: [] });
    const failed = evSnap({ sea: "failed", kind: "recheck", id: "r1", now: "2026-10-14T15:00:00Z" });
    const r = briefing(evTrip, { planning, latest: failed });
    expect(r.basis?.complete).toBe(false);
    expect(r.basis?.unchecked.map((u) => u.source)).toEqual(expect.arrayContaining(["nwsForecast", "nwsAlerts", "buoy", "tides"]));
    // the planning copy of the forecast is still known (it does not reach Oct 15); the failed read did not erase it
    expect(r.periods).toEqual([]);
    expect(r.periodsWhy).toMatch(/does not reach your departure day/);
    // with no earlier copy at all, the failure is said
    expect(briefing(evTrip, { planning: null, latest: failed, all: [failed] }).periodsWhy).toMatch(/could not be read, and no earlier copy/);
  });

  it("a saved copy is never complete, even when every source in it had answered", () => {
    const saved: Snapshot = { ...evSnap(), kind: "recheck", mode: "saved", savedFrom: "2026-10-13T06:00:00Z" };
    expect(briefing(evTrip, { planning: evSnap(), latest: saved }).basis).toMatchObject({ complete: false, mode: "saved" });
  });

  it("sets the departure day's periods beside the reader's limits, never as a verdict", () => {
    const planning = clone(evSnap());
    planning.forecast = {
      zone: "PZZ535",
      issued: "2026-10-13T02:31:00-07:00",
      url: null,
      periods: [
        { key: "2026-10-14|night", name: "Wednesday Night", date: "2026-10-14", night: true, start: null, end: null, text: "", windDir: "NW", windLo: 10, windHi: 15, gustKt: null, seasLo: 5, seasHi: 6, waves: [] },
        { key: "2026-10-15|day", name: "Thursday", date: "2026-10-15", night: false, start: null, end: null, text: "", windDir: "N", windLo: 15, windHi: 25, gustKt: 30, seasLo: null, seasHi: null, waves: [] },
        { key: "2026-10-16|day", name: "Friday", date: "2026-10-16", night: false, start: null, end: null, text: "", windDir: "N", windLo: 5, windHi: 10, gustKt: null, seasLo: 2, seasHi: 3, waves: [] },
      ],
    };
    const b = briefing(evTrip, { planning, latest: null });
    expect(b.periods.map((p) => p.name)).toEqual(["Wednesday Night", "Thursday"]);
    expect(b.periods[0].windCell.state).toBe("within");
    expect(b.periods[1].windCell).toEqual({ state: "over", text: "Wind stated up to 25 kt; your limit 20 kt" });
    expect(b.periods[1].seasCell.state).toBe("unstated");
    expect(b.periods[1].wind).toBe("N 15–25 kt, gusts 30 kt");
    for (const p of b.periods) expect(`${p.windCell.text} ${p.seasCell.text}`).not.toMatch(VERDICT);
    expect(cell(null, 20, "kt", "Wind").state).toBe("unstated");
    expect(cell(12, null, "kt", "Wind").state).toBe("noLimit");
    expect(departurePeriods(planning.forecast, null)).toHaveLength(3);
  });

  it("lists official records with their own verification state and links, and offers the agencies to verify", () => {
    const b = briefing(evTrip, { planning: evSnap(), latest: null });
    const cdfw = b.official.find((o) => o.source === "CDFW" && o.kind === "season")!;
    expect(cdfw.verified).toBe(false);
    expect(cdfw.url).toMatch(/^https:\/\/wildlife\.ca\.gov/);
    expect(b.verify.some((v) => v.url.startsWith("https://www.cdph.ca.gov"))).toBe(true);
    expect(JSON.stringify(b)).not.toMatch(/all clear|approved/i);
  });

  it("reads one trip and its snapshots from storage, and reports a newer or broken record instead of repairing it", () => {
    const kv = memory();
    const t = ownerTrip({ snapshots: { planning: "p1", rechecks: ["r1", "r2"], superseded: [] } });
    kv.setItem(`nx-trips-v1:trip:${t.id}`, JSON.stringify(t));
    kv.setItem("nx-trips-v1:snap:p1", JSON.stringify(snap("p1", t.id, "planning")));
    kv.setItem("nx-trips-v1:snap:r1", JSON.stringify(snap("r1", t.id)));
    kv.setItem("nx-trips-v1:snap:r2", "{broken");
    const r = readTrip(t.id, kv);
    expect(r.state).toBe("ok");
    const s = tripSnapshots(t, kv);
    expect(s.planning?.id).toBe("p1");
    expect(s.latest?.id).toBe("r1"); // the newest readable recheck; the broken one is left as it is
    expect(kv.getItem("nx-trips-v1:snap:r2")).toBe("{broken");
    kv.setItem("nx-trips-v1:trip:new", JSON.stringify({ ...t, id: "new", schema: 99 }));
    expect(readTrip("new", kv).state).toBe("newer");
    expect(readTrip("absent", kv).state).toBe("missing");
    void gates;
    void sea;
  });

  it("a recheck whose sea read failed keeps the forecast and the advisory the device had, with when they were read", () => {
    const planning = evSnap();
    const failed = evSnap({ sea: "failed", kind: "recheck", id: "r-503", now: "2026-10-14T15:00:00Z" });
    const b = briefing(evTrip, { planning, latest: failed, all: [planning, failed] });
    expect(b.basis?.complete).toBe(false);
    expect(b.forecast).toMatchObject({ zone: "PZZ535", held: { latest: false, readAt: planning.takenAt } });
    const sca = b.official.find((o) => o.kind === "advisory");
    expect(sca?.title).toBe("Small Craft Advisory");
    expect(sca?.held).toEqual({ readAt: planning.takenAt, latest: false });
    // the CDFW record was read again by the failed recheck (its gates answered): it is from the latest check
    expect(b.official.find((o) => o.kind === "season")?.held.latest).toBe(true);
    expect(b.periodsWhy ?? "").not.toMatch(/could not be read/);
  });
});
