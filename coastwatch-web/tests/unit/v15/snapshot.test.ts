import { describe, expect, it } from "vitest";
import { norm, stableHash, takeSnapshot, versionOf } from "@/nx/trips/snapshot";
import { CDFW_URL, clone, gates, LATER, PLAN_AT, sea, snap, trip } from "./evidenceFixtures";

describe("v15 snapshot: capture", () => {
  it("gives every piece of evidence a stable key", () => {
    const s = snap();
    expect(s.evidence.map((e) => e.key)).toEqual(["season:dungeness_commercial:port-592", "whale:3:dungeness_commercial", "cdph:monterey-bay:dungeness-crab", "advisory:PZZ535:small-craft-advisory", "observation:ndbc:46042"]);
    expect(s.kind).toBe("planning");
    expect(s.mode).toBe("live");
    expect(s.serverNow).toBe(PLAN_AT);
    expect(s.forecast?.zone).toBe("PZZ535");
    expect(s.forecast?.periods[0]).toMatchObject({ key: "2026-10-13|day", windDir: "NW", windLo: 10, windHi: 15 });
    expect(s.tides?.events).toHaveLength(1);
    expect(s.assumptions.limits).toEqual({ windKt: 20, seasFt: 8 });
  });

  it("reads every source ok when every answer arrived", () => {
    const s = snap();
    expect(Object.values(s.sources).map((r) => r.state)).toEqual(Array(7).fill("ok"));
  });

  it("versions ignore fetch and checked times and whitespace or case", () => {
    const g = gates();
    g.now = LATER;
    g.season!.checked = LATER;
    g.season!.text = "  Commercial   season DELAYED to at least Jan 1, 2027 ";
    const a = snap();
    const b = snap({ gates: g, now: LATER });
    expect(b.evidence[0].version).toBe(a.evidence[0].version);
    expect(norm(" A  b. ")).toBe("a b");
    expect(versionOf({ x: "a  B", y: 1 })).toBe(versionOf({ y: 1, x: "A b" }));
    expect(stableHash("x")).toMatch(/^[0-9a-f]{16}$/);
  });

  it("a different level or a new test date is a new version", () => {
    const g = gates();
    g.season!.level = "closed";
    g.test!.date = "2026-10-09";
    const a = snap();
    const b = snap({ gates: g });
    expect(b.evidence[0].version).not.toBe(a.evidence[0].version);
    expect(b.evidence[2].version).not.toBe(a.evidence[2].version);
  });

  it("maps failures to failed, a missing test to absent", () => {
    const g = gates();
    g.statusError = true;
    g.season = null;
    g.testsError = true;
    g.test = null;
    const s = snap({ gates: g });
    expect(s.sources.season.state).toBe("failed");
    expect(s.sources.cdphTest.state).toBe("failed");
    expect(s.sources.whale.state).toBe("ok");
    const g2 = gates();
    g2.test = null;
    expect(snap({ gates: g2 }).sources.cdphTest.state).toBe("absent");
  });

  it("a sea that did not answer fails all four sea sources, with the reason", () => {
    const s = snap({ sea: { ok: false, error: "upstream timeout" } });
    for (const k of ["nwsForecast", "nwsAlerts", "buoy", "tides"] as const) {
      expect(s.sources[k].state).toBe("failed");
      expect(s.sources[k].detail).toContain("upstream timeout");
    }
    expect(s.forecast).toBeNull();
    const t = snap({ sea: "failed" });
    expect(t.sources.tides.state).toBe("failed");
    const n = snap({ sea: null, gates: null });
    expect(n.sources.season.state).toBe("notChecked");
  });

  it("a buoy-free coast is absent, a buoy that failed is failed", () => {
    const x = sea();
    x.sea.buoy = null;
    x.sea.buoyReason = "No NDBC station reports within 60 km.";
    expect(snap({ sea: x }).sources.buoy.state).toBe("absent");
    x.sea.buoyReason = "No recent observation from the nearest stations (timeout).";
    expect(snap({ sea: x }).sources.buoy.state).toBe("failed");
  });

  it("writes its limitations in plain words", () => {
    const s = snap();
    expect(s.limitations).toContain("NWS zone forecast covers the whole zone (PZZ535), not your route.");
    expect(s.limitations).toContain("CDPH's latest test is from Oct 2; it says nothing about today.");
    expect(s.limitations).toContain("Official notices are not verified by a person yet.");
    expect(s.limitations.join(" ")).not.toMatch(/\b(safe|legal|allowed|approved|all clear)\b/i);
  });

  it("a conflict is carried, never resolved", () => {
    const g = gates();
    g.season = { ...g.season!, conflict: true, differ: "CDFW's page says delayed; a notice says open.", text: "Sources differ about this fishery here; awaiting a person's review." };
    const s = snap({ gates: g });
    expect(s.evidence[0].conflict).toBe("CDFW's page says delayed; a notice says open.");
    expect(s.evidence[0].url).toBe(CDFW_URL);
  });

  it("a saved snapshot keeps its original time and no server clock", () => {
    const s = takeSnapshot({ trip: clone(trip), gates: gates(), sea: sea(), forecast: null, now: LATER, mode: "saved", savedFrom: PLAN_AT, kind: "recheck", id: "s" });
    expect(s.savedFrom).toBe(PLAN_AT);
    expect(s.serverNow).toBeNull();
    expect(s.limitations.some((l) => l.startsWith("Rebuilt from a copy saved on this device"))).toBe(true);
  });
});
