import { describe, expect, it } from "vitest";
import { diffSnapshots } from "@/nx/trips/diff";
import { ALERT_URL, CDFW_URL, gates, LATER, sea, snap } from "./evidenceFixtures";

const later = (o: Parameters<typeof snap>[0] = {}) => snap({ now: LATER, kind: "recheck", id: "r1", ...o });

describe("v15 diff: what changed since planning", () => {
  it("a refetch of identical data finds nothing", () => {
    const g = gates();
    g.now = LATER;
    g.season!.checked = LATER;
    expect(diffSnapshots(snap(), later({ gates: g }), LATER)).toEqual([]);
  });

  it("a position-only difference (CDPH site, buoy) is not a change", () => {
    const g = gates();
    g.test = { ...g.test!, lat: 36.8, lon: -121.9 };
    const s = sea();
    s.sea.buoy = { ...s.sea.buoy!, lat: 36.75, lon: -122.42 };
    const b = later({ gates: g, sea: s });
    expect(b.evidence.find((e) => e.kind === "cdphTest")?.fields).toMatchObject({ lat: 36.8, lon: -121.9 });
    expect(b.evidence.find((e) => e.kind === "observation")?.fields).toMatchObject({ id: "46042", lat: 36.75, lon: -122.42 });
    expect(diffSnapshots(snap(), b, LATER)).toEqual([]);
  });

  it("a whitespace- or case-only difference is not a change", () => {
    const g = gates();
    g.season!.text = "commercial  season delayed to at least jan 1, 2027";
    g.whale!.measures = "Fleet  Advisory.   No depth limit or gear reduction on record";
    const s = sea();
    s.sea.alerts[0].headline = "SMALL CRAFT ADVISORY UNTIL TUESDAY 3 PM";
    expect(diffSnapshots(snap(), later({ gates: g, sea: s }), LATER)).toEqual([]);
  });

  it("a revised CDFW notice is one revised change, with the CDFW url and the fields that moved", () => {
    const g = gates();
    g.season!.level = "closed";
    g.season!.text = "Commercial season closed in Zone 3.";
    const c = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(c).toHaveLength(1);
    expect(c[0]).toMatchObject({ kind: "revised", category: "official", priority: 1, url: CDFW_URL, evidenceKey: "season:dungeness_commercial:port-592" });
    expect(c[0].title).toBe("Agency information changed; review the original CDFW page.");
    expect(c[0].detail).toContain("status: delayed → closed");
    expect(c[0].detail).toContain("wording");
    expect(c[0].before).toContain("delayed");
    expect(c[0].after).toContain("closed");
  });

  it("a new notice where the source read before is new", () => {
    const g = gates();
    g.season = null;
    const base = snap({ gates: g });
    const c = diffSnapshots(base, later(), LATER);
    expect(c).toHaveLength(1);
    expect(c[0]).toMatchObject({ kind: "new", category: "official", url: CDFW_URL });
  });

  it("a lifted notice (gone while the source read) is removed; gone while the source failed is couldn't check", () => {
    const g = gates();
    g.season = null;
    const lifted = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(lifted.map((x) => x.kind)).toEqual(["removed"]);
    expect(lifted[0].url).toBe(CDFW_URL);
    g.statusError = true;
    const failed = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(failed.map((x) => x.kind)).toEqual(["unavailable"]);
    expect(failed[0].priority).toBe(1);
    expect(failed[0].title).toContain("could not be checked");
  });

  it("a conflict appears, then persists, and is said both times", () => {
    const g = gates();
    g.season = { ...g.season!, conflict: true, differ: "CDFW's page says delayed; a notice says open.", text: "Sources differ about this fishery here; awaiting a person's review." };
    const first = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(first).toHaveLength(1);
    expect(first[0]).toMatchObject({ kind: "conflict", priority: 1 });
    expect(first[0].title).toMatch(/^Sources now disagree/);
    const again = diffSnapshots(later({ gates: g }), later({ gates: g, id: "r2" }), LATER);
    expect(again).toHaveLength(1);
    expect(again[0].title).toMatch(/^Sources still disagree/);
  });

  it("a newer CDPH test is new, with the published wording", () => {
    const g = gates();
    g.test = { ...g.test!, date: "2026-10-12", raw: ["12", "31"], values: [12, 31], max: 31, over: 1 };
    const c = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(c).toHaveLength(1);
    expect(c[0]).toMatchObject({ kind: "new", category: "toxin", priority: 1, title: "New CDPH test published; inspect measured values and dates." });
  });

  it("an older CDPH copy than planning is never news", () => {
    const g = gates();
    g.test = { ...g.test!, date: "2026-09-20" };
    const c = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(c.map((x) => x.kind)).toEqual(["stale"]);
    expect(c[0].priority).toBe(3);
  });

  it("whale measures changing is an official change", () => {
    const g = gates();
    g.whale!.measures = "Depth restriction to 30 fathoms.";
    const c = diffSnapshots(snap(), later({ gates: g }), LATER);
    expect(c).toHaveLength(1);
    expect(c[0]).toMatchObject({ kind: "revised", category: "official" });
    expect(c[0].title).toContain("whale measures changed");
  });

  it("a source failing then recovering is unavailable then available, and its records are not new", () => {
    const failed = later({ sea: "failed", id: "r1" });
    const c1 = diffSnapshots(snap(), failed, LATER);
    expect(c1.every((x) => x.kind === "unavailable")).toBe(true);
    expect(c1).toHaveLength(4);
    const c2 = diffSnapshots(failed, later({ id: "r2" }), LATER);
    expect(c2.filter((x) => x.kind === "new")).toEqual([]);
    expect(c2.filter((x) => x.kind === "available")).toHaveLength(4);
  });

  it("a new advisory is new; one no longer listed is removed (forecastDiff)", () => {
    const s = sea();
    s.sea.alerts = [];
    const base = snap({ sea: s });
    const c = diffSnapshots(base, later(), LATER);
    expect(c.map((x) => [x.kind, x.category])).toEqual([["new", "advisory"]]);
    expect(c[0].url).toBe(ALERT_URL);
    const gone = diffSnapshots(snap(), later({ sea: s }), LATER);
    expect(gone.map((x) => x.kind)).toEqual(["removed"]);
  });

  it("dedupes and puts official before forecast before observations", () => {
    const g = gates();
    g.season!.level = "closed";
    const s = sea();
    s.sea.lanes[1] = { ...s.sea.lanes[1], text: "NW winds 20 to 25 kt. Seas 8 ft.", seasLo: 8, seasHi: 8 };
    s.sea.buoy = { ...s.sea.buoy!, windKt: 18, at: "2026-10-13T11:40:00Z" };
    const c = diffSnapshots(snap(), later({ gates: g, sea: s }), LATER);
    expect(c.map((x) => x.category)).toEqual(["official", "forecast", "forecast", "observation"]);
    expect(c.map((x) => x.title)).toContain("Marine forecast updated since you planned: stronger winds for Wed 14, day.");
    expect(new Set(c.map((x) => x.id)).size).toBe(c.length);
  });
});
