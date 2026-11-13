import { describe, expect, it } from "vitest";
import { feedView, groupUnchecked, readableDetail, stopPosition, timelineOf } from "@/nx/trips/feed";
import type { Recheck, Snapshot } from "@/nx/trips/types";
import { gates, LATER, PLAN_AT, snap } from "./evidenceFixtures";

const base: Recheck = { at: LATER, mode: "live", complete: true, changed: [], same: [], unchecked: [], stale: [], verify: [], baseId: "plan-1", snapshotId: "r1" };
const FORBIDDEN = /\b(safe|legal|allowed|open for you|all clear|approved)\b/i;

describe("v15 feed: the words of 'Since you planned this trip'", () => {
  it("no planning snapshot, not rechecked, checking", () => {
    expect(feedView({ planning: null, result: null, checking: false, now: LATER }).state).toBe("none");
    expect(feedView({ planning: snap(), result: null, checking: false, now: LATER }).state).toBe("unchecked");
    expect(feedView({ planning: snap(), result: null, checking: true, now: LATER }).state).toBe("checking");
  });

  it("'No changes found' only after a complete live check", () => {
    const v = feedView({ planning: snap(), result: base, checking: false, now: LATER });
    expect(v.state).toBe("complete");
    expect(v.headline).toBe("No changes found in the sources checked at 5:00 AM PT");
  });

  it("an incomplete check is never an all-clear and names what it could not see", () => {
    const v = feedView({ planning: snap(), result: { ...base, complete: false, unchecked: [{ source: "cdphTest", why: "CDPH's test results could not be read.", required: true }] }, checking: false, now: LATER });
    expect(v.state).toBe("incomplete");
    expect(v.headline).toBe("Incomplete: 1 required source could not be checked");
    expect(v.headline).not.toMatch(/no changes/i);
    expect(v.unchecked.map((u) => u.label)).toEqual(["CDPH domoic acid test"]);
  });

  it("offline says the saved copy and its age, never a live check", () => {
    const s = { ...snap({ kind: "recheck", id: "s" }), mode: "saved" as const, savedFrom: PLAN_AT };
    const v = feedView({ planning: snap(), result: { ...base, mode: "saved", complete: false, snapshot: s } as Recheck, checking: false, now: "2026-10-15T18:00:00Z" });
    expect(v.state).toBe("saved");
    expect(v.sub).toBe("Not a live check. Compared against copies saved on this device 3 days ago.");
    for (const t of [v.headline, v.sub ?? ""]) expect(t).not.toMatch(FORBIDDEN);
  });

  it("the history puts planning before rechecks, with completeness and change counts", () => {
    const plan = snap();
    const g = gates();
    g.season!.level = "closed";
    const r1: Snapshot = snap({ gates: g, now: LATER, kind: "recheck", id: "r1" });
    const r2: Snapshot = snap({ gates: { ...gates(), testsError: true, test: null }, now: "2026-10-13T13:00:00Z", kind: "recheck", id: "r2" });
    const t = timelineOf(plan, [r2, r1], [], "2026-10-13T14:00:00Z");
    expect(t.map((x) => [x.id, x.complete, x.changes])).toEqual([
      ["plan-1", true, 0],
      ["r1", true, 1],
      ["r2", false, 1],
    ]);
    expect(stopPosition(PLAN_AT, PLAN_AT, LATER)).toBe(0);
    expect(stopPosition(LATER, PLAN_AT, LATER)).toBe(1);
  });

  it("complete with only optional sources missing says so, and lists stale records apart", () => {
    const v = feedView({
      planning: snap(),
      result: { ...base, complete: true, unchecked: [{ source: "tides", why: "CO-OPS did not answer.", required: false }], stale: [{ key: "cdph:x", title: "CDPH test, Monterey Bay", asOf: "2026-09-28" }] },
      checking: false,
      now: LATER,
    });
    expect(v.state).toBe("complete");
    expect(v.headline).toMatch(/^No changes found in the sources checked at/);
    expect(v.sub).toMatch(/Every required source answered/);
    expect(v.sub).toMatch(/CO-OPS tide predictions could not be read \(optional context/);
    expect(v.sub).toMatch(/1 record answered but is old/);
    expect(v.unchecked.map((u) => u.source)).toEqual(["tides"]);
    expect(v.stale).toHaveLength(1);
  });

  it("says one reason once for every source it applies to, and hides raw period keys", () => {
    const why = "Offline: not checked.";
    const all = (["season", "whale", "cdphTest", "nwsForecast", "nwsAlerts", "buoy", "tides"] as const).map((source) => ({ source, why }));
    expect(groupUnchecked(all)).toEqual([{ sources: all.map((x) => x.source), labels: "Every source", why, required: true }]);
    expect(groupUnchecked([{ source: "buoy", why: "a" }, { source: "tides", why: "a" }])[0].labels).toBe("NDBC buoy, CO-OPS tide predictions");
    // required and optional never share a line; optional is marked, and required comes first
    const g = groupUnchecked([{ source: "tides", why: "a", required: false }, { source: "nwsForecast", why: "a", required: true }]);
    expect(g.map((x) => x.labels)).toEqual(["NWS zone forecast", "CO-OPS tide predictions (optional)"]);
    expect(readableDetail("2026-10-12|night")).toBeNull();
    expect(readableDetail("NW winds 20 kt")).toBe("NW winds 20 kt");
  });
});
