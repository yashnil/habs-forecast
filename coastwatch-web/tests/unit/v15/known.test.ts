import { describe, expect, it } from "vitest";
import { feedView } from "@/nx/trips/feed";
import { knownView } from "@/nx/trips/known";
import { recheck, refreshPlanning, type FetchJson, type RecheckStore } from "@/nx/trips/recheck";
import type { Snapshot } from "@/nx/trips/types";
import { clone, gates, PLAN_AT, sea, snap, trip } from "./evidenceFixtures";

const at = (h: number) => () => new Date(Date.parse(PLAN_AT) + h * 3_600_000);
const store = (): RecheckStore & { snaps: Map<string, Snapshot> } => {
  const snaps = new Map<string, Snapshot>();
  return { snaps, getSnapshot: (id) => snaps.get(id) ?? null, putSnapshot: (s) => void snaps.set(s.id, s) };
};
const answer = (g: unknown, s: unknown, seaStatus = 200): FetchJson => async (url) => (url.includes("part=sea") ? { status: seaStatus, body: s } : { status: 200, body: g });

/** planning: a calm sea, no advisory */
const calm = () => {
  const s = sea();
  s.sea.alerts = [];
  s.sea.lanes = s.sea.lanes.map((p) => ({ ...p, text: "NW winds 5 to 10 kt. Seas 3 ft.", seasLo: 3, seasHi: 3 }));
  return s;
};
/** the changed answer: a Small Craft Advisory and 15–25 kt */
const rough = () => {
  const s = sea();
  s.sea.lanes = s.sea.lanes.map((p) => ({ ...p, text: "NW winds 15 to 25 kt. Seas 6 ft.", seasLo: 6, seasHi: 6 }));
  return s;
};

describe("v15 known: a failed read never replaces evidence the device holds", () => {
  it("changed → 503 → offline: the advisory and 15–25 kt survive", async () => {
    const st = store();
    const p = await refreshPlanning(clone(trip), { store: st, fetchJson: answer(gates(), calm()), now: at(0), online: true });
    if (!p.ok) throw new Error(p.reason);

    const r1 = await recheck(p.trip, { store: st, fetchJson: answer(gates(), rough()), now: at(1), online: true });
    expect(r1.changed.some((c) => c.category === "advisory" && c.kind === "new")).toBe(true);
    expect(r1.changed.some((c) => c.category === "forecast" && /stronger winds/.test(c.title))).toBe(true);

    // a 503 whose body names no error
    const r2 = await recheck(r1.trip, { store: st, fetchJson: answer(gates(), { ok: false }, 503), now: at(2), online: true });
    expect(r2.complete).toBe(false);
    const nws = r2.unchecked.find((u) => u.source === "nwsForecast")!;
    expect(nws.why).toContain("(HTTP 503)");
    expect(nws.why).not.toContain("undefined");
    expect(nws.why).toMatch(/Showing the forecast from Oct 1[23]/);

    const snaps = [...st.snaps.values()];
    const v = knownView(snaps);
    expect(v.forecast?.snapshotId).toBe(r1.snapshot.id);
    expect(v.forecast?.latest).toBe(false);
    expect(v.forecast?.value.periods.some((x) => x.windLo === 15 && x.windHi === 25)).toBe(true);
    expect(v.evidence.find((e) => e.value.kind === "advisory")?.snapshotId).toBe(r1.snapshot.id);
    expect(v.sources.nwsForecast.last.state).toBe("failed");
    expect(v.sources.nwsForecast.knownAt).toBe(r1.snapshot.takenAt);
    expect(v.sources.season.knownAt).toBe(r2.snapshot.takenAt);
    expect(v.evidence.find((e) => e.value.kind === "season")?.latest).toBe(true);

    const off = await recheck(r2.trip, { store: st, now: at(3), online: false });
    expect(off.mode).toBe("saved");
    expect(off.complete).toBe(false);
    expect(off.snapshot.forecast?.periods.some((x) => x.windHi === 25)).toBe(true);
    expect(off.snapshot.evidence.some((e) => e.kind === "advisory")).toBe(true);
    expect(off.snapshot.savedFrom).toBe(r1.snapshot.takenAt);
    expect(off.knownNewest).toBe(r2.snapshot.takenAt);
    expect(off.changed.some((c) => c.category === "advisory" && c.kind === "new")).toBe(true);
    expect(off.changed.some((c) => c.category === "forecast" && /stronger winds/.test(c.title))).toBe(true);
    expect(off.unchecked.find((u) => u.source === "nwsForecast")?.why).toMatch(/^Offline: not checked now\. The copy on this device is from/);
    const f = feedView({ planning: st.snaps.get(p.trip.snapshots.planning!)!, result: off, checking: false, now: at(3)().toISOString() });
    expect(f.state).toBe("saved");
    expect(`${f.headline} ${f.sub}`).not.toMatch(/nothing new/i);
    expect(f.headline).toMatch(/differences? in the copies saved on this device/);
  });

  it("offline reads the same changes as online when the device clock is skewed from the server's", async () => {
    // the device is a month ahead of the server (a pinned fixture, or a phone set wrong): the answers carry the
    // server's clock; offline only the device's exists, carried forward with the offset learned online
    const skew = 31 * 24;
    const st = store();
    const p = await refreshPlanning(clone(trip), { store: st, fetchJson: answer(gates(), calm()), now: at(skew), online: true });
    if (!p.ok) throw new Error(p.reason);
    const r1 = await recheck(p.trip, { store: st, fetchJson: answer(gates(), rough()), now: at(skew + 1), online: true });
    const off = await recheck(r1.trip, { store: st, now: at(skew + 1.5), online: false });
    const forecastIds = (cs: { id: string; category: string }[]) => cs.filter((c) => c.category === "forecast" || c.category === "advisory").map((c) => c.id).sort();
    expect(forecastIds(off.changed)).toEqual(forecastIds(r1.changed));
    expect(off.changed.some((c) => c.kind === "stale")).toBe(false);
  });

  it("a record is dropped only when a later answering read no longer lists it", () => {
    const plan = snap();
    const g = gates();
    g.statusError = true;
    g.season = null;
    const failed = snap({ gates: g, now: "2026-10-13T08:00:00Z", kind: "recheck", id: "r1" });
    expect(knownView([failed, plan]).evidence.some((e) => e.value.kind === "season")).toBe(true);
    const g2 = gates();
    g2.season = null;
    const lifted = snap({ gates: g2, now: "2026-10-13T09:00:00Z", kind: "recheck", id: "r2" });
    expect(knownView([plan, failed, lifted]).evidence.some((e) => e.value.kind === "season")).toBe(false);
  });

  it("offline with every copy unchanged says so, with the newest time, and that nothing was checked", async () => {
    const st = store();
    const p = await refreshPlanning(clone(trip), { store: st, fetchJson: answer(gates(), sea()), now: at(0), online: true });
    if (!p.ok) throw new Error(p.reason);
    const off = await recheck(p.trip, { store: st, now: at(1), online: false });
    const f = feedView({ planning: p.snapshot, result: off, checking: false, now: at(1)().toISOString() });
    expect(off.changed).toEqual([]);
    expect(f.headline).toBe("No changes in the copies saved on this device (newest Oct 12, 11:00 PM PT); nothing was checked now.");
  });
});
