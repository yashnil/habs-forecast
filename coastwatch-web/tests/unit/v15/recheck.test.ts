import { describe, expect, it } from "vitest";
import { recheck, refreshPlanning, type FetchJson, type RecheckStore } from "@/nx/trips/recheck";
import type { Snapshot } from "@/nx/trips/types";
import { createTripsStore } from "@/nx/trips/store";
import { clone, gates, LATER, PLAN_AT, sea, trip } from "./evidenceFixtures";
import { memory } from "./tripsFixtures";

function memStore(): RecheckStore & { snaps: Map<string, Snapshot> } {
  const snaps = new Map<string, Snapshot>();

  return { snaps, getSnapshot: (id) => snaps.get(id) ?? null, putSnapshot: (s) => void snaps.set(s.id, s) };
}

const answers = (g: unknown, s: unknown): FetchJson => async (url) => {
  const body = url.includes("part=sea") ? s : g;
  if (body instanceof Error) throw body;
  return { status: 200, body };
};

async function planned(store = memStore()) {
  const r = await refreshPlanning(clone(trip), { store, fetchJson: answers(gates(), sea()), now: () => new Date(PLAN_AT), online: true });
  if (!r.ok) throw new Error(r.reason);
  return { store, trip: r.trip, plan: r.snapshot };
}

describe("v15 recheck", () => {
  it("identical answers: no changes, complete, everything the same", async () => {
    const { store, trip: t } = await planned();
    const r = await recheck(t, { store, fetchJson: answers(gates(), sea()), now: () => new Date(LATER), online: true });
    expect(r.changed).toEqual([]);
    expect(r.complete).toBe(true);
    expect(r.mode).toBe("live");
    expect(r.unchecked).toEqual([]);
    expect(r.same.length).toBe(5);
    expect(r.verify.map((v) => v.url)).toContain("https://forecast.weather.gov/shmrn.php?mz=pzz535");
    expect(r.trip.snapshots.rechecks).toEqual([r.snapshot.id]);
    expect(store.snaps.get(r.trip.snapshots.planning!)?.kind).toBe("planning");
  });

  it("a failed CDPH read: unchecked and incomplete, never 'no changes'", async () => {
    const { store, trip: t } = await planned();
    const g = gates();
    g.testsError = true;
    g.test = null;
    const r = await recheck(t, { store, fetchJson: answers(g, sea()), now: () => new Date(LATER), online: true });
    expect(r.complete).toBe(false);
    expect(r.unchecked.map((u) => u.source)).toEqual(["cdphTest"]);
    // told once, under "couldn't check", with what it hides from the plan; never in "changed", never "no changes"
    expect(r.unchecked[0].why).toContain("Shown when you planned: CDPH test, Dungeness crab, Monterey Bay");
    expect(r.changed).toEqual([]);
  });

  it("an HTTP error is a failed source, not offline", async () => {
    const { store, trip: t } = await planned();
    const fj: FetchJson = async (url) => (url.includes("part=sea") ? { status: 503, body: { ok: false, error: "NWS down" } } : { status: 200, body: gates() });
    const r = await recheck(t, { store, fetchJson: fj, now: () => new Date(LATER), online: true });
    expect(r.mode).toBe("live");
    expect(r.complete).toBe(false);
    expect(r.unchecked.map((u) => u.source)).toEqual(["nwsForecast", "nwsAlerts", "buoy", "tides"]);
    expect(r.unchecked[0].why).toContain("NWS down");
  });

  it("a request that never answers times out on its own clock", async () => {
    const { store, trip: t } = await planned();
    const fj: FetchJson = (url, { signal }) =>
      url.includes("part=sea") ? new Promise((_, rej) => signal.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" })))) : Promise.resolve({ status: 200, body: gates() });
    const r = await recheck(t, { store, fetchJson: fj, now: () => new Date(LATER), online: true, timeoutMs: 20 });
    expect(r.mode).toBe("live");
    expect(r.unchecked.find((u) => u.source === "nwsForecast")?.why).toContain("no answer within");
    expect(r.changed.some((c) => c.category === "official")).toBe(false);
  });

  it("offline: saved mode against the newest copy, incomplete, with its age, nothing fetched", async () => {
    const { store, trip: t } = await planned();
    let calls = 0;
    const fj: FetchJson = async () => {
      calls++;
      return { status: 200, body: gates() };
    };
    const r = await recheck(t, { store, fetchJson: fj, now: () => new Date(LATER), online: false });
    expect(calls).toBe(0);
    expect(r.mode).toBe("saved");
    expect(r.complete).toBe(false);
    expect(r.snapshot.savedFrom).toBe(new Date(PLAN_AT).toISOString());
    expect(r.unchecked).toHaveLength(7);
    expect(r.unchecked[0].why).toMatch(/^Offline: not checked now\. The copy on this device is from Oct 12/);
    expect(r.trip).toBe(t);
    expect(JSON.stringify(r)).not.toMatch(/contacted|checked with|asked (CDFW|NWS|CDPH)/i);
  });

  it("both requests failing at the network is offline too", async () => {
    const { store, trip: t } = await planned();
    const r = await recheck(t, { store, fetchJson: answers(new TypeError("Failed to fetch"), new TypeError("Failed to fetch")), now: () => new Date(LATER), online: true });
    expect(r.mode).toBe("saved");
    expect(r.complete).toBe(false);
  });

  it("offline with a newer field pack compares against the pack", async () => {
    const { store, trip: t } = await planned();
    const g = gates();
    g.season!.level = "closed";
    store.savedCopy = async () => ({ gates: g, sea: sea(), savedAt: "2026-10-13T10:00:00Z" });
    const r = await recheck(t, { store, now: () => new Date(LATER), online: false });
    expect(r.mode).toBe("saved");
    expect(r.snapshot.savedFrom).toBe("2026-10-13T10:00:00Z");
    expect(r.changed.map((c) => c.kind)).toEqual(["revised"]);
  });

  it("reports changes since the previous recheck too", async () => {
    const { store, trip: t } = await planned();
    const g = gates();
    g.season!.level = "closed";
    const r1 = await recheck(t, { store, fetchJson: answers(g, sea()), now: () => new Date(LATER), online: true });
    expect(r1.sinceLast).toBeNull();
    const r2 = await recheck(r1.trip, { store, fetchJson: answers(g, sea()), now: () => new Date("2026-10-13T14:00:00Z"), online: true });
    expect(r2.changed).toHaveLength(1);
    expect(r2.sinceLast).toEqual([]);
    expect(r2.trip.snapshots.rechecks).toHaveLength(2);
  });

  it("refreshing the plan keeps the old planning snapshot as superseded", async () => {
    const { store, trip: t, plan } = await planned();
    const r = await refreshPlanning(t, { store, fetchJson: answers(gates(), sea()), now: () => new Date(LATER), online: true });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.trip.snapshots.superseded).toEqual([plan.id]);
    expect(r.trip.snapshots.planning).not.toBe(plan.id);
    expect(store.snaps.has(plan.id)).toBe(true);
    const off = await refreshPlanning(r.trip, { store, now: () => new Date(LATER), online: false });
    expect(off.ok).toBe(false);
  });

  it("works on the trips store: planning attached, rechecks appended and pruned to 8, the planning snapshot kept", async () => {
    const store = createTripsStore(memory(), () => new Date(PLAN_AT));
    expect(store.put(clone(trip)).ok).toBe(true);
    const p = await refreshPlanning(store.get(trip.id)!, { store, fetchJson: answers(gates(), sea()), now: () => new Date(PLAN_AT), online: true });
    if (!p.ok) throw new Error(p.reason);
    expect(p.persisted).toBe(true);
    expect(store.get(trip.id)?.snapshots.planning).toBe(p.snapshot.id);
    let t = p.trip;
    for (let i = 0; i < 10; i++) {
      const r = await recheck(t, { store, fetchJson: answers(gates(), sea()), now: () => new Date(Date.parse(LATER) + i * 60_000), online: true });
      expect(r.persisted).toBe(true);
      expect(r.changed).toEqual([]);
      t = r.trip;
    }
    const kept = store.get(trip.id)!;
    expect(kept.snapshots.rechecks).toHaveLength(8);
    expect(t.snapshots.rechecks).toEqual(kept.snapshots.rechecks);
    expect(store.getSnapshot(kept.snapshots.planning!)?.kind).toBe("planning");
  });
});
