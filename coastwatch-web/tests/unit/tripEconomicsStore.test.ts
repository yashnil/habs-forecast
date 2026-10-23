import { describe, expect, it } from "vitest";
import {
  MAX_SCENARIOS,
  MAX_VESSELS,
  NS,
  createTripStore,
  decodeShare,
  draftInputs,
  emptyDraft,
  encodeShare,
  migrateRecord,
  type Draft,
  type KV,
} from "@/lib/tripEconomicsStore";
import { dieselSnapshot } from "@/content/referenceValues";

function memory(init: Record<string, string> = {}): KV & { map: Map<string, string> } {
  const map = new Map(Object.entries(init));
  return {
    map,
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, String(v)),
    removeItem: (k) => void map.delete(k),
    key: (i) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
}

let t = Date.parse("2026-10-20T12:00:00Z");
const clock = () => new Date((t += 1000));

const draft = (): Draft => ({
  ...emptyDraft("dungeness"),
  inputs: { roundTripNm: 20, cruiseKt: 8, steamGph: 7.8, fishingHours: 8, fishingGph: 4.8, fuelUsdPerGal: 5, fuelShared: true, costs: [{ label: "Ice", usd: 40, shared: true }], catch: [{ species: "Dungeness crab", ref: "dungeness", lb: 1000, usdPerLb: 3.5 }], crew: { kind: "afterExpenses", pct: 28.1 } },
  pots: 300,
  baitUsdPerPot: 1.5,
  basis: { fishingGph: "preset", fuelUsdPerGal: "eia" },
});

describe("trip store", () => {
  it("saves, renames, duplicates and deletes scenarios with ids and timestamps", () => {
    const kv = memory();
    const s = createTripStore(kv, clock);
    const a = s.saveScenario({ name: "Opening day", placeKey: "port-452", species: "dungeness", draft: draft(), refs: [dieselSnapshot()] });
    expect(a.ok).toBe(true);
    let d = s.load();
    expect(d.scenarios).toHaveLength(1);
    const rec = d.scenarios[0];
    expect(rec).toMatchObject({ schema: 1, name: "Opening day", placeKey: "port-452" });
    expect(rec.id).toBeTruthy();
    expect(Date.parse(rec.createdAt)).toBeGreaterThan(0);
    expect(rec.refs[0]).toMatchObject({ key: "eia:diesel-ca", value: 8.242, period: "week of 2026-10-19" });
    s.renameScenario(rec.id, "Opener, 300 pots");
    const dup = s.duplicateScenario(rec.id);
    d = s.load();
    expect(d.scenarios.map((x) => x.name).sort()).toEqual(["Opener, 300 pots", "Opener, 300 pots (copy)"]);
    s.deleteScenario(dup.id!);
    expect(s.load().scenarios).toHaveLength(1);
    expect(JSON.parse(kv.map.get(`${NS}scenarios`)!)[0].schema).toBe(1);
  });

  it("caps vessels at 5 and scenarios at 20 (refusing, never evicting)", () => {
    const s = createTripStore(memory(), clock);
    for (let i = 0; i < MAX_VESSELS; i++) expect(s.saveVessel({ name: `V${i}`, cruiseKt: 8, steamGph: 7, fishingGph: 4, crew: 2, pots: null }).ok).toBe(true);
    expect(s.saveVessel({ name: "one more", cruiseKt: 8, steamGph: 7, fishingGph: 4, crew: 2, pots: null })).toEqual({ ok: false, reason: "full" });
    for (let i = 0; i < MAX_SCENARIOS; i++) expect(s.saveScenario({ name: `S${i}`, placeKey: "p", species: "x", draft: draft(), refs: [] }).ok).toBe(true);
    expect(s.saveScenario({ name: "21", placeKey: "p", species: "x", draft: draft(), refs: [] })).toEqual({ ok: false, reason: "full" });
    expect(s.load().scenarios).toHaveLength(MAX_SCENARIOS);
  });

  it("keeps a working draft per place and species", () => {
    const s = createTripStore(memory(), clock);
    s.saveDraft("port-452|dungeness", draft());
    expect(s.load().meta.drafts["port-452|dungeness"].draft.pots).toBe(300);
    expect(draftInputs(s.load().meta.drafts["port-452|dungeness"].draft).costs!.at(-1)).toEqual({ label: "Bait (300 pots)", usd: 450, shared: true });
  });

  it("upgrades a pre-schema record (migration)", () => {
    const now = "2026-10-20T00:00:00.000Z";
    const m = migrateRecord({ id: "old1", name: "Before v1", inputs: { fuelGal: 30, fuelUsdPerGal: -2, crew: { kind: "ofGross", pct: 30 } } }, "scenario", now);
    expect(m).toMatchObject({ schema: 1, id: "old1", createdAt: now, updatedAt: now, name: "Before v1" });
    const rec = m as Exclude<typeof m, "newer" | null> & { draft: Draft };
    expect(rec.draft.inputs.fuelGal).toBe(30);
    expect(rec.draft.inputs.fuelUsdPerGal).toBeNull();
    expect(rec.draft.inputs.crew!.kind).toBe("ofGross");
    // read through the store, it is shown and written back at schema 1
    const kv = memory({ [`${NS}scenarios`]: JSON.stringify([{ id: "old1", name: "Before v1", inputs: { fuelGal: 30 } }]) });
    const s = createTripStore(kv, clock);
    expect(s.load().scenarios[0].schema).toBe(1);
    s.renameScenario("old1", "Renamed");
    expect(JSON.parse(kv.map.get(`${NS}scenarios`)!)[0]).toMatchObject({ schema: 1, name: "Renamed" });
  });

  it("is read-only when a newer schema wrote here", () => {
    const newer = JSON.stringify([{ schema: 2, id: "n1", createdAt: "2027-01-01T00:00:00Z", updatedAt: "2027-01-01T00:00:00Z", whatever: true }]);
    const kv = memory({ [`${NS}scenarios`]: newer });
    const s = createTripStore(kv, clock);
    expect(s.load().status.readOnly).toBe(true);
    expect(s.saveScenario({ name: "x", placeKey: "p", species: "x", draft: draft(), refs: [] })).toEqual({ ok: false, reason: "readOnly" });
    expect(s.saveDraft("k", draft())).toEqual({ ok: false, reason: "readOnly" });
    expect(kv.map.get(`${NS}scenarios`)).toBe(newer);
    const meta = memory({ [`${NS}meta`]: JSON.stringify({ schema: 3, drafts: {} }) });
    expect(createTripStore(meta, clock).load().status.readOnly).toBe(true);
  });

  it("keeps and reports corrupt data, copying it aside before any write", () => {
    const kv = memory({ [`${NS}scenarios`]: "{not json", [`${NS}vessels`]: JSON.stringify([{ schema: 1, id: "v1", name: "Ok boat", createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" }, 42, { nonsense: true }]) });
    const s = createTripStore(kv, clock);
    const d = s.load();
    expect(d.status.corrupt).toEqual(["scenarios"]);
    expect(d.status.invalidRecords).toBe(2);
    expect(d.vessels.map((v) => v.name)).toEqual(["Ok boat"]);
    // nothing was touched by reading
    expect(kv.map.get(`${NS}scenarios`)).toBe("{not json");
    // a write keeps the damaged original under corrupt:, and the bad vessel records ride along
    expect(s.saveScenario({ name: "New", placeKey: "p", species: "x", draft: draft(), refs: [] }).ok).toBe(true);
    expect(kv.map.get(`${NS}corrupt:scenarios`)).toBe("{not json");
    s.saveVessel({ name: "Second", cruiseKt: 7, steamGph: 6, fishingGph: 4, crew: 1, pots: null });
    const raw = JSON.parse(kv.map.get(`${NS}vessels`)!);
    expect(raw).toContain(42);
    expect(raw).toContainEqual({ nonsense: true });
    expect(s.load().vessels).toHaveLength(2);
  });

  it("exports and imports JSON, newer records winning", () => {
    const a = createTripStore(memory(), clock);
    a.saveScenario({ name: "A", placeKey: "p", species: "dungeness", draft: draft(), refs: [] });
    a.saveVessel({ name: "Boat", cruiseKt: 8, steamGph: 7.8, fishingGph: 4.8, crew: 2, pots: 300 });
    const json = a.exportJson();
    const b = createTripStore(memory(), clock);
    expect(b.importJson(json)).toEqual({ ok: true, added: 2, updated: 0, skipped: 0 });
    expect(b.importJson(json)).toEqual({ ok: true, added: 0, updated: 0, skipped: 0 });
    expect(b.load().scenarios[0].draft.inputs.catch![0].lb).toBe(1000);
    expect(b.importJson("nope")).toEqual({ ok: false, reason: "invalid" });
    expect(b.importJson(JSON.stringify({ app: "neritic-trip", schema: 9, scenarios: [] }))).toEqual({ ok: false, reason: "newer" });
  });

  it("deletes all trip data, including damaged copies, and nothing else", () => {
    const kv = memory({ "cw-waters-v1": "[]", [`${NS}scenarios`]: "{bad" });
    const s = createTripStore(kv, clock);
    s.saveDraft("k", draft());
    s.saveScenario({ name: "x", placeKey: "p", species: "x", draft: draft(), refs: [] });
    expect(s.deleteAll().ok).toBe(true);
    expect([...kv.map.keys()]).toEqual(["cw-waters-v1"]);
  });

  it("works without storage, claiming nothing", () => {
    const s = createTripStore(null, clock);
    expect(s.load().status.available).toBe(false);
    expect(s.saveDraft("k", draft())).toEqual({ ok: false, reason: "unavailable" });
  });
});

describe("scenario links", () => {
  it("round-trips the numbers and carries no names", () => {
    const d = draft();
    d.inputs.costs = [{ label: "Skipper's cousin Bob", usd: 40, shared: false }];
    const enc = encodeShare(d);
    expect(enc).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(atob(enc.replace(/-/g, "+").replace(/_/g, "/"))).not.toMatch(/Bob/);
    const back = decodeShare(enc)!;
    expect(back.inputs.costs).toEqual([{ label: "Cost 1", usd: 40, shared: false }]);
    expect(back.inputs.catch).toEqual([{ species: "Dungeness crab", ref: "dungeness", lb: 1000, usdPerLb: 3.5 }]);
    expect(back.inputs.crew).toMatchObject({ kind: "afterExpenses", pct: 28.1 });
    expect(back.pots).toBe(300);
    expect(back.preset).toBe("dungeness");
  });
  it("rejects garbage and makes hostile numbers safe", () => {
    expect(decodeShare("!!!")).toBeNull();
    const evil = btoa(JSON.stringify({ v: 1, n: [-1, 1e12, null], k: [["x", -5, 2]], m: ["drop table", 500] })).replace(/=+$/, "");
    const d = decodeShare(evil)!;
    expect(d.inputs.roundTripNm).toBeNull();
    expect(d.inputs.cruiseKt).toBe(60);
    expect(d.inputs.catch![0].lb).toBeNull();
    expect(d.inputs.crew).toMatchObject({ kind: "afterExpenses", pct: 100 });
  });
});
