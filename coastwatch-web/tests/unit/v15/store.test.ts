import { describe, expect, it } from "vitest";
import { toDraft } from "@/nx/coast/workspace";
import { MAX_HISTORY, MAX_RECHECKS, TRIPS_NS, createTripsStore, csvCell } from "@/nx/trips/store";
import { clockAt, crab, memory, ownerTrip, ownerWs, snap } from "./tripsFixtures";

const T0 = "2026-11-13T08:00:00.000Z";

describe("the trips store: put, get, list, remove", () => {
  it("round-trips a trip under nx-trips-v1: with one key per trip and an index", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    const r = s.put(ownerTrip());
    expect(r.ok).toBe(true);
    expect([...kv.map.keys()].sort()).toEqual([`${TRIPS_NS}index`, `${TRIPS_NS}mark`, `${TRIPS_NS}trip:t-owner`]);
    const got = s.get("t-owner")!;
    expect(got.name).toBe("Moss Landing crab");
    expect(got.economics).toEqual(ownerTrip().economics);
    expect(got.createdAt).toBe("2026-11-01T08:00:00.000Z");
    expect(got.updatedAt > got.createdAt).toBe(true);
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
  });

  it("put bumps updatedAt and keeps earlier economics in history when they change (newest first, at most 10)", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    s.put(ownerTrip());
    const first = s.get("t-owner")!;
    // same economics: no history entry
    s.put({ ...first, notes: "new note" });
    expect(s.get("t-owner")!.history).toEqual([]);
    expect(s.get("t-owner")!.updatedAt > first.updatedAt).toBe(true);
    for (let p = 10; p < 23; p++) s.put({ ...s.get("t-owner")!, economics: toDraft(ownerWs({ catch: [{ label: "Dungeness crab", ref: "dungeness", amount: "400", price: String(p) }] }), crab) });
    const h = s.get("t-owner")!.history;
    expect(h).toHaveLength(MAX_HISTORY);
    expect(h[0].economics.inputs.catch![0].usdPerLb).toBe(21);
    expect(h[h.length - 1].economics.inputs.catch![0].usdPerLb).toBe(12);
  });

  it("remove deletes the trip and its snapshots, and undo puts both back exactly", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    s.putSnapshot(snap("s-plan", "t-owner", "planning"));
    s.putSnapshot(snap("s-r1", "t-owner"));
    const before = kv.map.get(`${TRIPS_NS}trip:t-owner`);
    const r = s.remove("t-owner");
    expect(r.ok).toBe(true);
    expect(s.list()).toEqual([]);
    expect(kv.map.has(`${TRIPS_NS}snap:s-plan`)).toBe(false);
    expect(s.status().evicted).toBeNull();
    if (!r.ok) return;
    expect(r.undo().ok).toBe(true);
    expect(kv.map.get(`${TRIPS_NS}trip:t-owner`)).toBe(before);
    expect(s.getSnapshot("s-plan")?.kind).toBe("planning");
    expect(s.getSnapshot("s-r1")?.kind).toBe("recheck");
  });
});

describe("snapshots: planning kept forever, at most 8 rechecks", () => {
  it("a refreshed planning snapshot moves the old one to superseded; rechecks beyond 8 are pruned oldest first", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    s.putSnapshot(snap("p1", "t-owner", "planning"));
    s.putSnapshot(snap("p2", "t-owner", "planning"));
    for (let i = 1; i <= 11; i++) s.putSnapshot(snap(`r${i}`, "t-owner"));
    const t = s.get("t-owner")!;
    expect(t.snapshots.planning).toBe("p2");
    expect(t.snapshots.superseded).toEqual(["p1"]);
    expect(t.snapshots.rechecks).toHaveLength(MAX_RECHECKS);
    expect(t.snapshots.rechecks[0]).toBe("r4");
    expect(s.getSnapshot("r1")).toBeNull();
    expect(s.getSnapshot("r3")).toBeNull();
    expect(s.getSnapshot("p1")?.id).toBe("p1");
    expect(s.getSnapshot("p2")?.id).toBe("p2");
  });
});

describe("storage that fails", () => {
  it("a full storage is a typed failure, never a throw, and nothing half-written shows", () => {
    const kv = memory({ quota: 50_000 });
    const s = createTripsStore(kv, clockAt(T0));
    expect(s.put(ownerTrip()).ok).toBe(true);
    kv.quota = 10;
    const r = s.put(ownerTrip({ id: "t-two", name: "Second" }));
    expect(r).toEqual({ ok: false, reason: "quota" });
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
    expect(s.putSnapshot(snap("p", "t-owner", "planning"))).toEqual({ ok: false, reason: "quota" });
  });

  it("no storage at all: unavailable, reads empty", () => {
    const s = createTripsStore(null);
    expect(s.status().available).toBe(false);
    expect(s.list()).toEqual([]);
    expect(s.put(ownerTrip())).toEqual({ ok: false, reason: "unavailable" });
  });

  it("a corrupt trip key is kept and reported; its raw text is copied aside before it is overwritten", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    kv.map.set(`${TRIPS_NS}trip:t-bad`, "{not json");
    kv.map.set(`${TRIPS_NS}trip:t-shape`, JSON.stringify({ schema: 1, id: "t-shape" }));
    const st = s.status();
    expect(st.corrupt.sort()).toEqual(["trip:t-bad", "trip:t-shape"]);
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
    // another write leaves the damaged keys exactly as they were
    s.put(ownerTrip({ notes: "x" }));
    expect(kv.map.get(`${TRIPS_NS}trip:t-bad`)).toBe("{not json");
    // writing over one copies its text aside first
    s.put(ownerTrip({ id: "t-bad", name: "Rebuilt" }));
    expect(kv.map.get(`${TRIPS_NS}corrupt:trip:t-bad`)).toBe("{not json");
    expect(s.get("t-bad")?.name).toBe("Rebuilt");
  });

  it("a corrupt index is reported, the trips are still found by their keys, and the index text is kept aside", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    kv.map.set(`${TRIPS_NS}index`, "[[[");
    expect(s.status().corrupt).toContain("index");
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
    s.put(ownerTrip({ notes: "after" }));
    expect(kv.map.get(`${TRIPS_NS}corrupt:index`)).toBe("[[[");
  });

  it("a trip written by a newer schema makes the store read-only: shown nothing it cannot read, overwrites nothing", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    kv.map.set(`${TRIPS_NS}trip:t-future`, JSON.stringify({ schema: 99, id: "t-future", fishery: "x" }));
    const st = s.status();
    expect(st.readOnly).toBe(true);
    expect(st.newer).toBe(1);
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
    const before = new Map(kv.map);
    expect(s.put(ownerTrip({ notes: "x" }))).toEqual({ ok: false, reason: "readOnly" });
    expect(s.remove("t-owner")).toEqual({ ok: false, reason: "readOnly" });
    expect(s.importAll(s.exportAll(), { consent: true })).toEqual({ ok: false, reason: "readOnly" });
    expect(kv.map).toEqual(before);
  });

  it("a newer index schema is read-only too", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    kv.map.set(`${TRIPS_NS}index`, JSON.stringify({ schema: 2, ids: ["t-owner"] }));
    expect(s.status().readOnly).toBe(true);
  });
});

describe("eviction", () => {
  it("trip keys gone while the index still names them: reported, until acknowledged", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    s.put(ownerTrip({ id: "t-2", name: "Two" }));
    kv.map.delete(`${TRIPS_NS}trip:t-2`);
    expect(s.status().evicted).toEqual({ ids: ["t-2"], all: false, had: 2 });
    // a write does not forget the loss
    s.put(ownerTrip({ notes: "still" }));
    expect(s.status().evicted?.ids).toEqual(["t-2"]);
    expect(s.acknowledgeEviction().ok).toBe(true);
    expect(s.status().evicted).toBeNull();
  });

  it("everything gone after the device had trips (the marker survived): reported as all", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    kv.map.delete(`${TRIPS_NS}index`);
    kv.map.delete(`${TRIPS_NS}trip:t-owner`);
    expect(s.status().evicted).toEqual({ ids: [], all: true, had: 1 });
  });

  it("deleting by the reader is never an eviction", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    s.remove("t-owner");
    expect(s.status().evicted).toBeNull();
    s.put(ownerTrip());
    s.deleteAll();
    expect(s.status().evicted).toBeNull();
  });
});

describe("export, import and deletion", () => {
  it("exports every trip and its snapshots with the schema; imports into an empty device as they were", () => {
    const a = createTripsStore(memory(), clockAt(T0));
    a.put(ownerTrip());
    a.putSnapshot(snap("p1", "t-owner", "planning"));
    const json = a.exportAll();
    const parsed = JSON.parse(json);
    expect(parsed).toMatchObject({ app: "neritic-trips", schema: 1, snapshotSchema: 1 });
    expect(parsed.trips).toHaveLength(1);
    expect(parsed.snapshots.map((x: { id: string }) => x.id)).toEqual(["p1"]);
    const b = createTripsStore(memory(), clockAt(T0));
    const r = b.importAll(json, { consent: true });
    expect(r).toEqual({ ok: true, added: 1, already: 0, reassigned: 0, snapshots: 1, errors: [] });
    expect(b.get("t-owner")?.snapshots.planning).toBe("p1");
    expect(b.getSnapshot("p1")?.tripId).toBe("t-owner");
  });

  it("never imports without consent", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    expect(s.importAll("{}", { consent: false })).toEqual({ ok: false, reason: "consent" });
  });

  it("an id already here with different content is never overwritten: the copy gets a new id and \"(imported)\", snapshots follow it", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    s.put(ownerTrip());
    s.putSnapshot(snap("p1", "t-owner", "planning"));
    const json = s.exportAll();
    s.put({ ...s.get("t-owner")!, name: "Mine, edited" });
    const r = s.importAll(json, { consent: true });
    expect(r.ok && r.added).toBe(1);
    expect(r.ok && r.reassigned).toBe(1);
    const trips = s.list();
    expect(trips).toHaveLength(2);
    expect(s.get("t-owner")?.name).toBe("Mine, edited");
    expect(s.get("t-owner")?.snapshots.planning).toBe("p1");
    const copy = trips.find((t) => t.id !== "t-owner")!;
    expect(copy.name).toBe("Moss Landing crab (imported)");
    expect(copy.snapshots.planning).not.toBe("p1");
    expect(s.getSnapshot(copy.snapshots.planning!)?.tripId).toBe(copy.id);
  });

  it("invalid files and invalid records are refused with reasons; valid ones in the same file still come in", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    expect(s.importAll("not json", { consent: true })).toEqual({ ok: false, reason: "invalid" });
    expect(s.importAll(JSON.stringify({ app: "other", trips: [] }), { consent: true })).toEqual({ ok: false, reason: "invalid" });
    expect(s.importAll(JSON.stringify({ app: "neritic-trips", schema: 9, trips: [] }), { consent: true })).toEqual({ ok: false, reason: "newer" });
    const r = s.importAll(JSON.stringify({ app: "neritic-trips", schema: 1, trips: [ownerTrip(), { schema: 1, name: "No id" }, { schema: 7, id: "x", fishery: "groundfish", name: "Future" }], snapshots: [{ schema: 1, id: "junk" }] }), { consent: true });
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.added).toBe(1);
    expect(r.errors).toHaveLength(3);
    expect(r.errors.join(" ")).toMatch(/"No id".*not imported/);
    expect(r.errors.join(" ")).toMatch(/"Future".*newer/);
  });

  it("deleteAll removes every trip, snapshot, marker and damaged copy; the migration record stays", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    s.putSnapshot(snap("p1", "t-owner", "planning"));
    kv.map.set(`${TRIPS_NS}corrupt:index`, "x");
    kv.map.set(`${TRIPS_NS}migrated`, JSON.stringify({ ids: ["s-1"] }));
    kv.map.set("nx-trip-v1:scenarios", "[]");
    const r = s.deleteAll();
    expect(r.ok).toBe(true);
    expect([...kv.map.keys()].sort()).toEqual(["nx-trip-v1:scenarios", `${TRIPS_NS}migrated`]);
  });

  it("CSV: one row per trip, blanks for figures not recorded, text guarded against formulas", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    s.put(ownerTrip({ name: "=HYPERLINK(1)" }));
    const csv = s.exportCsv((k) => (k === "port-592" ? "Moss Landing" : null));
    const [head, row] = csv.trim().split("\r\n");
    expect(head.split(",")).toContain("planned_result");
    const cells = row.split(",");
    expect(cells[0]).toBe("'=HYPERLINK(1)");
    expect(cells[3]).toBe("Moss Landing");
    expect(cells).toContain("2602.24");
    expect(cells[6]).toBe(""); // landed lb: not recorded
    expect(csvCell('a "b", c')).toBe('"a ""b"", c"');
  });

  it("re-importing a backup adds nothing: identical trips (same id, or same content under another id) are already here", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    s.put(ownerTrip());
    s.putSnapshot(snap("p1", "t-owner", "planning"));
    const backup = s.exportAll();
    expect(s.previewImport(backup)).toEqual({ ok: true, fresh: 0, already: 1, copies: 0, invalid: 0, errors: [] });
    const r = s.importAll(backup, { consent: true });
    expect(r).toMatchObject({ ok: true, added: 0, already: 1, reassigned: 0 });
    expect(s.list()).toHaveLength(1);
    // the same trip under another id (exported from another device after a duplicate's id change) is still the same record
    const other = JSON.parse(backup);
    other.trips[0].id = "t-elsewhere";
    other.trips[0].updatedAt = "2026-12-01T00:00:00.000Z";
    expect(s.importAll(JSON.stringify(other), { consent: true })).toMatchObject({ ok: true, added: 0, already: 1 });
    expect(s.list()).toHaveLength(1);
  });

  it("the preview counts new, already here, copies and invalid before any write, and writes nothing", () => {
    const kv = memory();
    const s = createTripsStore(kv, clockAt(T0));
    s.put(ownerTrip());
    const file = JSON.stringify({ app: "neritic-trips", schema: 1, trips: [ownerTrip(), ownerTrip({ notes: "edited elsewhere" }), ownerTrip({ id: "t-new", name: "New one" }), { schema: 1, name: "Broken" }], snapshots: [] });
    const before = new Map(kv.map);
    const p = s.previewImport(file);
    expect(p).toMatchObject({ ok: true, fresh: 1, already: 1, copies: 1, invalid: 1 });
    expect(kv.map).toEqual(before);
    for (const [bad, reason] of [["{nope", "invalid"], [JSON.stringify({ app: "something-else", trips: [] }), "invalid"], [JSON.stringify({ app: "neritic-trips", schema: 5, trips: [] }), "newer"]] as const) expect(s.previewImport(bad)).toEqual({ ok: false, reason });
    expect(kv.map).toEqual(before);
    const r = s.importAll(file, { consent: true });
    expect(r).toMatchObject({ ok: true, added: 2, already: 1, reassigned: 1 });
    expect(s.list().map((t) => t.name).sort()).toEqual(["Moss Landing crab", "Moss Landing crab (imported)", "New one"]);
  });
});

describe("the pending undo outlives the list", async () => {
  const { getPendingUndo, runPendingUndo, setPendingUndo } = await import("@/nx/trips/undo");
  it("deleting the last trip leaves an Undo that puts it back", () => {
    const s = createTripsStore(memory(), clockAt(T0));
    s.put(ownerTrip());
    const r = s.remove("t-owner");
    if (!r.ok) throw new Error(r.reason);
    setPendingUndo({ name: r.trip.name, run: r.undo });
    expect(s.list()).toEqual([]);
    expect(getPendingUndo()?.name).toBe("Moss Landing crab");
    expect(runPendingUndo()?.ok).toBe(true);
    expect(getPendingUndo()).toBeNull();
    expect(s.list().map((t) => t.id)).toEqual(["t-owner"]);
  });
});
