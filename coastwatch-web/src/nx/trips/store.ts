import { sanitizeDraft, type Draft, type KV } from "@/lib/tripEconomicsStore";
import type { RefSnapshot } from "@/content/referenceValues";
import type { FisheryId } from "@/pilot/contracts";
import { tripsCsv } from "./outcome";
import { SNAPSHOT_SCHEMA, TRIP_SCHEMA, type Activity, type Departure, type Lifecycle, type OperatingLimits, type Outcome, type Snapshot, type Trip } from "./types";

/**
 * The trips kept on this device (v15): localStorage namespace "nx-trips-v1:". Never sent anywhere; the only way
 * anything leaves the device is the reader's own export download.
 *
 * Keys:
 *   index          {schema, ids}: the trips, newest first, and the record of which trips there should be
 *   trip:<id>      one Trip
 *   snap:<id>      one Snapshot (planning, recheck or superseded planning)
 *   mark           {had, at}: how many trips this device held at the last write (so an eviction is noticed)
 *   corrupt:<key>  the raw text of a damaged key, copied aside once before anything replaces it
 *   migrated       which nx-trip-v1 scenarios have been copied in (migrate.ts), so none is copied twice
 *
 * The discipline of lib/tripEconomicsStore holds:
 * - a key or record written by a newer schema makes the store read-only here (nothing it does not understand
 *   is overwritten);
 * - a key that does not parse or validate is kept and reported; its raw text is copied aside before any write
 *   over it, so nothing is ever dropped;
 * - a full storage (quota) or a blocked one is a typed failure returned to the caller, never thrown at the UI.
 * Pure apart from the KV it is given (tests pass a Map-backed one).
 */

export const TRIPS_NS = "nx-trips-v1:";
export const INDEX_SCHEMA = 1;
/** earlier economics kept per trip (newest first) */
export const MAX_HISTORY = 10;
/** rechecks kept per trip; the planning snapshot and superseded plannings are never pruned */
export const MAX_RECHECKS = 8;
export const MAX_TRIP_NAME = 90;
export const EXPORT_APP = "neritic-trips";

export type { KV };

export type Failure = { ok: false; reason: "unavailable" | "readOnly" | "quota" | "missing" | "invalid" | "consent" | "newer" };
export type Done = { ok: true };
export type Result<T = object> = (Done & T) | Failure;

export type TripsStatus = {
  available: boolean;
  /** a newer version of Neritic wrote here: everything is shown, nothing is written */
  readOnly: boolean;
  /** keys whose text did not parse or validate (kept untouched; copied aside before any write over them) */
  corrupt: string[];
  /** trips or snapshots written by a newer schema (kept, not shown) */
  newer: number;
  /**
   * Trips this device held that are gone without the reader deleting them (the browser cleared storage).
   * `ids`: the index still names them; `all`: the index itself is gone after the device had trips.
   */
  evicted: { ids: string[]; all: boolean; had: number } | null;
};

export type TripsData = { trips: Trip[]; status: TripsStatus };

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, max = 80) => (typeof v === "string" ? v.slice(0, max) : "");
const strOrNull = (v: unknown, max = 80) => (typeof v === "string" && v !== "" ? v.slice(0, max) : null);
const iso = (v: unknown, fallback: string) => (typeof v === "string" && !Number.isNaN(Date.parse(v)) ? v : fallback);
/** a figure the reader typed: finite and not negative, else unknown (never 0) */
const fig = (v: unknown, max = 1e9): number | null => (typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= max ? v : null);
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const TIME = /^\d{2}:\d{2}$/;
const STATES: Lifecycle[] = ["draft", "planned", "completed", "cancelled"];

function sanitizeRefs(v: unknown): RefSnapshot[] {
  if (!Array.isArray(v)) return [];
  return v
    .filter(isObj)
    .filter((r) => typeof r.value === "number" && Number.isFinite(r.value))
    .slice(0, 20)
    .map((r) => ({ key: str(r.key), value: r.value as number, unit: str(r.unit, 12), source: str(r.source, 300), period: str(r.period, 80), fetchedAt: str(r.fetchedAt, 40) }));
}

function sanitizeDepart(v: unknown): Departure | null {
  if (!isObj(v) || typeof v.date !== "string" || !DATE.test(v.date)) return null;
  return { date: v.date, time: typeof v.time === "string" && TIME.test(v.time) ? v.time : null, timeZone: str(v.timeZone, 60) || "America/Los_Angeles" };
}

function sanitizeLimits(v: unknown): OperatingLimits {
  const o = isObj(v) ? v : {};
  return { windKt: fig(o.windKt, 200), seasFt: fig(o.seasFt, 100) };
}

export function sanitizeOutcome(v: unknown, now: string): Outcome | null {
  if (!isObj(v)) return null;
  const day = (x: unknown) => (typeof x === "string" && DATE.test(x) ? x : null);
  const recordedAt = iso(v.recordedAt, now);
  return {
    departedOn: day(v.departedOn),
    returnedOn: day(v.returnedOn),
    landed: (Array.isArray(v.landed) ? v.landed : []).filter(isObj).slice(0, 10).map((l) => ({ label: str(l.label, 60), lb: fig(l.lb), usdPerLb: fig(l.usdPerLb, 100000) })),
    fuelGal: fig(v.fuelGal, 100000),
    fuelUsdPerGal: fig(v.fuelUsdPerGal, 50),
    costs: (Array.isArray(v.costs) ? v.costs : []).filter(isObj).slice(0, 20).map((c) => ({ label: str(c.label, 60), usd: fig(c.usd) })),
    crewPaidUsd: fig(v.crewPaidUsd),
    notes: str(v.notes, 2000),
    conditions: str(v.conditions, 500),
    landingRef: strOrNull(v.landingRef, 60),
    recordedAt,
    updatedAt: iso(v.updatedAt, recordedAt),
  };
}

/**
 * One stored trip brought to the current schema. "newer" for a trip this version cannot read, null when it is not
 * a trip at all (it is then kept as it is and reported).
 */
export function sanitizeTrip(v: unknown, now: string): Trip | "newer" | null {
  if (!isObj(v)) return null;
  if (typeof v.schema !== "number" || !Number.isInteger(v.schema) || v.schema < 1) return null;
  if (v.schema > TRIP_SCHEMA) return "newer";
  if (typeof v.id !== "string" || v.id === "" || typeof v.fishery !== "string" || v.fishery === "") return null;
  const createdAt = iso(v.createdAt, now);
  const snaps = isObj(v.snapshots) ? v.snapshots : {};
  const ids = (x: unknown) => (Array.isArray(x) ? x.filter((i): i is string => typeof i === "string" && i !== "").map((i) => i.slice(0, 60)) : []);
  const history = (Array.isArray(v.history) ? v.history : [])
    .filter(isObj)
    .filter((h) => isObj(h.economics))
    .slice(0, MAX_HISTORY)
    .map((h) => ({ at: iso(h.at, createdAt), economics: sanitizeDraft(h.economics), refs: sanitizeRefs(h.refs) }));
  const state = STATES.includes(v.state as Lifecycle) ? (v.state as Lifecycle) : "draft";
  return {
    schema: TRIP_SCHEMA,
    id: v.id.slice(0, 60),
    name: str(v.name, MAX_TRIP_NAME) || "Trip",
    autoName: v.autoName === true,
    activity: (v.activity === "recreational" ? "recreational" : "commercial") as Activity,
    fishery: v.fishery.slice(0, 40) as FisheryId,
    placeKey: str(v.placeKey, 60),
    departKey: strOrNull(v.departKey, 60),
    depart: sanitizeDepart(v.depart),
    area: strOrNull(v.area, 200),
    vesselId: strOrNull(v.vesselId, 40),
    limits: sanitizeLimits(v.limits),
    economics: isObj(v.economics) ? sanitizeDraft(v.economics) : null,
    refs: sanitizeRefs(v.refs),
    history,
    notes: str(v.notes, 4000),
    state,
    createdAt,
    updatedAt: iso(v.updatedAt, createdAt),
    snapshots: { planning: strOrNull(snaps.planning, 60), rechecks: ids(snaps.rechecks), superseded: ids(snaps.superseded) },
    outcome: sanitizeOutcome(v.outcome, now),
    ...(typeof v.legacyId === "string" && v.legacyId ? { legacyId: v.legacyId.slice(0, 40) } : {}),
  };
}

/** A snapshot as stored: the shape is the snapshot module's; here only what identifies it is checked. */
export function checkSnapshot(v: unknown): Snapshot | "newer" | null {
  if (!isObj(v) || typeof v.schema !== "number") return null;
  if (v.schema > SNAPSHOT_SCHEMA) return "newer";
  if (typeof v.id !== "string" || !v.id || typeof v.tripId !== "string" || (v.kind !== "planning" && v.kind !== "recheck")) return null;
  if (!isObj(v.sources) || !Array.isArray(v.evidence) || !Array.isArray(v.limitations)) return null;
  return v as unknown as Snapshot;
}

/** A trip with nothing in it but what is given: the starting point for the planner, migration and tests. */
export function blankTrip(p: { id: string; fishery: FisheryId; placeKey: string; now: string } & Partial<Omit<Trip, "schema" | "id" | "fishery" | "placeKey">>): Trip {
  const { now, ...rest } = p;
  return {
    schema: TRIP_SCHEMA,
    name: "Trip",
    autoName: true,
    activity: "commercial",
    departKey: null,
    depart: null,
    area: null,
    vesselId: null,
    limits: { windKt: null, seasFt: null },
    economics: null,
    refs: [],
    history: [],
    notes: "",
    state: "draft",
    createdAt: now,
    updatedAt: now,
    snapshots: { planning: null, rechecks: [], superseded: [] },
    outcome: null,
    ...rest,
  };
}

const econKey = (d: Draft | null) => (d ? JSON.stringify(sanitizeDraft(d)) : "");

export type TripsStore = ReturnType<typeof createTripsStore>;

type Read = { text: string | null; parsed: unknown; corrupt: boolean };

export function createTripsStore(kv: KV | null, clock: () => Date = () => new Date()) {
  const nowIso = () => clock().toISOString();
  const K = { index: `${TRIPS_NS}index`, mark: `${TRIPS_NS}mark`, trip: (id: string) => `${TRIPS_NS}trip:${id}`, snap: (id: string) => `${TRIPS_NS}snap:${id}` };

  function read(key: string): Read {
    if (!kv) return { text: null, parsed: null, corrupt: false };
    let text: string | null = null;
    try {
      text = kv.getItem(key);
    } catch {
      return { text: null, parsed: null, corrupt: false };
    }
    if (text === null) return { text, parsed: null, corrupt: false };
    try {
      return { text, parsed: JSON.parse(text), corrupt: false };
    } catch {
      return { text, parsed: null, corrupt: true };
    }
  }

  function keys(): string[] {
    if (!kv || typeof kv.key !== "function" || typeof kv.length !== "number") return [];
    const out: string[] = [];
    try {
      for (let i = 0; i < kv.length; i++) {
        const k = kv.key(i);
        if (k && k.startsWith(TRIPS_NS)) out.push(k);
      }
    } catch {
      // a blocked enumeration: the index alone is used
    }
    return out;
  }

  type Index = { ids: string[]; state: "ok" | "absent" | "corrupt" | "newer" };
  function readIndex(): Index {
    const r = read(K.index);
    if (r.corrupt) return { ids: [], state: "corrupt" };
    if (r.parsed === null) return { ids: [], state: "absent" };
    if (!isObj(r.parsed) || !Array.isArray(r.parsed.ids)) return { ids: [], state: "corrupt" };
    if (typeof r.parsed.schema === "number" && r.parsed.schema > INDEX_SCHEMA) return { ids: [], state: "newer" };
    return { ids: r.parsed.ids.filter((i): i is string => typeof i === "string" && i !== ""), state: "ok" };
  }

  function readMark(): number {
    const r = read(K.mark);
    return isObj(r.parsed) && typeof r.parsed.had === "number" ? r.parsed.had : 0;
  }

  /** what load found, kept for writes: ids the index names that are gone (kept in the index until acknowledged) */
  let lost: string[] = [];

  function load(): TripsData {
    const status: TripsStatus = { available: kv !== null, readOnly: false, corrupt: [], newer: 0, evicted: null };
    const out: TripsData = { trips: [], status };
    if (!kv) return out;
    const now = nowIso();
    const idx = readIndex();
    if (idx.state === "newer") status.readOnly = true;
    if (idx.state === "corrupt") status.corrupt.push("index");
    // the index orders the trips; any trip key it does not name (a write cut short) is found and shown too
    const fromKeys = keys()
      .filter((k) => k.startsWith(`${TRIPS_NS}trip:`))
      .map((k) => k.slice(`${TRIPS_NS}trip:`.length));
    const ids = [...new Set([...idx.ids, ...fromKeys])];
    lost = [];
    for (const id of ids) {
      const r = read(K.trip(id));
      if (r.text === null) {
        if (idx.ids.includes(id)) lost.push(id);
        continue;
      }
      if (r.corrupt) {
        status.corrupt.push(`trip:${id}`);
        continue;
      }
      const t = sanitizeTrip(r.parsed, now);
      if (t === "newer") {
        status.newer += 1;
        status.readOnly = true;
      } else if (t === null) status.corrupt.push(`trip:${id}`);
      else out.trips.push(t);
    }
    const had = readMark();
    if (lost.length) status.evicted = { ids: lost, all: false, had };
    else if (idx.state === "absent" && had > 0 && out.trips.length === 0) status.evicted = { ids: [], all: true, had };
    out.trips.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return out;
  }

  /** a damaged value is copied aside before anything replaces it (once: the first copy is the original) */
  function aside(key: string) {
    if (!kv) return;
    const r = read(key);
    let bad = r.corrupt;
    if (!bad && r.parsed !== null) {
      if (key === K.index) bad = !isObj(r.parsed) || !Array.isArray(r.parsed.ids);
      else if (key.startsWith(`${TRIPS_NS}trip:`)) bad = sanitizeTrip(r.parsed, nowIso()) === null;
      else if (key.startsWith(`${TRIPS_NS}snap:`)) bad = checkSnapshot(r.parsed) === null;
    }
    if (!bad || r.text === null) return;
    const copy = `${TRIPS_NS}corrupt:${key.slice(TRIPS_NS.length)}`;
    if (kv.getItem(copy) === null) kv.setItem(copy, r.text);
  }

  function set(key: string, value: unknown): Result {
    if (!kv) return { ok: false, reason: "unavailable" };
    try {
      aside(key);
      kv.setItem(key, JSON.stringify(value));
      return { ok: true };
    } catch {
      return { ok: false, reason: "quota" };
    }
  }

  function del(key: string) {
    try {
      kv?.removeItem(key);
    } catch {
      // nothing to do: a key that cannot be removed stays, and the next load still reads it
    }
  }

  function guard(d: TripsData): Failure | null {
    if (!d.status.available) return { ok: false, reason: "unavailable" };
    if (d.status.readOnly) return { ok: false, reason: "readOnly" };
    return null;
  }

  /** the index and the eviction marker after a write: present trips newest first, still-unacknowledged lost ids after */
  function writeIndex(trips: Trip[]): Result {
    const ids = [...trips].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).map((t) => t.id);
    const r = set(K.index, { schema: INDEX_SCHEMA, ids: [...ids, ...lost.filter((id) => !ids.includes(id))] });
    if (!r.ok) return r;
    if (trips.length) set(K.mark, { had: trips.length, at: nowIso() });
    else del(K.mark);
    return r;
  }

  const newId = (p: string) => `${p}-${clock().getTime().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

  function getSnapshot(id: string): Snapshot | null {
    const r = read(K.snap(id));
    const s = r.corrupt ? null : checkSnapshot(r.parsed);
    return s === "newer" ? null : s;
  }

  /** Write a trip as given (no bump), and the index. */
  function writeTrip(d: TripsData, t: Trip): Result {
    const w = set(K.trip(t.id), t);
    if (!w.ok) return w;
    return writeIndex([t, ...d.trips.filter((x) => x.id !== t.id)]);
  }

  function rawSnapshots(t: Trip): Record<string, string> {
    const out: Record<string, string> = {};
    for (const id of [t.snapshots.planning, ...t.snapshots.rechecks, ...t.snapshots.superseded]) {
      if (!id) continue;
      const r = read(K.snap(id));
      if (r.text !== null) out[id] = r.text;
    }
    return out;
  }

  return {
    load,
    list: (): Trip[] => load().trips,
    status: (): TripsStatus => load().status,
    get: (id: string): Trip | null => load().trips.find((t) => t.id === id) ?? null,
    newId: () => newId("t"),

    /**
     * Save a trip. `updatedAt` becomes now; a trip that already exists keeps its id and creation time. When its
     * economics changed, the previous economics (and their references) go to the front of `history` (at most 10).
     */
    put(input: Trip): Result<{ trip: Trip }> {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const now = nowIso();
      const clean = sanitizeTrip({ ...input, schema: TRIP_SCHEMA }, now);
      if (!clean || clean === "newer") return { ok: false, reason: "invalid" };
      const prev = d.trips.find((t) => t.id === clean.id);
      let history = clean.history;
      if (prev) {
        history = prev.history;
        if (prev.economics && econKey(prev.economics) !== econKey(clean.economics)) history = [{ at: prev.updatedAt, economics: prev.economics, refs: prev.refs }, ...history].slice(0, MAX_HISTORY);
      }
      const trip: Trip = { ...clean, createdAt: prev?.createdAt ?? clean.createdAt, updatedAt: now, history };
      const r = writeTrip(d, trip);
      return r.ok ? { ok: true, trip } : r;
    },

    /** Delete a trip and its snapshots. `undo` puts back exactly what was there (raw), if nothing took its id meanwhile. */
    remove(id: string): Result<{ trip: Trip; undo: () => Result }> {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const trip = d.trips.find((t) => t.id === id);
      if (!trip) return { ok: false, reason: "missing" };
      const tripText = read(K.trip(id)).text;
      const snaps = rawSnapshots(trip);
      lost = lost.filter((x) => x !== id);
      const r = writeIndex(d.trips.filter((t) => t.id !== id));
      if (!r.ok) return r;
      del(K.trip(id));
      for (const s of Object.keys(snaps)) del(K.snap(s));
      const undo = (): Result => {
        const now = load();
        const g2 = guard(now);
        if (g2) return g2;
        if (now.trips.some((t) => t.id === id)) return { ok: true };
        if (!kv || tripText === null) return { ok: false, reason: "missing" };
        try {
          for (const [s, text] of Object.entries(snaps)) kv.setItem(K.snap(s), text);
          kv.setItem(K.trip(id), tripText);
        } catch {
          return { ok: false, reason: "quota" };
        }
        return writeIndex([...now.trips, trip]);
      };
      return { ok: true, trip, undo };
    },

    getSnapshot,

    /**
     * Store a snapshot and attach it to its trip: a planning snapshot replaces the trip's planning one, which moves
     * to `superseded` (kept, inspectable, never deleted); a recheck is appended and the oldest rechecks beyond 8 are
     * pruned. `attach: false` only writes the snapshot.
     */
    putSnapshot(snap: Snapshot, o: { attach?: boolean } = {}): Result<{ trip: Trip | null }> {
      const d = load();
      const g = guard(d);
      if (g) return g;
      if (checkSnapshot(snap) === null || checkSnapshot(snap) === "newer") return { ok: false, reason: "invalid" };
      const w = set(K.snap(snap.id), snap);
      if (!w.ok) return w;
      if (o.attach === false) return { ok: true, trip: null };
      const t = d.trips.find((x) => x.id === snap.tripId);
      if (!t) return { ok: true, trip: null };
      const s = { ...t.snapshots, rechecks: [...t.snapshots.rechecks], superseded: [...t.snapshots.superseded] };
      if (snap.kind === "planning") {
        if (s.planning && s.planning !== snap.id) s.superseded.push(s.planning);
        s.planning = snap.id;
      } else if (!s.rechecks.includes(snap.id)) s.rechecks.push(snap.id);
      const next: Trip = { ...t, snapshots: s };
      const r = writeTrip(d, next);
      if (!r.ok) return r;
      const p = this.pruneRechecks(t.id);
      return p.ok ? { ok: true, trip: p.trip ?? next } : p;
    },

    /** Keep the planning snapshot and every superseded one forever, and at most `max` rechecks (the newest). */
    pruneRechecks(tripId: string, max: number = MAX_RECHECKS): Result<{ trip: Trip | null; removed: string[] }> {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const t = d.trips.find((x) => x.id === tripId);
      if (!t) return { ok: false, reason: "missing" };
      const over = Math.max(0, t.snapshots.rechecks.length - max);
      if (!over) return { ok: true, trip: t, removed: [] };
      const removed = t.snapshots.rechecks.slice(0, over);
      const next: Trip = { ...t, snapshots: { ...t.snapshots, rechecks: t.snapshots.rechecks.slice(over) } };
      const r = writeTrip(d, next);
      if (!r.ok) return r;
      for (const id of removed) if (id !== t.snapshots.planning && !t.snapshots.superseded.includes(id)) del(K.snap(id));
      return { ok: true, trip: next, removed };
    },

    /** The reader has seen that trips were lost: stop reporting them. */
    acknowledgeEviction(): Result {
      const d = load();
      const g = guard(d);
      if (g) return g;
      lost = [];
      const r = writeIndex(d.trips);
      if (r.ok && !d.trips.length) del(K.mark);
      return r;
    },

    /** Everything this store holds, as JSON (the reader's own download). */
    exportAll(): string {
      const d = load();
      const snapshots: Snapshot[] = [];
      for (const t of d.trips) for (const id of [t.snapshots.planning, ...t.snapshots.superseded, ...t.snapshots.rechecks]) {
        const s = id ? getSnapshot(id) : null;
        if (s) snapshots.push(s);
      }
      return JSON.stringify({ app: EXPORT_APP, schema: TRIP_SCHEMA, snapshotSchema: SNAPSHOT_SCHEMA, exportedAt: nowIso(), trips: d.trips, snapshots }, null, 2);
    },

    /** One row per trip, planned against actual (a spreadsheet of the reader's own records). */
    exportCsv(portName?: (key: string) => string | null): string {
      return tripsCsv(load().trips, portName, csvCell);
    },

    /**
     * What an import of this file would do, without writing anything: the counts the consent dialog states (new trips,
     * trips already on this device, trips that come in as copies, records that cannot be read) or why the file cannot
     * be imported at all. A file that fails here never reaches the consent.
     */
    previewImport(json: string): ImportPreview {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const p = planImport(json, d.trips, nowIso(), newId);
      if (!p.ok) return p;
      return { ok: true, fresh: p.fresh, already: p.already, copies: p.copies, invalid: p.invalid, errors: p.errors };
    },

    /**
     * Bring in an export. Only with the reader's explicit consent. Nothing here is overwritten:
     * - a trip identical to one already here (the same id with the same content, or the same content under another
     *   id) is skipped and counted as already on this device;
     * - a trip whose id is used here by a different trip comes in as a copy with a new id, named "… (imported)";
     * - every record is validated; what fails is counted and said, never half-written.
     */
    importAll(json: string, o: { consent: boolean }): Result<{ added: number; already: number; reassigned: number; snapshots: number; errors: string[] }> {
      if (o.consent !== true) return { ok: false, reason: "consent" };
      const d = load();
      const g = guard(d);
      if (g) return g;
      const p = planImport(json, d.trips, nowIso(), newId);
      if (!p.ok) return p;
      const errors = [...p.errors];
      let trips = [...d.trips];
      let added = 0;
      let reassigned = 0;
      let snapshots = 0;
      for (const e of p.entries) {
        if (e.kind === "already") continue;
        // snapshots first, then the trip: a write cut short leaves orphan snapshots, never a trip pointing at nothing
        let failed = false;
        for (const snap of e.snapshots) {
          const w = set(K.snap(snap.id), snap);
          if (!w.ok) {
            failed = true;
            break;
          }
        }
        const w = failed ? ({ ok: false, reason: "quota" } as Failure) : set(K.trip(e.trip.id), e.trip);
        if (!w.ok) {
          errors.push(`${e.label} and the trips after it did not fit in this browser's storage.`);
          break;
        }
        snapshots += e.snapshots.length;
        trips = [e.trip, ...trips];
        added += 1;
        if (e.kind === "copy") reassigned += 1;
      }
      const r = writeIndex(trips);
      if (!r.ok) return r;
      return { ok: true, added, already: p.already, reassigned, snapshots, errors };
    },

    /** Delete every trip and snapshot on this device, and the copies of damaged data. Which legacy scenarios were already copied is kept (ids only), so deleted trips do not come back from the old store. */
    deleteAll(): Result<{ removed: number }> {
      if (!kv) return { ok: false, reason: "unavailable" };
      const all = new Set([...keys(), K.index, K.mark]);
      const d = load();
      for (const t of d.trips) {
        all.add(K.trip(t.id));
        for (const id of [t.snapshots.planning, ...t.snapshots.rechecks, ...t.snapshots.superseded]) if (id) all.add(K.snap(id));
      }
      all.delete(`${TRIPS_NS}migrated`);
      let removed = 0;
      for (const k of all) {
        try {
          if (kv.getItem(k) !== null) removed += 1;
          kv.removeItem(k);
        } catch {
          return { ok: false, reason: "quota" };
        }
      }
      lost = [];
      return { ok: true, removed };
    },

    /** raw access for migrate.ts (same namespace, same failure discipline) */
    readKey: (suffix: string) => read(TRIPS_NS + suffix),
    writeKey: (suffix: string, value: unknown) => set(TRIPS_NS + suffix, value),
    /** write a trip exactly as given (migration and import keep their own timestamps) */
    restore(t: Trip): Result {
      const d = load();
      const g = guard(d);
      if (g) return g;
      return writeTrip(d, t);
    },
  };
}

export type ImportPreview = (Done & { fresh: number; already: number; copies: number; invalid: number; errors: string[] }) | Failure;

type ImportEntry = { kind: "new" | "copy" | "already"; label: string; trip: Trip; snapshots: Snapshot[] };
type ImportPlan = (Done & { entries: ImportEntry[]; fresh: number; already: number; copies: number; invalid: number; errors: string[] }) | Failure;

/** What makes two trips the same record: everything but their ids, snapshot ids and timestamps. */
export function tripContentKey(t: Trip): string {
  return JSON.stringify({ ...t, id: null, snapshots: null, createdAt: null, updatedAt: null, legacyId: null });
}

/** "Name (imported)", "Name (imported 2)", …: never a name already used. */
function importedName(name: string, taken: Set<string>): string {
  const fit = (suffix: string) => `${name.slice(0, MAX_TRIP_NAME - suffix.length)}${suffix}`;
  let out = fit(" (imported)");
  for (let n = 2; taken.has(out) && n < 100; n++) out = fit(` (imported ${n})`);
  return out;
}

/** Parse, validate and classify an export against the trips already here. Pure: writes nothing. */
export function planImport(json: string, existing: readonly Trip[], now: string, newId: (p: string) => string): ImportPlan {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return { ok: false, reason: "invalid" };
  }
  if (!isObj(parsed) || parsed.app !== EXPORT_APP || !Array.isArray(parsed.trips)) return { ok: false, reason: "invalid" };
  if (typeof parsed.schema === "number" && parsed.schema > TRIP_SCHEMA) return { ok: false, reason: "newer" };
  const errors: string[] = [];
  const snapsIn = new Map<string, Snapshot>();
  for (const [i, s] of (Array.isArray(parsed.snapshots) ? parsed.snapshots : []).entries()) {
    const c = checkSnapshot(s);
    if (c === "newer") errors.push(`Snapshot ${i + 1} was written by a newer version and was not imported.`);
    else if (c === null) errors.push(`Snapshot ${i + 1} is not a snapshot Neritic can read and was not imported.`);
    else snapsIn.set(c.id, c);
  }
  const byId = new Map(existing.map((t) => [t.id, t]));
  const contents = new Set(existing.map(tripContentKey));
  const names = new Set(existing.map((t) => t.name));
  const takenSnaps = new Set(existing.flatMap((t) => [t.snapshots.planning, ...t.snapshots.rechecks, ...t.snapshots.superseded]).filter((x): x is string => !!x));
  const entries: ImportEntry[] = [];
  let invalid = 0;
  for (const [i, raw] of parsed.trips.entries()) {
    const t = sanitizeTrip(raw, now);
    const label = isObj(raw) && typeof raw.name === "string" ? `"${raw.name.slice(0, 40)}"` : `Trip ${i + 1}`;
    if (t === "newer" || t === null) {
      invalid += 1;
      errors.push(t === "newer" ? `${label} was written by a newer version and was not imported.` : `${label} is missing what a trip needs (an id and a fishery) and was not imported.`);
      continue;
    }
    const key = tripContentKey(t);
    if (contents.has(key)) {
      entries.push({ kind: "already", label, trip: t, snapshots: [] });
      continue;
    }
    const clash = byId.has(t.id);
    const id = clash ? newId("t") : t.id;
    const remap = (sid: string) => (clash || takenSnaps.has(sid) ? newId("s") : sid);
    const snaps: Snapshot[] = [];
    const keep = (sid: string | null) => {
      if (!sid) return null;
      const s = snapsIn.get(sid);
      if (!s || s.tripId !== t.id) return null;
      const nid = remap(sid);
      takenSnaps.add(nid);
      snaps.push({ ...s, id: nid, tripId: id });
      return nid;
    };
    const name = clash ? importedName(t.name, names) : t.name;
    const next: Trip = {
      ...t,
      id,
      name,
      ...(clash ? { autoName: false } : {}),
      snapshots: { planning: keep(t.snapshots.planning), rechecks: t.snapshots.rechecks.map(keep).filter((x): x is string => !!x), superseded: t.snapshots.superseded.map(keep).filter((x): x is string => !!x) },
    };
    byId.set(id, next);
    contents.add(key);
    names.add(name);
    entries.push({ kind: clash ? "copy" : "new", label, trip: next, snapshots: snaps });
  }
  const count = (k: ImportEntry["kind"]) => entries.filter((e) => e.kind === k).length;
  return { ok: true, entries, fresh: count("new"), already: count("already"), copies: count("copy"), invalid, errors };
}

/** One CSV row per trip, planned against actual (outcome.ts writes the figures). Text cells are escaped against formulas. */
export function csvCell(v: string | number | null): string {
  if (v === null) return "";
  if (typeof v === "number") return Number.isFinite(v) ? String(Math.round(v * 100) / 100) : "";
  const safe = /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}
