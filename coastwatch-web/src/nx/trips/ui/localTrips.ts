import { checkSnapshot, sanitizeTrip, TRIPS_NS } from "../store";
import type { Snapshot, Trip } from "../types";

/**
 * Read-only access to one trip and its snapshots in this device's storage (localStorage "nx-trips-v1:", one key per
 * trip `trip:<id>` and per snapshot `snap:<id>`), for the briefing: it opens with no network and never writes. A value
 * that does not parse, or was written by a newer schema, is reported as such, never repaired or dropped here.
 */
export const NS = TRIPS_NS;

export type Read<T> = { state: "ok"; value: T } | { state: "missing" } | { state: "unreadable" } | { state: "newer" };

type KV = Pick<Storage, "getItem">;

const storage = (): KV | null => {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
};

function readJson<T>(kv: KV | null, key: string, check: (v: unknown) => T | "newer" | null): Read<T> {
  if (!kv) return { state: "unreadable" };
  let raw: string | null;
  try {
    raw = kv.getItem(NS + key);
  } catch {
    return { state: "unreadable" };
  }
  if (raw == null) return { state: "missing" };
  try {
    const v = check(JSON.parse(raw));
    return v === "newer" ? { state: "newer" } : v ? { state: "ok", value: v } : { state: "unreadable" };
  } catch {
    return { state: "unreadable" };
  }
}

/** Validated as the trips store validates (store.ts sanitizeTrip, checkSnapshot); never written back from here. */
export const readTrip = (id: string, kv: KV | null = storage()) => readJson<Trip>(kv, `trip:${id}`, (v) => sanitizeTrip(v, new Date().toISOString()));
export const readSnapshot = (id: string, kv: KV | null = storage()) => readJson<Snapshot>(kv, `snap:${id}`, checkSnapshot);

/** The trip's planning snapshot and its newest recheck (null where none is saved or readable). */
export function tripSnapshots(trip: Trip, kv: KV | null = storage()): { planning: Snapshot | null; latest: Snapshot | null } {
  const get = (id: string | null) => {
    if (!id) return null;
    const r = readSnapshot(id, kv);
    return r.state === "ok" ? r.value : null;
  };
  const planning = get(trip.snapshots.planning);
  let latest: Snapshot | null = null;
  for (const id of [...trip.snapshots.rechecks].reverse()) {
    latest = get(id);
    if (latest) break;
  }
  return { planning, latest };
}

/** Every readable snapshot of the trip: planning, rechecks and superseded plans (what knownView reads). */
export function allSnapshots(trip: Trip, kv: KV | null = storage()): Snapshot[] {
  const ids = [trip.snapshots.planning, ...trip.snapshots.superseded, ...trip.snapshots.rechecks].filter((x): x is string => !!x);
  const out: Snapshot[] = [];
  for (const id of [...new Set(ids)]) {
    const r = readSnapshot(id, kv);
    if (r.state === "ok") out.push(r.value);
  }
  return out;
}
