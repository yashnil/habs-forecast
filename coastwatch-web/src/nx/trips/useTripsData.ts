"use client";

import { useMemo, useSyncExternalStore } from "react";
import { migrateLegacy, type MigrationReport } from "./migrate";
import { TRIPS_NS, createTripsStore, type TripsData, type TripsStore } from "./store";

/**
 * One trips store per tab over localStorage ("nx-trips-v1:*"). Views re-read it when this tab writes (`notifyTrips`)
 * or another tab does (the storage event), so the workspace, the history and a second window agree. The first read
 * in a tab copies any saved nx-trip-v1 scenarios in (migrate.ts: copy only, once).
 */
let store: TripsStore | null = null;
let local: Storage | null = null;
let migration: MigrationReport | null = null;
let version = 0;
const listeners = new Set<() => void>();

function kv(): Storage | null {
  try {
    const s = window.localStorage;
    const probe = `${TRIPS_NS}probe`;
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    // a full storage still reads: use it read-and-report rather than pretending it is not there
    try {
      return window.localStorage ?? null;
    } catch {
      return null;
    }
  }
}

export function tripsStore(): TripsStore {
  if (!store) {
    local = typeof window === "undefined" ? null : kv();
    store = createTripsStore(local);
  }
  return store;
}

/** What the first read in this tab did with the old saved scenarios (null before it ran, or during server render). */
export function migrationReport(): MigrationReport | null {
  if (!migration && typeof window !== "undefined") {
    const s = tripsStore();
    migration = migrateLegacy(local, s);
  }
  return migration;
}

export function notifyTrips() {
  version += 1;
  listeners.forEach((l) => l());
}

function onStorage(e: StorageEvent) {
  if (e.key === null || e.key.startsWith(TRIPS_NS)) notifyTrips();
}

function subscribe(l: () => void) {
  if (listeners.size === 0) window.addEventListener("storage", onStorage);
  listeners.add(l);
  return () => {
    listeners.delete(l);
    if (listeners.size === 0) window.removeEventListener("storage", onStorage);
  };
}

/** The stored trips (null during server render and before hydration). */
export function useTripsData(): TripsData | null {
  const v = useSyncExternalStore(
    subscribe,
    () => version,
    () => -1,
  );
  return useMemo(() => {
    if (v < 0) return null;
    migrationReport();
    return tripsStore().load();
  }, [v]);
}
