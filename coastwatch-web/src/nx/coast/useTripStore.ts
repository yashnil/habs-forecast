"use client";

import { useMemo, useSyncExternalStore } from "react";
import { NS, createTripStore, type TripData, type TripStore } from "@/lib/tripEconomicsStore";

/**
 * One trip store per tab over localStorage ("nx-trip-v1:*"). Views re-read it when this tab writes (notify)
 * or another tab does (the storage event), so the workspace, the saved list and a second window agree.
 */
let store: TripStore | null = null;
let version = 0;
const listeners = new Set<() => void>();

function kv(): Storage | null {
  try {
    const s = window.localStorage;
    const probe = `${NS}probe`;
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

export function tripStore(): TripStore {
  if (!store) store = createTripStore(typeof window === "undefined" ? null : kv());
  return store;
}

export function notifyTrip() {
  version += 1;
  listeners.forEach((l) => l());
}

function onStorage(e: StorageEvent) {
  if (e.key === null || e.key.startsWith(NS)) notifyTrip();
}

function subscribe(l: () => void) {
  if (listeners.size === 0) window.addEventListener("storage", onStorage);
  listeners.add(l);
  return () => {
    listeners.delete(l);
    if (listeners.size === 0) window.removeEventListener("storage", onStorage);
  };
}

/** The stored trip data (null during server render and before hydration). */
export function useTripData(): TripData | null {
  const v = useSyncExternalStore(
    subscribe,
    () => version,
    () => -1,
  );
  return useMemo(() => (v < 0 ? null : tripStore().load()), [v]);
}
