"use client";

import { useMemo, useSyncExternalStore } from "react";
import { NS, createTripStore, draftKey, type Draft, type TripData, type TripStore } from "@/lib/tripEconomicsStore";
import { initialDraft } from "./draft";

/**
 * One trip store per tab over localStorage. Views re-read it when this tab writes (notify) or another
 * tab does (the storage event), so the worksheet, the glance rows and a second window agree.
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

const noop = () => () => {};
/** True only after hydration: the economics render from this device's storage, never from the server. */
export function useMounted(): boolean {
  return useSyncExternalStore(
    noop,
    () => true,
    () => false,
  );
}

/**
 * The worksheet a place and species opens with: the working draft kept on this device, else the latest
 * saved scenario for both, else a fresh one (nothing numeric filled in). Null before hydration.
 */
export function useWorkingDraft(placeKey: string, species: string): { draft: Draft; from: "working" | "saved" | "fresh"; name: string | null } | null {
  const data = useTripData();
  return useMemo(() => {
    if (!data) return null;
    const working = data.meta.drafts[draftKey(placeKey, species)];
    if (working) return { draft: working.draft, from: "working" as const, name: null };
    const saved = data.scenarios.find((s) => s.placeKey === placeKey && s.species === species);
    if (saved) return { draft: saved.draft, from: "saved" as const, name: saved.name };
    return { draft: initialDraft(species), from: "fresh" as const, name: null };
  }, [data, placeKey, species]);
}
