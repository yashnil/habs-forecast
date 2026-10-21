"use client";

import { useMemo, useSyncExternalStore } from "react";
import { computeTrip } from "@/lib/tripEconomics";
import { NS, createTripStore, draftInputs, draftKey, type TripData, type TripStore } from "@/lib/tripEconomicsStore";

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

export type TripSummary = { name: string; costUsd: number | null; breakEvenLb: number | null; breakEvenUsdPerLb: number | null; netUsd: number | null };

/**
 * For My Coast's glance rows: the working numbers for this place and species (or, without them, the
 * latest saved scenario for both), worked through. Null when nothing has been entered.
 */
export function useTripSummary(placeKey: string | null | undefined, species: string | null | undefined): TripSummary | null {
  const data = useTripData();
  return useMemo(() => {
    if (!data || !placeKey || !species) return null;
    const working = data.meta.drafts[draftKey(placeKey, species)];
    const saved = data.scenarios.find((s) => s.placeKey === placeKey && s.species === species);
    const src = working ? { name: "Your working numbers", draft: working.draft } : saved ? { name: saved.name, draft: saved.draft } : null;
    if (!src) return null;
    const r = computeTrip(draftInputs(src.draft));
    const out: TripSummary = {
      name: src.name,
      costUsd: r.tripCostUsd,
      breakEvenLb: r.breakEvenLb.value,
      breakEvenUsdPerLb: r.breakEvenUsdPerLb.value,
      netUsd: r.netUsd,
    };
    return out.costUsd === null && out.breakEvenLb === null && out.breakEvenUsdPerLb === null && out.netUsd === null ? null : out;
  }, [data, placeKey, species]);
}
