"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { FisheryId } from "@/pilot/contracts";
import { parseFisheries } from "@/pilot/facts";
import { PORT_KEY, readPortRecord, updatePortRecord } from "@/lib/portRecord";

/**
 * What Today keeps on this device. The fisheries live in the saved-port record (nx-port-v1, lib/portRecord),
 * shared with My Coast and the account page; the reader's first fishery orders the page. Nothing is chosen
 * for the reader: an empty list means Today asks. The tour's dismissal is its own key.
 */
const EVENT = "nx-today-store";
const TOUR_KEY = "nx-tour-v1";

function subscribe(l: () => void) {
  const onStorage = (e: StorageEvent) => {
    if (e.key === PORT_KEY || e.key === TOUR_KEY) l();
  };
  window.addEventListener("storage", onStorage);
  window.addEventListener(EVENT, l);
  return () => {
    window.removeEventListener("storage", onStorage);
    window.removeEventListener(EVENT, l);
  };
}
const notify = () => window.dispatchEvent(new Event(EVENT));

const fisheriesKey = () => readPortRecord().fisheries.join(",");

/** The reader's fisheries, in their order; undefined on the server and during hydration. */
export function useFisheries(): FisheryId[] | undefined {
  const key = useSyncExternalStore(subscribe, fisheriesKey, () => undefined);
  return useMemo(() => (key === undefined ? undefined : (parseFisheries(key) ?? [])), [key]);
}

/** Make `id` the fishery Today leads with; the others the reader followed stay after it. */
export function chooseFishery(id: FisheryId) {
  updatePortRecord((r) => ({ ...r, fisheries: [id, ...r.fisheries.filter((f) => f !== id)] }));
  notify();
}

/** Forget the fisheries (the chooser asks again). */
export function clearFisheries() {
  updatePortRecord((r) => ({ ...r, fisheries: [] }));
  notify();
}

const tourState = () => {
  try {
    return window.localStorage.getItem(TOUR_KEY) ?? "";
  } catch {
    return "dismissed";
  }
};

/** "" never seen, "dismissed" gone for good; undefined until hydrated. */
export function useTourState(): string | undefined {
  return useSyncExternalStore(subscribe, tourState, () => undefined);
}

export function dismissTour() {
  try {
    window.localStorage.setItem(TOUR_KEY, "dismissed");
  } catch {
    // this page view only
  }
  notify();
}
