import type { FisheryId } from "@/pilot/contracts";
import { parseFisheries, type FactSnap } from "@/pilot/facts";

/**
 * What Today keeps on this device only (never sent anywhere): the last port opened, the fisheries
 * followed, and the facts seen at the last visit to each port. Every read and write survives storage
 * being blocked (private windows), and then simply remembers nothing.
 */
export const TODAY_FISHERIES_KEY = "nx-today-fisheries-v1";
export const TODAY_SEEN_PREFIX = "nx-today-seen-v1:";
export const TODAY_SAVED_KEY = "nx-today-saved-v1";

const get = (k: string): string | null => {
  try {
    return window.localStorage.getItem(k);
  } catch {
    return null;
  }
};
const set = (k: string, v: string | null) => {
  try {
    if (v == null) window.localStorage.removeItem(k);
    else window.localStorage.setItem(k, v);
    return true;
  } catch {
    return false;
  }
};

export { TODAY_PLACE_KEY, readTodayPlace, writeTodayPlace, readTodayHref } from "./navPlace";

export const readFisheries = (): FisheryId[] | null => parseFisheries(get(TODAY_FISHERIES_KEY));
export const writeFisheries = (ids: FisheryId[]) => set(TODAY_FISHERIES_KEY, ids.join(","));

export type Seen = { at: string; facts: FactSnap[] };
export function readSeen(place: string): Seen | null {
  const raw = get(TODAY_SEEN_PREFIX + place);
  if (!raw) return null;
  try {
    const o = JSON.parse(raw) as Seen;
    return typeof o.at === "string" && Array.isArray(o.facts) ? o : null;
  } catch {
    return null;
  }
}
export const writeSeen = (place: string, s: Seen) => set(TODAY_SEEN_PREFIX + place, JSON.stringify(s));

export function storageWorks(): boolean {
  try {
    const k = "nx-today-probe";
    window.localStorage.setItem(k, "1");
    window.localStorage.removeItem(k);
    return true;
  } catch {
    return false;
  }
}

export type SavedCopy = { place: string; label: string; at: string };
export function readSaved(): SavedCopy[] {
  try {
    const v = JSON.parse(get(TODAY_SAVED_KEY) ?? "[]") as SavedCopy[];
    return Array.isArray(v) ? v.filter((x) => x && typeof x.place === "string" && typeof x.at === "string") : [];
  } catch {
    return [];
  }
}
export const writeSaved = (list: SavedCopy[]) => set(TODAY_SAVED_KEY, JSON.stringify(list));
