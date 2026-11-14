"use client";

import { createContext, useContext, useSyncExternalStore } from "react";
import { useSearchParams } from "next/navigation";
import { isPortKey, PORT_KEY, readPortRecord, updatePortRecord } from "@/lib/portRecord";

export type ShellPort = { key: string; name: string; lon: number; lat: number; pilot: boolean; county: string | null };

const PortsContext = createContext<ShellPort[]>([]);
export const PortsProvider = PortsContext.Provider;
export const usePorts = () => useContext(PortsContext);

// the saved record, as an external store: the tab it was changed in and every other tab (storage events)
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
function subscribe(l: () => void) {
  listeners.add(l);
  const onStorage = (e: StorageEvent) => {
    if (e.key === PORT_KEY) l();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(l);
    window.removeEventListener("storage", onStorage);
  };
}

/** The home port the reader set (nx-port-v1 `place`, when it is a port), or null. Only `setHomePort` and a first choice change it. */
export function homePortKey(): string | null {
  const r = readPortRecord();
  return isPortKey(r.place) ? r.place : null;
}

/** The saved port key: the home port, else the port last opened, or null. `/` opens this one. */
export function savedPortKey(): string | null {
  return homePortKey() ?? readPortRecord().lastPort;
}

/** null on the server and during hydration (the first client render must match the server's HTML). */
export function useSavedPortKey(): string | null | undefined {
  return useSyncExternalStore(subscribe, savedPortKey, () => undefined);
}

/**
 * The port this page is about: the address's `place` (a deep link wins), else the saved port. The shell's
 * port control, every tab's link and each page's scene read this, so the port travels across the four pages.
 */
export function useCurrentPort(): { port: ShellPort | null; key: string | null; fromUrl: boolean; pending: boolean } {
  const sp = useSearchParams();
  const ports = usePorts();
  const saved = useSavedPortKey();
  const urlKey = sp.get("place");
  const key = isPortKey(urlKey) ? urlKey : (saved ?? null);
  return { port: ports.find((p) => p.key === key) ?? null, key, fromUrl: isPortKey(urlKey), pending: saved === undefined && !isPortKey(urlKey) };
}

/** undefined on the server and during hydration, then the home port or null. */
export function useHomePortKey(): string | null | undefined {
  return useSyncExternalStore(subscribe, homePortKey, () => undefined);
}

/** The reader chose `key` themselves when the address does not already name it (a deep link's page reports the port its address names). */
function chosenHere(key: string): boolean {
  try {
    return new URLSearchParams(window.location.search).get("place") !== key;
  } catch {
    return false;
  }
}

/**
 * `key` is the port last opened on this device (it stays in the browser). The first port a reader chooses
 * themselves also becomes their home port; after that the home port changes only through `setHomePort`, so
 * opening a link to another port never moves it.
 */
export function savePort(key: string, opts: { chosen?: boolean } = {}) {
  if (!isPortKey(key)) return;
  const chosen = opts.chosen ?? chosenHere(key);
  updatePortRecord((r) => ({ ...r, lastPort: key, ...(chosen && r.place === null ? { place: key } : {}) }));
  notify();
}

/** The reader picked `key` in a port finder. */
export const choosePort = (key: string) => savePort(key, { chosen: true });

/** Set or clear the home port. A home that is a harvest site or a point (an earlier My Coast) is replaced only by a port, never cleared here. */
export function setHomePort(key: string | null): boolean {
  if (key !== null && !isPortKey(key)) return false;
  const kept = updatePortRecord((r) => {
    if (key === null) return isPortKey(r.place) ? { ...r, place: null, for: "all", species: "any" } : r;
    return r.place === key ? r : { ...r, place: key, for: "all", species: "any" };
  });
  notify();
  return kept;
}
