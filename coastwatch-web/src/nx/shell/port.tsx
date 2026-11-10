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

/** The saved port key (nx-port-v1: the port last opened, else a home that is a port), or null. */
export function savedPortKey(): string | null {
  const r = readPortRecord();
  return r.lastPort ?? (isPortKey(r.place) ? r.place : null);
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

/** Make `key` the reader's port on this device (it stays in the browser). */
export function savePort(key: string) {
  if (!isPortKey(key)) return;
  updatePortRecord((r) => ({ ...r, lastPort: key }));
  notify();
}
