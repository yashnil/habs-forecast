"use client";

import { useSyncExternalStore } from "react";

/**
 * The count beside "Alerts" in the bar: notices and data updates the reader has not seen. The shell only shows
 * it; whoever knows it (the alerts feature) sets it with `setUnreadCount`. Zero, the default, shows nothing,
 * so the bar never implies monitoring that is not happening.
 */
let count = 0;
const listeners = new Set<() => void>();

export function setUnreadCount(n: number) {
  const next = Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
  if (next === count) return;
  count = next;
  listeners.forEach((l) => l());
}

export function useUnread(): number {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => count,
    () => 0,
  );
}

/** Where the bar's Alerts link goes, carrying the port. */
export const alertsHref = (key: string | null) => (key ? `/alerts?place=${encodeURIComponent(key)}` : "/alerts");
