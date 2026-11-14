"use client";

import { useSyncExternalStore } from "react";
import type { Result } from "./store";

/**
 * The trip deleted last in this tab, until it is put back, dismissed or another is deleted. Held outside any view so
 * the Undo survives the history giving way to the empty state when the last trip goes.
 */
export type PendingUndo = { name: string; run: () => Result };

let pending: PendingUndo | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function setPendingUndo(u: PendingUndo | null) {
  pending = u;
  emit();
}

export const getPendingUndo = () => pending;

/** Put the deleted trip back; the pending undo is cleared either way. */
export function runPendingUndo(): Result | null {
  const u = pending;
  if (!u) return null;
  pending = null;
  const r = u.run();
  emit();
  return r;
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => void listeners.delete(l);
}

export function usePendingUndo(): PendingUndo | null {
  return useSyncExternalStore(subscribe, getPendingUndo, () => null);
}
