"use client";

import { useSyncExternalStore } from "react";
import type { Progress } from "./packs";

/**
 * A field pack's download, kept outside any component: closing the pack dialog does not stop or forget
 * it, and reopening the dialog shows the same progress. `version` counts changes to the saved pack (a
 * download finished, the pack removed) so every view of it reads the record again.
 */
export type RunState = "idle" | "downloading" | "done" | "failed" | "quota" | "cancelled";
export type Run = { state: RunState; prog: Progress | null; ac: AbortController | null; version: number };

const IDLE: Run = { state: "idle", prog: null, ac: null, version: 0 };
const runs = new Map<string, Run>();
const listeners = new Set<() => void>();

export function getRun(id: string): Run {
  return runs.get(id) ?? IDLE;
}

export function setRun(id: string, patch: Partial<Omit<Run, "version">>) {
  runs.set(id, { ...getRun(id), ...patch });
  listeners.forEach((l) => l());
}

/** The saved pack changed on this device: views of it read its record again. */
export function packChanged(id: string) {
  const r = getRun(id);
  runs.set(id, { ...r, version: r.version + 1 });
  listeners.forEach((l) => l());
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};

export function useRun(id: string | null): Run {
  return useSyncExternalStore(
    subscribe,
    () => (id ? getRun(id) : IDLE),
    () => IDLE,
  );
}
