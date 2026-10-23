"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useSyncExternalStore, type ReactNode } from "react";

/**
 * Readiness: what is ready, what is still loading, whether what is shown is fresh or cached,
 * whether the device is online, and what the reader can do next. Each part of the app reports its
 * own source; the status centre and the initial shell read them. No progress percentages are
 * invented: a source is in one of these states, with its own words.
 */
export type SourceState =
  | "waiting" // not started (shown only in the detail list)
  | "loading"
  | "ready"
  | "partial" // usable, part of it missing
  | "stale" // ready, but older than its source's normal cadence
  | "cached" // shown from a saved copy (offline or a failed refresh)
  | "unavailable" // failed; nothing current to show
  | "downloading"
  | "incomplete"; // a download stopped part-way

export type SourceId = "map" | "data" | "place" | "conditions" | "notices" | "model" | "satellite" | "currents" | "evidence" | "pack";

export type SourceAction = { label: string; run: () => void };

export type SourceEntry = {
  id: SourceId;
  /** what this source is, in the reader's words ("Coastline and map") */
  label: string;
  state: SourceState;
  /** one line on what is happening or what is shown ("Run of Oct 14, valid Oct 16") */
  detail?: string;
  /** ISO time of the information shown (observation, issue or last check) */
  asOf?: string | null;
  actions?: SourceAction[];
};

/** The app's overall state, derived from its sources and the connection. */
export type Overall = "initializing" | "partial" | "ready" | "stale" | "offline" | "unavailable" | "downloading" | "incomplete" | "outdated";

type Store = {
  get: () => Snapshot;
  subscribe: (fn: () => void) => () => void;
  set: (e: SourceEntry) => void;
  clear: (id: SourceId) => void;
  setOnline: (v: boolean) => void;
};
type Snapshot = { sources: Partial<Record<SourceId, SourceEntry>>; online: boolean; version: number };

function createStore(): Store {
  let snap: Snapshot = { sources: {}, online: true, version: 0 };
  const subs = new Set<() => void>();
  const emit = () => subs.forEach((f) => f());
  return {
    get: () => snap,
    subscribe: (fn) => {
      subs.add(fn);
      return () => subs.delete(fn);
    },
    set: (e) => {
      const prev = snap.sources[e.id];
      if (prev && prev.state === e.state && prev.detail === e.detail && prev.asOf === e.asOf && prev.label === e.label && prev.actions === e.actions) return;
      snap = { ...snap, sources: { ...snap.sources, [e.id]: e }, version: snap.version + 1 };
      emit();
    },
    clear: (id) => {
      if (!snap.sources[id]) return;
      const sources = { ...snap.sources };
      delete sources[id];
      snap = { ...snap, sources, version: snap.version + 1 };
      emit();
    },
    setOnline: (v) => {
      if (snap.online === v) return;
      snap = { ...snap, online: v, version: snap.version + 1 };
      emit();
    },
  };
}

const Ctx = createContext<Store | null>(null);
const SERVER: Snapshot = { sources: {}, online: true, version: 0 };

export function ReadinessProvider({ children }: { children: ReactNode }) {
  const store = useMemo(() => createStore(), []);
  useEffect(() => {
    const on = () => store.setOnline(navigator.onLine);
    on();
    window.addEventListener("online", on);
    window.addEventListener("offline", on);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", on);
    };
  }, [store]);
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("ReadinessProvider missing");
  return s;
}

export function useReadinessSnapshot(): Snapshot {
  const s = useStore();
  return useSyncExternalStore(s.subscribe, s.get, () => SERVER);
}

/** Report one source; cleared when the component that reports it leaves, or when it reports null (a
 *  page that stops showing a source, such as a briefing closed back to its landing, no longer speaks for it). */
export function useReportSource(entry: SourceEntry | null) {
  const s = useStore();
  const id = useRef<SourceId | null>(null);
  useEffect(() => {
    if (entry) {
      if (id.current && id.current !== entry.id) s.clear(id.current);
      id.current = entry.id;
      s.set(entry);
    } else if (id.current) {
      s.clear(id.current);
      id.current = null;
    }
  });
  useEffect(() => () => {
    if (id.current) s.clear(id.current);
  }, [s]);
}

export function useSetSource() {
  const s = useStore();
  return useCallback((e: SourceEntry) => s.set(e), [s]);
}

const ORDER: SourceId[] = ["map", "data", "place", "notices", "model", "satellite", "currents", "conditions", "evidence", "pack"];

export function sortedSources(snap: Snapshot): SourceEntry[] {
  return ORDER.map((id) => snap.sources[id]).filter((e): e is SourceEntry => !!e);
}

/**
 * One state for the whole app, by priority: offline first (it changes what everything means),
 * then a download in progress or incomplete, a failed source, loading (nothing usable yet vs.
 * partly ready), outdated or stale data, ready.
 */
export function overall(snap: Snapshot): Overall {
  const list = sortedSources(snap);
  if (!snap.online) return "offline";
  if (list.some((e) => e.state === "downloading")) return "downloading";
  if (list.some((e) => e.state === "incomplete")) return "incomplete";
  const core = list.filter((e) => e.id === "map" || e.id === "data");
  if (!core.length || core.some((e) => e.state === "loading" || e.state === "waiting")) {
    return list.some((e) => e.state === "ready" || e.state === "stale" || e.state === "partial") ? "partial" : "initializing";
  }
  if (list.some((e) => e.state === "unavailable")) return "unavailable";
  if (list.some((e) => e.state === "loading")) return "partial";
  if (list.some((e) => e.state === "cached")) return "outdated";
  if (list.some((e) => e.state === "stale" || e.state === "partial")) return "stale";
  return "ready";
}

export const OVERALL_WORDS: Record<Overall, string> = {
  initializing: "Starting",
  partial: "Partly ready",
  ready: "Ready",
  stale: "Some data is old",
  offline: "Offline",
  unavailable: "A source is unavailable",
  downloading: "Downloading",
  incomplete: "Download incomplete",
  outdated: "Showing saved data",
};

export const STATE_WORDS: Record<SourceState, string> = {
  waiting: "Waiting",
  loading: "Loading",
  ready: "Ready",
  partial: "Partly available",
  stale: "Older than usual",
  cached: "Saved copy",
  unavailable: "Unavailable",
  downloading: "Downloading",
  incomplete: "Incomplete",
};
