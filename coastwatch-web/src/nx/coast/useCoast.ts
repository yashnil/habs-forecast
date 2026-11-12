"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { readPortRecord } from "@/lib/portRecord";
import type { FisheryId } from "@/pilot/contracts";
import type { CoastError, CoastGates, CoastSea } from "./brief";
import { defaultFishery, tripFishery } from "./fisheries";

export type Loadable<T> = { state: "idle" } | { state: "loading" } | { state: "ok"; data: T } | { state: "error"; error: string };

/** One request per key; a newer key cancels an older answer (a quick change of fishery never shows the old gates). */
function useJson<T extends { ok: true }>(url: string | null): Loadable<T> {
  const [v, setV] = useState<{ url: string | null; value: Loadable<T> }>({ url: null, value: { state: "idle" } });
  useEffect(() => {
    if (!url) return;
    let live = true;
    fetch(url)
      .then(async (r) => {
        const body = (await r.json().catch(() => null)) as T | CoastError | null;
        if (!live) return;
        if (body && body.ok) setV({ url, value: { state: "ok", data: body } });
        else setV({ url, value: { state: "error", error: body && !body.ok ? body.error : `HTTP ${r.status}` } });
      })
      .catch((e: Error) => live && setV({ url, value: { state: "error", error: e.message } }));
    return () => {
      live = false;
    };
  }, [url]);
  if (!url) return { state: "idle" };
  return v.url === url ? v.value : { state: "loading" };
}

/** The two addresses My Coast reads; a field pack (src/nx/offline/pack.ts) saves exactly these. */
export const gatesUrl = (place: string, fishery: string) => `/coast/data?${new URLSearchParams({ place, fishery })}`;
export const seaUrl = (place: string) => `/coast/data?${new URLSearchParams({ place, part: "sea" })}`;
export const useGates = (place: string | null, fishery: FisheryId | null) => useJson<CoastGates>(place && fishery ? gatesUrl(place, fishery) : null);
export const useSea = (place: string | null) => useJson<CoastSea>(place ? seaUrl(place) : null);

// ---------------------------------------------------------------- the fishery a trip is planned for

/** My Coast's own choice on this device; it never rewrites the fisheries Today follows. */
export const COAST_KEY = "nx-coast-v1";
const listeners = new Set<() => void>();

function readChoice(): string {
  try {
    const raw = window.localStorage.getItem(COAST_KEY);
    const v = raw ? (JSON.parse(raw) as { fishery?: unknown }) : null;
    if (v && typeof v.fishery === "string" && tripFishery(v.fishery)) return v.fishery;
  } catch {
    /* storage blocked: fall through to the followed fisheries */
  }
  return defaultFishery(readPortRecord().fisheries) ?? "";
}

function subscribe(l: () => void) {
  listeners.add(l);
  const onStorage = (e: StorageEvent) => (e.key === COAST_KEY || e.key === null) && l();
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(l);
    window.removeEventListener("storage", onStorage);
  };
}

/** The chosen fishery: null until the reader chooses or follows one; undefined before hydration. */
export function useFishery(): [FisheryId | null | undefined, (f: FisheryId) => void] {
  const v = useSyncExternalStore(subscribe, readChoice, () => "\u0000");
  const set = (f: FisheryId) => {
    // kept for this page view even where storage is blocked
    memo = f;
    try {
      window.localStorage.setItem(COAST_KEY, JSON.stringify({ fishery: f }));
    } catch {
      /* private window: this page view only */
    }
    listeners.forEach((l) => l());
  };
  if (v === "\u0000") return [undefined, set];
  return [((memo ?? v) || null) as FisheryId | null, set];
}
let memo: FisheryId | null = null;

// ---------------------------------------------------------------- Brief or Trip

export type CoastView = "brief" | "trip";

/**
 * My Coast's two states on one address: the brief (gates, sea window) and the trip workspace (`?view=trip`).
 * Entering the workspace is a history entry, so Back returns to the brief; every other parameter (the port,
 * a point and date carried from Explore) stays.
 */
export function useCoastView(): [CoastView, (v: CoastView) => void] {
  const sp = useSearchParams();
  const router = useRouter();
  const path = usePathname();
  const view: CoastView = sp.get("view") === "trip" ? "trip" : "brief";
  const set = useCallback(
    (v: CoastView) => {
      const q = new URLSearchParams(sp.toString());
      if (v === "trip") q.set("view", "trip");
      else q.delete("view");
      const s = q.toString();
      router.push(`${path}${s ? `?${s}` : ""}`, { scroll: false });
    },
    [sp, router, path],
  );
  return [view, set];
}
