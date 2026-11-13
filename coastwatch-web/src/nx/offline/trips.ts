"use client";

import { staticRefs } from "@/v4/offline/kept";
import { CHART_FILES, publicRefs } from "./pack";

/**
 * Your trips, readable offline (v15). The trips themselves live in this device's storage (localStorage
 * "nx-trips-v1:", src/nx/trips/store.ts); what they need to open with no connection is the /coast/trips page and
 * its code. Each time the page is opened online it keeps a fresh copy of that page and every file it loads in one
 * cache, "neritic-trips-v1", the way a kept briefing is kept (src/v4/offline/kept.ts). The service worker
 * (public/sw.js) answers /coast/trips from it (any ?id=, the history, a workspace and its briefing) when the
 * network is gone, marked as a saved copy with the time it was kept: never shown as live.
 *
 * Small: one page, its code and fonts, and the baked chart pictures the static tier draws offline (no data, no
 * tiles). A failure (storage full, blocked) leaves everything else working online.
 */
export const TRIPS_CACHE = "neritic-trips-v1";
export const TRIPS_PAGE = "/coast/trips";
export const TRIPS_META = "/__trips__/meta";
const SW_URL = `/sw.js${process.env.NEXT_PUBLIC_NERITIC_BUILD ? `?b=${process.env.NEXT_PUBLIC_NERITIC_BUILD}` : ""}`;

export type TripsKept = { keptAt: string; files: number };

/** The posters only: the static tier's chart pictures (the live chart's geometry is not needed offline). */
const POSTERS = CHART_FILES.filter((f) => f.endsWith(".webp"));

type Deps = { fetch: typeof fetch; caches: CacheStorage; now: () => Date; register: (() => Promise<unknown>) | null };

const browserDeps = (): Deps | null => {
  if (typeof window === "undefined" || !("caches" in window)) return null;
  return {
    fetch: window.fetch.bind(window),
    caches: window.caches,
    now: () => new Date(),
    register:
      "serviceWorker" in navigator
        ? async () => {
            await navigator.serviceWorker.register(SW_URL, { scope: "/" });
            await navigator.serviceWorker.ready;
          }
        : null,
  };
};

/**
 * Keep /coast/trips and the files it loads. Old files the new page no longer names are dropped, so the cache holds
 * one build's code. Returns what was kept, or null (offline, no Cache Storage, or the page did not load).
 */
export async function cacheTripsPage(deps: Deps | null = browserDeps()): Promise<TripsKept | null> {
  if (!deps) return null;
  try {
    if (deps.register) await deps.register().catch(() => undefined);
    const page = await deps.fetch(TRIPS_PAGE, { credentials: "same-origin", cache: "no-store" });
    if (!page.ok || !(page.headers.get("Content-Type") ?? "").includes("text/html")) return null;
    const html = await page.text();
    const at = deps.now().toISOString();
    const c = await deps.caches.open(TRIPS_CACHE);
    const files = new Set([...staticRefs(html), ...publicRefs(html), ...POSTERS]);
    const css: string[] = [];
    const fetchInto = async (f: string) => {
      if (await c.match(f)) {
        if (f.endsWith(".css")) css.push(await (await c.match(f))!.text());
        return;
      }
      const r = await deps.fetch(f).catch(() => null);
      if (!r?.ok) return;
      if (f.endsWith(".css")) css.push(await r.clone().text());
      await c.put(f, r).catch(() => undefined);
    };
    await Promise.all([...files].map(fetchInto));
    // the fonts a stylesheet names (self-hosted; no third-party request)
    const fonts = css.flatMap(staticRefs).filter((f) => !files.has(f));
    for (const f of fonts) files.add(f);
    await Promise.all(fonts.map(fetchInto));
    // the page last, so a copy is only ever answered with its code beside it
    await c.put(
      TRIPS_PAGE,
      new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8", "x-neritic-saved-at": at, "x-neritic-saved-url": TRIPS_PAGE, "x-neritic-saved-label": "Your%20trips" } }),
    );
    const kept: TripsKept = { keptAt: at, files: files.size };
    await c.put(TRIPS_META, new Response(JSON.stringify(kept), { headers: { "Content-Type": "application/json" } }));
    // one build's code: what the new page does not name goes
    for (const req of await c.keys()) {
      const path = new URL(req.url, "http://x").pathname;
      if (path !== TRIPS_PAGE && path !== TRIPS_META && !files.has(path)) await c.delete(req);
    }
    return kept;
  } catch {
    return null; // storage full or blocked: the trips stay in storage, readable online
  }
}

let once: Promise<TripsKept | null> | null = null;

/**
 * Keep the trips page for offline use, once per page view and only online, after the page has drawn (so the copy
 * never competes with the first paint). Safe to call from every trips view.
 */
export function keepTripsOffline(): Promise<TripsKept | null> {
  if (typeof window === "undefined" || navigator.onLine === false) return Promise.resolve(null);
  once ??= new Promise<TripsKept | null>((resolve) => {
    const go = () => void cacheTripsPage().then(resolve, () => resolve(null));
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
    if (idle) idle(go, { timeout: 4000 });
    else window.setTimeout(go, 1500);
  }).then((r) => {
    if (!r) once = null; // a failed keep may be tried again
    return r;
  });
  return once;
}

/** When the trips page was last kept on this device, or null. */
export async function tripsKept(cs: CacheStorage | null = typeof caches === "undefined" ? null : caches): Promise<TripsKept | null> {
  if (!cs) return null;
  try {
    if (!(await cs.has(TRIPS_CACHE))) return null;
    const r = await (await cs.open(TRIPS_CACHE)).match(TRIPS_META);
    return r ? ((await r.json()) as TripsKept) : null;
  } catch {
    return null;
  }
}

/** Remove the kept page (the trips in storage are not touched). */
export async function forgetTripsPage(cs: CacheStorage | null = typeof caches === "undefined" ? null : caches): Promise<void> {
  try {
    await cs?.delete(TRIPS_CACHE);
  } catch {
    // nothing to remove
  }
  once = null;
}
