"use client";

import { staticRefs } from "@/v4/offline/kept";
import { readSaved, writeSaved } from "./prefs";

/**
 * Save Today for offline use, with the field-pack mechanism already in the service worker
 * (public/sw.js). A cache named "neritic-pack-today" holds this port's page, the files it loads and
 * its /api/conditions answer. The worker answers from any "neritic-pack-*" cache when the network
 * fails, and marks a saved page (self.__NERITIC_SAVED__) so the page can say it is a saved copy and
 * when it was saved. Saving again replaces the copy. Nothing is invented: the copy keeps its own time.
 */
export const TODAY_CACHE = "neritic-pack-today";
/** The same worker URL as the field packs (src/v4/offline/useOfflinePack.ts), without importing the pack code. */
const SW_URL = `/sw.js${process.env.NEXT_PUBLIC_NERITIC_BUILD ? `?b=${process.env.NEXT_PUBLIC_NERITIC_BUILD}` : ""}`;

export async function saveTodayOffline(place: string, label: string, conditionsUrl: string): Promise<{ ok: true; at: string } | { ok: false; why: string }> {
  if (typeof window === "undefined" || !("caches" in window)) return { ok: false, why: "this browser cannot keep pages offline." };
  if (!navigator.onLine) return { ok: false, why: "you are offline now; save while you have a connection." };
  try {
    if ("serviceWorker" in navigator) {
      await navigator.serviceWorker.register(SW_URL, { scope: "/" });
      await navigator.serviceWorker.ready;
    }
    const at = new Date().toISOString();
    const c = await caches.open(TODAY_CACHE);
    const pageUrl = `/today?place=${encodeURIComponent(place)}`;
    const page = await fetch(pageUrl, { credentials: "same-origin", cache: "no-store" });
    if (!page.ok) return { ok: false, why: `the page did not load (HTTP ${page.status}).` };
    const html = await page.text();
    await c.put(pageUrl, new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8", "x-neritic-saved-at": at, "x-neritic-saved-url": pageUrl } }));
    const cond = await fetch(conditionsUrl, { cache: "no-store" }).catch(() => null);
    if (cond?.ok) await c.put(conditionsUrl, cond);
    const files = staticRefs(html);
    const css: string[] = [];
    await Promise.all(
      files.map(async (f) => {
        const r = await fetch(f).catch(() => null);
        if (!r?.ok) return;
        if (f.endsWith(".css")) css.push(await r.clone().text());
        await c.put(f, r).catch(() => undefined);
      }),
    );
    const fonts = css.flatMap(staticRefs).filter((f) => !files.includes(f));
    await Promise.all(
      fonts.map(async (f) => {
        const r = await fetch(f).catch(() => null);
        if (r?.ok) await c.put(f, r).catch(() => undefined);
      }),
    );
    writeSaved([{ place, label, at }, ...readSaved().filter((x) => x.place !== place)].slice(0, 6));
    return { ok: true, at };
  } catch (e) {
    return { ok: false, why: e instanceof Error && /quota/i.test(e.message) ? "this device is out of storage." : "the browser refused to store it." };
  }
}

/** The mark the service worker puts on a page it answered from a saved copy, or null. */
export function savedMark(): { at: string | null; url: string | null } | null {
  if (typeof window === "undefined") return null;
  const m = (window as unknown as { __NERITIC_SAVED__?: { at: string | null; url: string | null } }).__NERITIC_SAVED__;
  return m ?? null;
}
