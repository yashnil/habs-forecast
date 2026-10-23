"use client";

/**
 * A kept briefing, readable offline (v10 pass 2). The copy itself lives in this device's storage
 * (lib/keptBriefing); what it needs to open without a connection is the /coast/kept page and its code.
 * Keeping a copy stores that page and every file it loads in one cache ("neritic-kept-v1"), which the
 * service worker (public/sw.js) answers from when the network is gone, the same way a field pack's
 * pages are answered. It also records which copy was kept last, so the offline page can name it.
 * Small (one page and its code, no data or map); a failure leaves the copy in storage, readable online.
 */
export const KEPT_CACHE = "neritic-kept-v1";
export const KEPT_PAGE = "/coast/kept";
export const KEPT_META = "/__kept__/meta";

/** Same-origin build files a page names (scripts, styles, fonts), from its HTML or CSS. */
export function staticRefs(text: string): string[] {
  const out = new Set<string>();
  for (const m of text.matchAll(/\/_next\/static\/[A-Za-z0-9_\-./%~]+?\.(?:js|css|woff2?|ttf|otf)/g)) out.add(m[0]);
  // chunk names the RSC payload carries without the leading /_next/
  for (const m of text.matchAll(/(?<![\w/])static\/(?:chunks|css|media)\/[A-Za-z0-9_\-./%~]+?\.(?:js|css|woff2?)/g)) out.add(`/_next/${m[0]}`);
  return [...out];
}

/** Store /coast/kept and the files it loads for offline use; true when the page itself was stored. */
export async function cacheKeptPage(meta: { title: string; keptAt: string; id: string }): Promise<boolean> {
  if (typeof window === "undefined" || !("caches" in window) || !navigator.onLine) return false;
  try {
    const c = await caches.open(KEPT_CACHE);
    const page = await fetch(KEPT_PAGE, { credentials: "same-origin", cache: "no-store" });
    if (!page.ok) return false;
    const html = await page.clone().text();
    await c.put(KEPT_PAGE, page);
    const files = staticRefs(html);
    const css: string[] = [];
    await Promise.all(
      files.map(async (f) => {
        if (await c.match(f)) return;
        const r = await fetch(f).catch(() => null);
        if (!r?.ok) return;
        if (f.endsWith(".css")) css.push(await r.clone().text());
        await c.put(f, r).catch(() => undefined);
      }),
    );
    // the fonts a stylesheet names (self-hosted; no third-party request)
    const fonts = css.flatMap(staticRefs).filter((f) => !files.includes(f));
    await Promise.all(
      fonts.map(async (f) => {
        if (await c.match(f)) return;
        const r = await fetch(f).catch(() => null);
        if (r?.ok) await c.put(f, r).catch(() => undefined);
      }),
    );
    await c.put(KEPT_META, new Response(JSON.stringify(meta), { headers: { "Content-Type": "application/json" } }));
    return true;
  } catch {
    return false; // storage full or blocked: the copy stays in storage, readable online
  }
}
