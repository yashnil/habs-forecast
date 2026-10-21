/*
 * Neritic service worker: offline field packs (v4).
 *
 * What it does, and nothing more:
 * - /_next/static/* (hashed, immutable): cache-first, kept in "neritic-static-<build>" (the build named
 *   in this worker's own URL, /sw.js?b=<build>): a new build's worker drops the old build's code cache.
 * - Anything a downloaded field pack holds ("neritic-pack-*" caches): served from the pack when the
 *   network fails. Pages, data and API answers are network-first, so online the reader always gets
 *   current information; the pack is only the fallback. When the browser says it is offline, the
 *   pack is asked first (no request that cannot succeed is made); the network is still tried for
 *   anything the pack lacks.
 * - The pack's PMTiles basemap: answered from the pack with byte ranges (206), online or not.
 * - A slow network (a bar of signal, a harbour's wifi): with a field pack on this device, a page or an
 *   /api answer waits at most 5 s for the network, then the pack's copy is shown (the network answer is
 *   still taken if it arrives later, for the next visit). With no pack saved, the network is waited for.
 * - A page with no saved copy, offline: the saved Explore or briefing page for that path (only those two
 *   paths), else a short offline page with a way to try again. A page the pack never held (Research,
 *   Fisheries, About) is never answered with another page.
 * - /api/briefing for another trip of a saved place: the pack's briefing (saved for all species),
 *   marked so (x-neritic-saved-for), never relabelled as the reader's own trip. Nothing is invented: every saved answer keeps its original time. A page answered
 *   from a pack carries a mark (self.__NERITIC_SAVED__: when it was saved, and the address it was saved
 *   for), so the page can say it is a saved copy and hydrate the HTML it was actually given.
 *
 * Packs are never deleted here: a new worker version keeps them (they are removed only from the
 * page, by the reader). Browsers may still evict storage; the page says so.
 */
const VERSION = "neritic-sw-2";
/** How long a page or an /api answer waits for the network before a field pack answers instead. */
const DEADLINE_MS = 5000;
// one code cache per build: hashed files of an old build are never needed by the new one
const BUILD = new URL(self.location.href).searchParams.get("b");
const STATIC = `neritic-static-${BUILD ? BUILD.replace(/[^\w.-]/g, "").slice(0, 40) : "v1"}`;
const PACK_PREFIX = "neritic-pack-";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => {
  e.waitUntil(
    (async () => {
      const keys = await caches.keys();
      // only old static caches go; field packs stay across worker updates
      await Promise.all(keys.filter((k) => k.startsWith("neritic-static-") && k !== STATIC).map((k) => caches.delete(k)));
      await self.clients.claim();
    })(),
  );
});

async function packMatch(request, opts) {
  const keys = (await caches.keys()).filter((k) => k.startsWith(PACK_PREFIX));
  for (const k of keys) {
    const c = await caches.open(k);
    const r = await c.match(request, opts);
    if (r) return r;
  }
  return null;
}

/** A saved page, and the address it was saved for: the exact address first, then the same path with
 *  other search params (one saved briefing answers every trip and place offline). */
async function packPage(request) {
  const keys = (await caches.keys()).filter((k) => k.startsWith(PACK_PREFIX));
  for (const ignoreSearch of [false, true]) {
    for (const k of keys) {
      const c = await caches.open(k);
      const [key] = await c.keys(request, { ignoreSearch });
      if (!key) continue;
      const r = await c.match(key);
      if (r) return marked(r, key.url);
    }
  }
  return null;
}

/** A saved HTML page with its mark: when it was saved, and the address it was saved for. */
async function marked(resp, keyUrl) {
  if (!(resp.headers.get("Content-Type") || "").includes("text/html")) return resp;
  const at = resp.headers.get("x-neritic-saved-at");
  let url = resp.headers.get("x-neritic-saved-url");
  if (!url && keyUrl && !new URL(keyUrl).pathname.startsWith("/__pack__/")) {
    const u = new URL(keyUrl);
    url = u.pathname + u.search;
  }
  const mark = `<script>self.__NERITIC_SAVED__=${JSON.stringify({ at, url }).replace(/</g, "\\u003c")}</script>`;
  const html = await resp.text();
  // after the opening <head …> tag, or else after the doctype: never before it (quirks mode)
  const m = /<head[^>]*>/i.exec(html) || /<!doctype[^>]*>/i.exec(html);
  const i = m ? m.index + m[0].length : 0;
  const headers = new Headers(resp.headers);
  headers.delete("content-length");
  return new Response(html.slice(0, i) + mark + html.slice(i), { status: 200, headers });
}

/** A byte range of a saved file (PMTiles reads its archive by ranges): sliced from the stored blob,
 *  so a read of a few kilobytes never copies the whole archive into memory. */
async function ranged(request, resp) {
  const range = request.headers.get("range");
  if (!range) return resp;
  const m = /bytes=(\d+)-(\d*)/.exec(range);
  if (!m) return resp;
  const blob = await resp.blob();
  const start = Number(m[1]);
  const end = m[2] ? Math.min(Number(m[2]), blob.size - 1) : blob.size - 1;
  return new Response(blob.slice(start, end + 1), {
    status: 206,
    headers: { "Content-Type": resp.headers.get("Content-Type") || "application/octet-stream", "Content-Range": `bytes ${start}-${end}/${blob.size}`, "Content-Length": String(end - start + 1), "Accept-Ranges": "bytes" },
  });
}

/** The pack's own page for the paths it saves (Explore and the briefing), else null. */
const PACK_PAGE = { "/": "/__pack__/page/explore", "/coast": "/__pack__/page/coast" };
async function savedPageFor(path) {
  const key = PACK_PAGE[path];
  return key ? packPage(new Request(key)) : null;
}

async function offlinePage(path) {
  const saved = await savedPageFor(path);
  if (saved) return saved;
  const a = 'style="color:#5fd0e6;display:inline-block;min-height:44px;line-height:44px"';
  const brief = (await savedPageFor("/coast")) ? `<a href="/coast" ${a}>Open the saved briefing</a> · ` : "";
  return new Response(
    `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline · Neritic</title><body style="font:17px/1.5 system-ui;margin:0;padding:24px;background:#061016;color:#ebf2f6"><h1 style="font-size:24px">You are offline</h1><p>This page was not saved for offline use. A field pack saves Explore and the briefing (My Coast) for its region; download one while you have a connection.</p><p>${brief}<a href="" ${a}>Try again</a></p><p>Official notices are always on the agencies' own pages: CDFW and CDPH.</p></body>`,
    { headers: { "Content-Type": "text/html; charset=utf-8", "x-neritic-offline": "1" } },
  );
}

/** Whether any field pack is on this device (only then does a slow network give way to it). */
async function hasPack() {
  return (await caches.keys()).some((k) => k.startsWith(PACK_PREFIX));
}

/**
 * The network, given until the deadline; then `fallback()` answers if it can (a pack copy). A late
 * network answer is not lost: the browser still completes it, and the next visit gets it. With no
 * fallback the network is waited for (and its error, if any, goes to the caller).
 */
async function withDeadline(req, fallback, ms) {
  const net = fetch(req);
  if (!ms) return net;
  let timer;
  const late = new Promise((resolve) => (timer = setTimeout(() => resolve("late"), ms)));
  const first = await Promise.race([net.then((r) => ({ r }), (e) => ({ e })), late]);
  clearTimeout(timer);
  if (first !== "late") {
    if (first.e) throw first.e;
    return first.r;
  }
  const saved = await fallback();
  if (saved) return saved;
  return net;
}

/** /api/briefing for another trip (for, species) of a place the pack saved: the pack's own briefing,
 *  which is for all species, marked so the page can say it. */
async function briefingAnyTrip(url) {
  const place = url.searchParams.get("place");
  if (!place) return null;
  const keys = (await caches.keys()).filter((k) => k.startsWith(PACK_PREFIX));
  for (const k of keys) {
    const c = await caches.open(k);
    for (const key of await c.keys(new Request("/api/briefing"), { ignoreSearch: true })) {
      if (new URL(key.url).searchParams.get("place") !== place) continue;
      const r = await c.match(key);
      if (!r) continue;
      const headers = new Headers(r.headers);
      headers.set("x-neritic-saved-for", "all-species");
      return new Response(r.body, { status: r.status, headers });
    }
  }
  return null;
}

/** The pack's copy of a non-page request (exact address; a briefing for another trip of the same place). */
async function packAnswer(req, url) {
  const hit = await packMatch(req);
  if (hit) return hit;
  return url.pathname === "/api/briefing" ? briefingAnyTrip(url) : null;
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  if (url.origin === self.location.origin && url.pathname.startsWith("/_next/static/")) {
    event.respondWith(
      (async () => {
        const c = await caches.open(STATIC);
        const hit = (await c.match(req)) || (await packMatch(req));
        if (hit) return hit;
        const r = await fetch(req);
        if (r.ok) c.put(req, r.clone()).catch(() => undefined); // storage full: served, not kept
        return r;
      })(),
    );
    return;
  }

  if (url.origin === self.location.origin && url.pathname.startsWith("/offline/") && url.pathname.endsWith(".pmtiles")) {
    event.respondWith(
      (async () => {
        const hit = await packMatch(new Request(url.pathname));
        if (hit) return ranged(req, hit);
        return fetch(req);
      })(),
    );
    return;
  }

  // offline (as the browser reports it): the saved copy first, so a reopen does not log a failed request per file
  const offline = self.navigator && self.navigator.onLine === false;

  if (req.mode === "navigate") {
    event.respondWith(
      (async () => {
        if (offline) {
          const saved = await packPage(req);
          if (saved) return saved;
          // offline and the pack never held this page: say so at once (no request that cannot succeed)
          if (!PACK_PAGE[url.pathname] && (await hasPack())) return offlinePage(url.pathname);
        }
        try {
          // a slow network gives way to the pack after the deadline: the saved copy of this page, else
          // the short offline page (with a way to try again); with no pack at all, the network is waited for
          return await withDeadline(req, async () => (await packPage(req)) || offlinePage(url.pathname), (await hasPack()) ? DEADLINE_MS : 0);
        } catch {
          return (await packPage(req)) || offlinePage(url.pathname);
        }
      })(),
    );
    return;
  }

  // everything else: network first; a pack's saved copy when the network fails (first, when offline);
  // the site's own /api answers give way to the pack after the deadline
  const api = url.origin === self.location.origin && url.pathname.startsWith("/api/");
  event.respondWith(
    (async () => {
      if (offline) {
        const saved = await packAnswer(req, url);
        if (saved) return saved;
      }
      try {
        return await withDeadline(req, () => packAnswer(req, url), api && (await hasPack()) ? DEADLINE_MS : 0);
      } catch (err) {
        const hit = await packAnswer(req, url);
        if (hit) return hit;
        throw err;
      }
    })(),
  );
});

self.addEventListener("message", (e) => {
  if (e.data === "version") e.source && e.source.postMessage({ version: VERSION });
});
