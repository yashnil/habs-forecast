/*
 * Neritic service worker: offline field packs (v4).
 *
 * What it does, and nothing more:
 * - /_next/static/* (hashed, immutable): cache-first, kept in "neritic-static-v1".
 * - Anything a downloaded field pack holds ("neritic-pack-*" caches): served from the pack when the
 *   network fails. Pages, data and API answers are network-first, so online the reader always gets
 *   current information; the pack is only the fallback. When the browser says it is offline, the
 *   pack is asked first (no request that cannot succeed is made); the network is still tried for
 *   anything the pack lacks.
 * - The pack's PMTiles basemap: answered from the pack with byte ranges (206), online or not.
 * - A page with no saved copy, offline: the saved Explore or briefing page for that path, else a
 *   short offline page. Nothing is invented: every saved answer keeps its original time.
 *
 * Packs are never deleted here: a new worker version keeps them (they are removed only from the
 * page, by the reader). Browsers may still evict storage; the page says so.
 */
const VERSION = "neritic-sw-1";
const STATIC = "neritic-static-v1";
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

/** A byte range of a saved file (PMTiles reads its archive by ranges). */
async function ranged(request, resp) {
  const range = request.headers.get("range");
  if (!range) return resp;
  const m = /bytes=(\d+)-(\d*)/.exec(range);
  if (!m) return resp;
  const buf = await resp.arrayBuffer();
  const start = Number(m[1]);
  const end = m[2] ? Math.min(Number(m[2]), buf.byteLength - 1) : buf.byteLength - 1;
  return new Response(buf.slice(start, end + 1), {
    status: 206,
    headers: { "Content-Type": resp.headers.get("Content-Type") || "application/octet-stream", "Content-Range": `bytes ${start}-${end}/${buf.byteLength}`, "Content-Length": String(end - start + 1), "Accept-Ranges": "bytes" },
  });
}

async function offlinePage(path) {
  // the pack's own page for this path, if it saved one
  const saved = await packMatch(new Request(path === "/coast" ? "/__pack__/page/coast" : path === "/research" ? "/__pack__/page/research" : "/__pack__/page/explore"));
  if (saved) return saved;
  return new Response(
    `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline · Neritic</title><body style="font:17px/1.5 system-ui;margin:0;padding:24px;background:#061016;color:#ebf2f6"><h1 style="font-size:24px">You are offline</h1><p>This page was not saved for offline use. Download a field pack (Explore or My Coast) while you have a connection.</p><p>Official notices are always on the agencies' own pages: CDFW and CDPH.</p></body>`,
    { headers: { "Content-Type": "text/html; charset=utf-8" } },
  );
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
          const saved = (await packMatch(req)) || (await packMatch(req, { ignoreSearch: true }));
          if (saved) return saved;
        }
        try {
          return await fetch(req);
        } catch {
          return (await packMatch(req)) || (await packMatch(req, { ignoreSearch: true })) || offlinePage(url.pathname);
        }
      })(),
    );
    return;
  }

  // everything else: network first; a pack's saved copy when the network fails (first, when offline)
  event.respondWith(
    (async () => {
      if (offline) {
        const saved = await packMatch(req);
        if (saved) return saved;
      }
      try {
        return await fetch(req);
      } catch (err) {
        const hit = await packMatch(req);
        if (hit) return hit;
        throw err;
      }
    })(),
  );
});

self.addEventListener("message", (e) => {
  if (e.data === "version") e.source && e.source.postMessage({ version: VERSION });
});
