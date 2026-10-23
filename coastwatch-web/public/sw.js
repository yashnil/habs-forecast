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
 * - A kept briefing (My Coast's "Keep a dated copy"): the page keeps /coast/kept and the files it loads
 *   in "neritic-kept-v1" (src/v4/offline/kept.ts). Offline, /coast/kept is answered from it (the copy
 *   itself is in the device's storage), its code from it, and the offline page links to it by name.
 *
 * Packs are never deleted here: a new worker version keeps them (they are removed only from the
 * page, by the reader). Browsers may still evict storage; the page says so.
 */
const VERSION = "neritic-sw-3";
/** How long a page or an /api answer waits for the network before a field pack answers instead. */
const DEADLINE_MS = 5000;
// one code cache per build: hashed files of an old build are never needed by the new one
const BUILD = new URL(self.location.href).searchParams.get("b");
const STATIC = `neritic-static-${BUILD ? BUILD.replace(/[^\w.-]/g, "").slice(0, 40) : "v1"}`;
const PACK_PREFIX = "neritic-pack-";
const KEPT = "neritic-kept-v1";
const KEPT_PAGE = "/coast/kept";

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

/** A file the kept briefing's page needs (its code, styles and fonts), from the kept cache. */
async function keptMatch(request) {
  if (!(await caches.has(KEPT))) return null;
  return (await caches.open(KEPT)).match(request, { ignoreSearch: true });
}

/** The kept briefing's page (/coast/kept, any ?id=), or null when no copy was kept on this device. */
async function keptPage(path) {
  if (path !== KEPT_PAGE || !(await caches.has(KEPT))) return null;
  return (await caches.open(KEPT)).match(KEPT_PAGE);
}

/** What the offline page says about the kept copy: its title and when it was kept, or null. */
async function keptMeta() {
  if (!(await caches.has(KEPT))) return null;
  const c = await caches.open(KEPT);
  if (!(await c.match(KEPT_PAGE))) return null;
  try {
    const r = await c.match("/__kept__/meta");
    return r ? await r.json() : {};
  } catch {
    return {};
  }
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

const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const MARK = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="30" height="30" aria-hidden="true"><clipPath id="nxmmicrolight0"><path d="M11 0H37A11 11 0 0 1 48 11V37A11 11 0 0 1 37 48H11A11 11 0 0 1 0 37V11A11 11 0 0 1 11 0Z"/></clipPath><g clip-path="url(#nxmmicrolight0)"><path d="M11 0H37A11 11 0 0 1 48 11V37A11 11 0 0 1 37 48H11A11 11 0 0 1 0 37V11A11 11 0 0 1 11 0Z" fill="#0e3352"/><g fill="none" stroke-linecap="round"><path d="M-1.95 41.92C-1.28 40.55 -0.82 40.14 0.05 37.8C0.92 35.46 2.31 30.6 3.25 27.89C4.19 25.18 4.81 23.63 5.69 21.53C6.57 19.44 7.65 17.07 8.52 15.34C9.4 13.62 10.09 12.42 10.94 11.2C11.79 9.97 12.64 8.92 13.63 8C14.61 7.09 15.71 6.28 16.84 5.69C17.96 5.1 19.19 4.68 20.37 4.46C21.55 4.25 22.96 4.32 23.9 4.4C24.83 4.48 25.32 4.69 25.99 4.94C26.66 5.19 27.29 5.5 27.92 5.9C28.55 6.3 29.21 6.82 29.78 7.36C30.35 7.9 30.76 8.31 31.36 9.14C31.96 9.97 32.3 10.35 33.4 12.33C34.49 14.32 36.25 18.11 37.91 21.04C39.57 23.98 42.11 27.79 43.33 29.93C44.56 32.07 44.71 32.53 45.28 33.88C45.86 35.22 46.15 35.46 46.77 38.01C47.4 40.56 48.27 45.46 49.02 49.19" stroke="#a9d3df" stroke-width="3.6"/><path d="M6.72 49.4C7.33 47.63 7.72 46.88 8.55 44.1C9.38 41.32 10.79 35.54 11.69 32.72C12.6 29.9 13.29 28.62 13.99 27.19C14.69 25.77 15.25 25.03 15.9 24.16C16.55 23.29 17.21 22.56 17.9 21.97C18.58 21.38 19.28 20.91 20 20.6C20.71 20.3 21.37 20.1 22.18 20.15C23 20.19 23.98 20.4 24.89 20.89C25.8 21.38 26.58 21.96 27.64 23.08C28.7 24.2 30.16 26.02 31.25 27.6C32.34 29.19 33.45 31.14 34.18 32.59C34.9 34.05 35.23 35.08 35.58 36.32C35.94 37.56 36.16 38.79 36.31 40.04C36.46 41.29 36.51 42.23 36.49 43.82C36.47 45.42 36.29 47.69 36.19 49.62" stroke="#5fd0e6" stroke-width="5.4"/></g></g></svg>';

/** The offline page, in the site's own dark system: what is not saved, the kept briefing and the saved
 *  briefing when there is one, a way to try again, and the agencies (pages and tap-to-call lines). */
async function offlinePage(path) {
  const saved = await savedPageFor(path);
  if (saved) return saved;
  const kept = await keptMeta();
  const brief = await savedPageFor("/coast");
  const when = kept && kept.keptAt ? new Date(kept.keptAt).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles", timeZoneName: "short" }) : null;
  const opts = [
    kept ? `<a class="btn" href="/coast/kept" data-testid="offline-kept">Open your kept briefing${kept.title ? `<small>${esc(kept.title)}${when ? `, kept ${esc(when)}` : ""}</small>` : ""}</a>` : "",
    brief ? `<a class="btn" href="/coast">Open the saved briefing<small>from your field pack</small></a>` : "",
  ].join("");
  return new Response(
    `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline · Neritic</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#05090d;color:#e6eef2;font:16px/1.5 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:620px;margin:0 auto;padding:20px 16px 40px}.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em;font-size:18px;margin-bottom:36px}
.caps{font:500 11px/1.3 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#a9bac5}
h1{font:500 36px/1.08 Newsreader,Georgia,"Times New Roman",serif;letter-spacing:-.015em;margin:6px 0 10px}p{margin:0 0 12px;color:#a9bac5}
.btn{display:flex;flex-direction:column;justify-content:center;min-height:52px;padding:10px 14px;margin:0 0 10px;border:1px solid rgba(190,215,230,.32);border-radius:4px;color:#e6eef2;text-decoration:none;font-weight:500}
.btn small{font-weight:400;color:#a9bac5;font-size:13.5px}.btn:focus-visible,a:focus-visible{outline:2px solid #74c7ff;outline-offset:2px}
.agency{margin-top:28px;border-left:2px solid #f0a94a;padding-left:14px}.agency .caps{color:#f0a94a}.agency p{color:#e6eef2}
.agency a{color:#e6eef2;text-decoration:underline;text-decoration-color:rgba(240,169,74,.5);text-underline-offset:4px;display:inline-flex;align-items:center;min-height:44px;margin-right:14px}
.tel{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:300}
</style></head><body><main data-testid="offline-page"><div class="brand">${MARK}<span>Neritic</span></div>
<p class="caps">Offline</p><h1>You are offline</h1>
<p>This page was not saved for offline use. A kept briefing (My Coast, &ldquo;Keep a dated copy&rdquo;) and a field pack open without a connection; download a pack while you have one.</p>
${opts}<a class="btn" href="" style="border-style:dashed">Try again</a>
<section class="agency"><p class="caps">Official notices · CDFW and CDPH</p>
<p>No notice listed does not mean open or safe. The agencies are the authority; call before you go.</p>
<p><a class="tel" href="tel:+18316492883">CDFW closure line (831) 649-2883</a><a class="tel" href="tel:+18005534133">CDPH biotoxin line 1 (800) 553-4133</a></p></section>
</main></body></html>`,
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
  const hit = (await packMatch(req)) || (url.origin === self.location.origin && url.pathname.startsWith("/_next/") ? await keptMatch(req) : null);
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
        const hit = (await c.match(req)) || (await packMatch(req)) || (await keptMatch(req));
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
          const saved = (await packPage(req)) || (await keptPage(url.pathname));
          if (saved) return saved;
          // offline and the pack never held this page: say so at once (no request that cannot succeed)
          if (!PACK_PAGE[url.pathname] && (await hasPack())) return offlinePage(url.pathname);
        }
        try {
          // a slow network gives way to the pack after the deadline: the saved copy of this page, else
          // the short offline page (with a way to try again); with no pack at all, the network is waited for
          return await withDeadline(req, async () => (await packPage(req)) || offlinePage(url.pathname), (await hasPack()) ? DEADLINE_MS : 0);
        } catch {
          return (await packPage(req)) || (await keptPage(url.pathname)) || offlinePage(url.pathname);
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
