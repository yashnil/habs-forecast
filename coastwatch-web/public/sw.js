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
 * - A port's field pack (v12, Menu > Offline): "neritic-pack-field" holds Today, My Coast (its gates and sea
 *   window), Explore (each layer's latest frame around the port) and Research (a station's record) for one
 *   port, with a record of what it holds (/__field__/record.json). Its pages are answered at their addresses
 *   with or without the port in them; the offline page links to each one.
 * - Today at your port ("Save for offline"): a pack named "neritic-pack-today" holding the port's page,
 *   its code and its /api/conditions answer. It is answered like any pack, and the offline page links
 *   to each saved port by name with the time it was saved.
 * - Web push for the pilot's alerts comes from /sw-push.js (imported below; listeners only).
 *
 * - A kept briefing (My Coast's "Keep a dated copy"): the page keeps /coast/kept and the files it loads
 *   in "neritic-kept-v1" (src/v4/offline/kept.ts). Offline, /coast/kept is answered from it (the copy
 *   itself is in the device's storage), its code from it, and the offline page links to it by name.
 *
 * Packs are never deleted here: a new worker version keeps them (they are removed only from the
 * page, by the reader). Browsers may still evict storage; the page says so.
 */
const VERSION = "neritic-sw-6";
/** How long a page or an /api answer waits for the network before a field pack answers instead. */
const DEADLINE_MS = 5000;
// one code cache per build: hashed files of an old build are never needed by the new one
const BUILD = new URL(self.location.href).searchParams.get("b");
const STATIC = `neritic-static-${BUILD ? BUILD.replace(/[^\w.-]/g, "").slice(0, 40) : "v1"}`;
const PACK_PREFIX = "neritic-pack-";
const KEPT = "neritic-kept-v1";
const KEPT_PAGE = "/coast/kept";
/** Today at your port, saved from its "Save for offline" button (src/pilot/today/offline.ts). */
const TODAY_PACK = "neritic-pack-today";
/** A port's field pack (v12, Menu > Offline; src/nx/offline/pack.ts): Today, My Coast, Explore, Research. */
const FIELD_PACK = "neritic-pack-field";

// Pilot: web push ("push", "notificationclick" and a ping). It adds listeners only; it never touches
// fetch handling or the caches above (public/sw-push.js; docs/coastwatch/pilot/accounts.md).
importScripts("/sw-push.js");

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

/**
 * Which saved copy of a path answers an address it was not saved for. A page is rendered for its whole address
 * (`/coast?place=port-592&view=trip` is the trip workspace, `/coast?place=port-592` the brief), so the copy
 * that shares the most parameters with the request is the nearest thing this device holds: the trip view for a
 * trip address, the brief for anything else. Ties go to the copy saved with the fewest parameters, then to the
 * first saved. Pure (the unit test reads it): `saved` are the saved addresses of one path.
 */
function nearestSaved(requestUrl, saved) {
  const want = new URL(requestUrl).searchParams;
  let best = null;
  let bestScore = -1;
  let bestSize = Infinity;
  for (const href of saved) {
    const have = new URL(href).searchParams;
    let score = 0;
    let size = 0;
    for (const [k, v] of have) {
      size += 1;
      if (want.get(k) === v) score += 1;
    }
    if (score > bestScore || (score === bestScore && size < bestSize)) {
      best = href;
      bestScore = score;
      bestSize = size;
    }
  }
  return best;
}

/**
 * A navigation answered from a pack. The copy saved for exactly this address is served as it is. An address
 * the pack holds only under other parameters (a bare `/coast`, another port, `?view=trip` from a link that
 * also carries a fishery, a point and date carried from Explore) is SENT ON to the nearest saved address
 * (`nearestSaved`), never answered with that copy's HTML in place: the page reads its address when it draws,
 * so HTML rendered for one address and hydrated at another is a different page (React error 418, and a page
 * rebuilt on the client). After the redirect the address bar says what is shown. A v4 pack's own pages
 * (`/__pack__/…`) have no address to send to and are served as before.
 */
async function packNavigate(request) {
  const names = (await caches.keys()).filter((k) => k.startsWith(PACK_PREFIX));
  const saved = [];
  for (const k of names) {
    const c = await caches.open(k);
    const [exact] = await c.keys(request);
    if (exact) {
      const r = await c.match(exact);
      if (r) return marked(r, exact.url);
    }
    for (const key of await c.keys(request, { ignoreSearch: true })) if (!new URL(key.url).pathname.startsWith("/__pack__/")) saved.push(key.url);
  }
  const to = nearestSaved(request.url, saved);
  if (to && to !== request.url) return Response.redirect(to, 302);
  return packPage(request);
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

/**
 * The pack's own page for the paths it saves (Explore and the briefing), else null.
 * v12 (sw-5) moved Explore from `/?…` to `/explore` and made `/` the front door, which opens Today at the saved
 * port. Packs are not rewritten (their page keys never named a URL, and Today's are still `/today?place=…`);
 * the worker answers the new addresses from them instead, and an old `/?…` Explore address too:
 *   /explore, or / with Explore's parameters   the pack's Explore page
 *   / (bare), /today without a port            the newest saved Today, else the pack's Explore page
 *   /coast                                     the pack's briefing
 */
const PACK_PAGE = { "/explore": "/__pack__/page/explore", "/coast": "/__pack__/page/coast" };
const CAMPAIGN = /^(utm_[a-z_]+|fbclid|gclid|msclkid|mc_[a-z]+|ref|source|pick)$/i;
const namesExplore = (url) => [...url.searchParams.keys()].some((k) => !CAMPAIGN.test(k));
/** Pages the worker can answer from a pack for this address (see savedPageFor). */
function hasPackPage(url) {
  return !!PACK_PAGE[url.pathname] || url.pathname === "/" || (url.pathname === "/today" && !url.searchParams.get("place"));
}
async function savedPageFor(path, url) {
  const u = url || new URL(path, self.location.origin);
  if (u.pathname === "/" && namesExplore(u)) return packPage(new Request(PACK_PAGE["/explore"]));
  if (u.pathname === "/" || (u.pathname === "/today" && !u.searchParams.get("place"))) {
    const t = (await savedTodays())[0];
    if (t) {
      const c = await caches.open(TODAY_PACK);
      const r = await c.match(t.href);
      if (r) return marked(r, new URL(t.href, self.location.origin).href);
    }
    return packPage(new Request(PACK_PAGE["/explore"]));
  }
  const key = PACK_PAGE[u.pathname];
  return key ? packPage(new Request(key)) : null;
}

const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** The Today pages saved on this device: their address, the port's name and when each was saved. */
async function savedTodays() {
  if (!(await caches.has(TODAY_PACK))) return [];
  const c = await caches.open(TODAY_PACK);
  const out = [];
  for (const key of await c.keys()) {
    const u = new URL(key.url);
    if (u.pathname !== "/today") continue;
    const r = await c.match(key);
    if (!r) continue;
    let label = "";
    try {
      label = decodeURIComponent(r.headers.get("x-neritic-saved-label") || "");
    } catch {
      label = "";
    }
    out.push({ href: u.pathname + u.search, label, at: r.headers.get("x-neritic-saved-at") });
  }
  return out.sort((a, b) => String(b.at).localeCompare(String(a.at)));
}

/** The field pack's record: the port, when it was saved and which pages it holds; null without one. */
async function fieldRecord() {
  if (!(await caches.has(FIELD_PACK))) return null;
  try {
    const r = await (await caches.open(FIELD_PACK)).match("/__field__/record.json");
    return r ? await r.json() : null;
  } catch {
    return null;
  }
}

/** The offline page, in the chart's own paper and ink (v12 Day tokens): it says this page was not saved,
 *  links to what this device holds (the field pack's pages, saved Todays, a kept or v4 pack briefing), offers
 *  a way to try again, and gives the agencies (pages and tap-to-call lines). */
async function offlinePage(path, url) {
  const saved = await savedPageFor(path, url);
  if (saved) return saved;
  const kept = await keptMeta();
  const brief = await savedPageFor("/coast");
  const field = await fieldRecord();
  const stamp = (t) => new Date(t).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/Los_Angeles", timeZoneName: "short" });
  const when = kept && kept.keptAt ? stamp(kept.keptAt) : null;
  const todays = (await savedTodays()).filter((t) => !(field && field.pages && field.pages.some((p) => p.ok && p.path === t.href)));
  const fieldLinks = field && Array.isArray(field.pages) ? field.pages.filter((p) => p.ok && typeof p.path === "string" && p.path.startsWith("/")) : [];
  const opts = [
    ...fieldLinks.map((p) => `<a class="btn" href="${esc(p.path)}" data-testid="offline-field">${esc(p.label)}${field.port && field.port.name ? ` at ${esc(field.port.name)}` : ""}<small>saved ${esc(stamp(field.savedAt))}, as it was then</small></a>`),
    // the pilot's front door: the reader's own port, as it was when they saved it
    ...todays.map((t) => `<a class="btn" href="${esc(t.href)}" data-testid="offline-today">Today${t.label ? ` at ${esc(t.label)}` : ""}<small>${t.at && !Number.isNaN(Date.parse(t.at)) ? `saved ${esc(stamp(t.at))}, as it was then` : "your saved copy"}</small></a>`),
    kept ? `<a class="btn" href="/coast/kept" data-testid="offline-kept">Your kept briefing${kept.title ? `<small>${esc(kept.title)}${when ? `, kept ${esc(when)}` : ""}</small>` : ""}</a>` : "",
    brief && !fieldLinks.length ? `<a class="btn" href="/coast">The saved briefing<small>from your field pack</small></a>` : "",
  ].join("");
  return new Response(
    `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Offline · Neritic</title><style>
:root{color-scheme:light;--paper:#fffefa;--ink:#0e1417;--ink-2:#2f4a5b;--rule-soft:rgba(14,20,23,.26);--reg:#a83ba6}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--paper:#1c2226;--ink:#e2e7e8;--ink-2:#aebbc0;--rule-soft:rgba(226,231,232,.24);--reg:#cf8ccd}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 "Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:620px;margin:0 auto;padding:16px 16px 40px}.brand{font:500 22px/1 Literata,Georgia,serif;min-height:52px;display:flex;align-items:center;margin-bottom:24px}
h1{font:500 36px/1.1 Literata,Georgia,serif;letter-spacing:-.012em;margin:0 0 12px;padding-top:12px;border-top:1.5px solid var(--ink)}p{margin:0 0 12px;color:var(--ink-2)}
h2{font-size:16px;font-weight:600;margin:24px 0 8px}
.btn{display:flex;flex-direction:column;justify-content:center;min-height:52px;padding:8px 0;border-top:1px solid var(--rule-soft);color:var(--ink);text-decoration:none;font-weight:600}
.btn small{font-weight:400;color:var(--ink-2);font-size:14px}.btn:focus-visible,a:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.agency{margin-top:32px;border-left:2px solid var(--reg);padding-left:12px}.agency p{color:var(--ink)}
.agency a{color:var(--ink);display:inline-flex;align-items:center;min-height:44px;margin-right:16px;font-variant-numeric:tabular-nums}
</style></head><body><main data-testid="offline-page"><div class="brand">Neritic</div>
<h1>You are offline</h1>
<p>This page was not saved for offline use. ${opts ? "This device holds these pages, each as it was when saved:" : "Save your port while you have a connection: Menu, Offline keeps Today, My Coast, Explore&rsquo;s latest frames and Research&rsquo;s record."}</p>
${opts}<a class="btn" href="">Try again</a>
<section class="agency"><h2>Official notices: CDFW and CDPH</h2>
<p>No notice listed does not mean open. The agencies are the authority; call before you go.</p>
<p><a href="tel:+18316492883">CDFW closure line (831) 649-2883</a><a href="tel:+18005534133">CDPH biotoxin line 1 (800) 553-4133</a></p></section>
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
          const saved = (await packNavigate(req)) || (await keptPage(url.pathname));
          if (saved) return saved;
          // offline and the pack never held this page: say so at once (no request that cannot succeed)
          if (!hasPackPage(url) && (await hasPack())) return offlinePage(url.pathname, url);
        }
        try {
          // a slow network gives way to the pack after the deadline: the saved copy of this page, else
          // the short offline page (with a way to try again); with no pack at all, the network is waited for
          return await withDeadline(req, async () => (await packNavigate(req)) || offlinePage(url.pathname, url), (await hasPack()) ? DEADLINE_MS : 0);
        } catch {
          return (await packNavigate(req)) || (await keptPage(url.pathname)) || offlinePage(url.pathname, url);
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
