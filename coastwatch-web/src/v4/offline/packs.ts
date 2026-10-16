"use client";

import type { Manifest } from "@/generated/schema";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, satelliteLatest, tileIndexUrl } from "@/lib/layers";
import { currentsHourly, currentsMean } from "@/lib/currents";

/**
 * Offline field packs (v4). One bounded, licensed region per pack. A pack is a Cache Storage cache
 * ("neritic-pack-<id>") holding everything its pages need to open with no network, plus a record
 * ("/__pack__/record.json") that says what was saved, when, and from which data. The service worker
 * (public/sw.js) answers from it only when the network fails.
 */
export type PackDef = { id: string; name: string; port: number; place: string; bbox: [number, number, number, number]; lat: number; lon: number };

export const PACKS: Record<string, PackDef> = {
  monterey: { id: "monterey", name: "Monterey Bay", port: 550, place: "port-550", bbox: [-122.124, 36.44, -121.676, 36.8], lat: 36.60024, lon: -121.89469 },
};
export const packForPort = (code: number | null | undefined) => Object.values(PACKS).find((p) => p.port === code) ?? null;
export const cacheName = (id: string) => `neritic-pack-${id}`;
const RECORD = "/__pack__/record.json";

export type Group = "app" | "basemap" | "model" | "satellite" | "currents" | "briefing";
export const GROUP_WORDS: Record<Group, { name: string; offline: string }> = {
  app: { name: "Neritic's pages and code", offline: "Explore and the briefing open with no connection" },
  basemap: { name: "Map of the region", offline: "Coast, roads and names (OpenStreetMap), seafloor relief and fonts" },
  model: { name: "NOAA C-HARM run", offline: "The run's four days for all three variables: no day beyond its horizon" },
  satellite: { name: "Satellite chlorophyll, latest clear view", offline: "Sentinel-3 tiles over the region, each pixel with its own date" },
  currents: { name: "HF-radar currents", offline: "The newest hour and the 24 h mean" },
  briefing: { name: "Briefing and evidence snapshot", offline: "Notices, shore samples and harbour conditions as they were when saved" },
};

export type PlanItem = { url: string; group: Group; bytes: number | null; key?: string };
export type PackRecord = {
  id: string;
  version: number;
  savedAt: string;
  dataGeneratedAt: string;
  charmIssued: string | null;
  satelliteNewest: string | null;
  noticesCheckedAt: string | null;
  items: { url: string; group: Group; bytes: number }[];
  totalBytes: number;
  complete: boolean;
  verified: { basemapSha256: boolean; missing: number; failed: string[] } | null;
  osmDataAsOf: string | null;
};

const xyz = (lon: number, lat: number, z: number) => {
  const n = 2 ** z;
  const x = Math.floor(((lon + 180) / 360) * n);
  const r = (lat * Math.PI) / 180;
  const y = Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n);
  return [x, y];
};

/** Every resource the pack needs, from the published manifest (nothing is guessed). */
export function plan(pack: PackDef, m: Manifest, baseUrl: string, pageAssets: string[]): PlanItem[] {
  const out: PlanItem[] = [];
  const add = (url: string, group: Group, key?: string) => out.some((x) => x.url === url) || out.push({ url, group, bytes: null, key });
  // the app: its pages (saved under stable keys for the offline fallback) and the code they load
  add(`/?sel=harbor:${pack.port}&layer=forecast`, "app", "/__pack__/page/explore");
  add(`/coast?place=${pack.place}`, "app", "/__pack__/page/coast");
  for (const a of pageAssets) add(a, "app");
  add("/vendor/maplibre/maplibre-gl-worker.mjs", "app");
  // the basemap
  add(`/offline/${pack.id}/basemap.pmtiles`, "basemap");
  add(`/offline/${pack.id}/pack.json`, "basemap");
  for (const f of ["relief-land.webp", "relief-sea.webp", "relief-sea-tint.webp", "relief-sea-monterey.webp", "relief-sea-monterey-tint.webp", "relief.json", "isobaths.geojson", "water-fallback.json", "ca-counties.json"]) add(`/basemap/${f}`, "basemap");
  for (const stack of ["Noto Sans Regular", "Noto Sans Italic"]) for (const r of ["0-255", "256-511", "512-767", "768-1023", "7680-7935", "8192-8447", "8448-8703", "8704-8959"]) add(`/fonts/${encodeURIComponent(stack)}/${r}.pbf`, "basemap");
  // the C-HARM run: every published day of every variable (its own horizon, never more)
  for (const v of CHARM_VARIABLES)
    for (const l of CHARM_LEADS) {
      const lyr = charmLayer(m, v, l);
      if (lyr?.grid) add(artifactUrl(baseUrl, lyr.grid.url), "model");
    }
  // satellite: the latest clear view over the region, every native zoom
  const sat = satelliteLatest(m, "olci300");
  const t = sat?.tiles;
  if (sat && t?.relative) {
    const idx = tileIndexUrl(baseUrl, t);
    if (idx) add(idx, "satellite");
    for (let z = t.min_zoom ?? 5; z <= t.max_native_zoom; z++) {
      const [x0, y0] = xyz(pack.bbox[0], pack.bbox[3], z);
      const [x1, y1] = xyz(pack.bbox[2], pack.bbox[1], z);
      for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) add(artifactUrl(baseUrl, t.url_template.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y))), "satellite");
    }
  }
  // satellite grids (the briefing reads its chlorophyll from them): the chunks over the region, value and age
  for (const prod of ["olci300", "viirs750"] as const) {
    const l = satelliteLatest(m, prod);
    for (const g of [l?.grid, l?.composite?.age_grid]) {
      const ch = g?.chunks;
      if (!g || !ch) continue;
      const rowOf = (lat: number) => Math.floor((lat - g.lat_first) / g.lat_step / ch.rows);
      const colOf = (lon: number) => Math.floor((lon - g.lon_first) / g.lon_step / ch.cols);
      const [r0, r1] = [rowOf(pack.bbox[3]), rowOf(pack.bbox[1])].sort((a, b) => a - b);
      const [c0, c1] = [colOf(pack.bbox[0]), colOf(pack.bbox[2])].sort((a, b) => a - b);
      for (let r = r0; r <= r1; r++)
        for (let c = c0; c <= c1; c++) if (ch.present.includes(`${r}_${c}`)) add(artifactUrl(baseUrl, ch.url_template.replace("{row}", String(r)).replace("{col}", String(c))), "satellite");
    }
  }
  // currents: the newest hour and the 24 h mean
  for (const l of [currentsHourly(m).at(-1), currentsMean(m)]) {
    if (!l?.vectors) continue;
    add(artifactUrl(baseUrl, l.vectors.u_grid.url), "currents");
    add(artifactUrl(baseUrl, l.vectors.v_grid.url), "currents");
  }
  // the briefing's evidence and conditions, as they are now (dated by their own fields)
  add(`/api/evidence?place=${pack.place}`, "briefing");
  add(`/api/conditions?lat=${pack.lat.toFixed(4)}&lon=${pack.lon.toFixed(4)}`, "briefing");
  add(`/api/briefing?place=${pack.place}&for=all&species=any`, "briefing");
  return out;
}

/** The script and style files a page needs, read from its HTML (and the CSS fonts they reference). */
export async function pageAssetsOf(urls: string[]): Promise<string[]> {
  const found = new Set<string>();
  // what the current page has already loaded (includes lazily loaded map code)
  for (const e of performance.getEntriesByType("resource")) {
    const u = new URL(e.name);
    if (u.origin === location.origin && u.pathname.startsWith("/_next/static/")) found.add(u.pathname);
  }
  // open each page once, off-screen, so the code it loads on demand (the map, its workers) is known too
  for (const page of urls) {
    await new Promise<void>((resolve) => {
      const f = document.createElement("iframe");
      f.setAttribute("aria-hidden", "true");
      f.tabIndex = -1;
      f.style.cssText = "position:fixed;left:-2000px;top:0;width:1024px;height:768px;opacity:0;pointer-events:none;border:0";
      const done = () => {
        try {
          for (const e of f.contentWindow?.performance.getEntriesByType("resource") ?? []) {
            const u = new URL(e.name);
            if (u.origin === location.origin && u.pathname.startsWith("/_next/static/")) found.add(u.pathname);
          }
        } catch {
          // not readable: its HTML is still read below
        }
        f.remove();
        resolve();
      };
      f.onload = () => setTimeout(done, 6000);
      setTimeout(done, 20000);
      f.src = page;
      document.body.appendChild(f);
    });
  }
  for (const page of urls) {
    try {
      const html = await (await fetch(page, { cache: "no-store" })).text();
      for (const m of html.matchAll(/\/_next\/static\/[^"'\\\s)]+/g)) found.add(m[0]);
    } catch {
      // the page could not be read now; its assets are picked up at runtime instead
    }
  }
  for (const css of [...found].filter((x) => x.endsWith(".css"))) {
    try {
      const txt = await (await fetch(css)).text();
      for (const m of txt.matchAll(/url\((\/_next\/static\/[^)]+)\)/g)) found.add(m[1]);
    } catch {
      // fonts are cached at runtime if missed here
    }
  }
  return [...found];
}

/** Satellite tiles that were never published (cloud, land) are not requested: the tile index lists those that exist. */
export async function keepPublishedTiles(items: PlanItem[], m: Manifest, baseUrl: string): Promise<PlanItem[]> {
  const t = satelliteLatest(m, "olci300")?.tiles;
  const idx = t ? tileIndexUrl(baseUrl, t) : null;
  if (!t || !idx) return items;
  try {
    const have = new Set((await (await fetch(idx)).json()) as string[]);
    const prefix = artifactUrl(baseUrl, t.url_template.split("{z}")[0]);
    return items.filter((it) => {
      if (it.group !== "satellite" || !it.url.startsWith(prefix) || it.url === idx) return true;
      const zxy = it.url.slice(prefix.length).replace(/\.png$/, "");
      return have.has(zxy);
    });
  } catch {
    return items;
  }
}

/** Sizes from the servers (HEAD), so the reader sees the real total before downloading. */
export async function sizeOf(items: PlanItem[], signal?: AbortSignal): Promise<PlanItem[]> {
  const out = [...items];
  let i = 0;
  const worker = async () => {
    while (i < out.length) {
      const k = i++;
      try {
        const r = await fetch(out[k].url, { method: "HEAD", signal });
        const n = Number(r.headers.get("content-length"));
        out[k] = { ...out[k], bytes: Number.isFinite(n) && n > 0 ? n : null };
      } catch {
        out[k] = { ...out[k], bytes: null };
      }
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
  return out;
}

export async function readRecord(id: string): Promise<PackRecord | null> {
  if (typeof caches === "undefined") return null;
  try {
    // never caches.open() to read: it would create the cache it was asked about
    if (!(await caches.has(cacheName(id)))) return null;
    const c = await caches.open(cacheName(id));
    const r = await c.match(RECORD);
    return r ? ((await r.json()) as PackRecord) : null;
  } catch {
    return null;
  }
}

export async function deletePack(id: string) {
  await caches.delete(cacheName(id));
}

export type Progress = { done: number; total: number; bytes: number; totalBytes: number; failed: string[]; current: Group | null; rate: number | null };

async function sha256(buf: ArrayBuffer) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Download a pack: four requests at a time, real bytes counted as they arrive, already-saved items
 * skipped (a stopped download resumes), a full-storage error reported as such. The record is
 * written last; the pack is "complete" only when every item is saved and the basemap's checksum
 * matches the published one.
 */
export async function download(pack: PackDef, items: PlanItem[], meta: Omit<PackRecord, "items" | "totalBytes" | "complete" | "verified" | "savedAt" | "id" | "version" | "osmDataAsOf">, onProgress: (p: Progress) => void, signal: AbortSignal): Promise<PackRecord> {
  const c = await caches.open(cacheName(pack.id));
  const p: Progress = { done: 0, total: items.length, bytes: 0, totalBytes: items.reduce((a, x) => a + (x.bytes ?? 0), 0), failed: [], current: null, rate: null };
  const saved: { url: string; group: Group; bytes: number }[] = [];
  const t0 = performance.now();
  let i = 0;
  let quota = false;
  const one = async (it: PlanItem) => {
    const have = await c.match(it.url);
    if (have) {
      const n = Number(have.headers.get("content-length")) || it.bytes || 0;
      saved.push({ url: it.url, group: it.group, bytes: n });
      p.bytes += n;
      return;
    }
    const r = await fetch(it.url, { signal, cache: "no-store", mode: it.url.startsWith("/") ? "same-origin" : "cors" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    const headers = new Headers(r.headers);
    headers.set("content-length", String(buf.byteLength));
    headers.set("x-neritic-saved-at", new Date().toISOString());
    const resp = new Response(buf, { status: 200, headers });
    await c.put(it.url, resp.clone());
    if (it.key) await c.put(it.key, resp);
    saved.push({ url: it.url, group: it.group, bytes: buf.byteLength });
    p.bytes += buf.byteLength;
  };
  const worker = async () => {
    while (i < items.length && !signal.aborted && !quota) {
      const it = items[i++];
      p.current = it.group;
      try {
        await one(it);
      } catch (e) {
        if ((e as Error).name === "QuotaExceededError") quota = true;
        if ((e as Error).name !== "AbortError") p.failed.push(it.url);
      }
      p.done++;
      const s = (performance.now() - t0) / 1000;
      p.rate = s > 0.5 ? p.bytes / s : null;
      onProgress({ ...p, failed: [...p.failed] });
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
  // verify: the basemap's checksum and every item present
  let shaOk = false;
  try {
    const bm = await c.match(`/offline/${pack.id}/basemap.pmtiles`);
    const pj = await c.match(`/offline/${pack.id}/pack.json`);
    if (bm && pj) {
      const want = ((await pj.clone().json()) as { basemap?: { sha256?: string } }).basemap?.sha256;
      shaOk = !!want && (await sha256(await bm.clone().arrayBuffer())) === want;
    }
  } catch {
    shaOk = false;
  }
  let osm: string | null = null;
  try {
    const pj = await c.match(`/offline/${pack.id}/pack.json`);
    osm = pj ? ((await pj.json()) as { basemap?: { osm_data_as_of?: string } }).basemap?.osm_data_as_of ?? null : null;
  } catch {
    osm = null;
  }
  const missing = items.filter((it) => !saved.some((s) => s.url === it.url)).length;
  const record: PackRecord = {
    id: pack.id,
    version: 1,
    savedAt: new Date().toISOString(),
    ...meta,
    items: saved,
    totalBytes: saved.reduce((a, s) => a + s.bytes, 0),
    complete: missing === 0 && shaOk && !signal.aborted,
    verified: { basemapSha256: shaOk, missing, failed: p.failed.slice(0, 20) },
    osmDataAsOf: osm,
  };
  if (quota) throw Object.assign(new Error("Storage is full"), { name: "QuotaExceededError", record });
  await c.put(RECORD, new Response(JSON.stringify(record), { headers: { "content-type": "application/json" } }));
  return record;
}
