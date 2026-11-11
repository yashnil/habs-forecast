"use client";

import { staticRefs } from "@/v4/offline/kept";
import { saveTodayOffline } from "@/pilot/today/offline";
import { COAST_FISHERIES } from "@/nx/coast/fisheries";
import { gatesUrl, seaUrl } from "@/nx/coast/useCoast";
import { conditionsUrl } from "@/nx/today/data";

/**
 * A port's field pack (README §12: "offline field pack keeps every page readable with the baked chart"). One
 * control saves, for the reader's port: Today (the copy Today keeps of itself, `neritic-pack-today`), My Coast
 * with its gates for every fishery and its sea window (the worksheet itself lives in this device's storage),
 * Explore with each layer's latest frame around the port, and Research with the station record open now or
 * nearest the port. Everything goes into one cache, `neritic-pack-field`, which the service worker (public/sw.js)
 * answers from when the network is gone, like every `neritic-pack-*` cache. Saving again replaces the pack.
 *
 * A pack only ever shows its own dates: every saved page carries the time it was saved (x-neritic-saved-at),
 * the worker marks it (self.__NERITIC_SAVED__), and the pages draw the baked chart (the static tier) under it.
 */
export const FIELD_CACHE = "neritic-pack-field";
export const FIELD_RECORD = "/__field__/record.json";
const RECORD_KEY = "nx-field-v1";
const SW_URL = `/sw.js${process.env.NEXT_PUBLIC_NERITIC_BUILD ? `?b=${process.env.NEXT_PUBLIC_NERITIC_BUILD}` : ""}`;

export type PackPage = { id: "today" | "notices" | "coast" | "explore" | "research"; label: string; path: string; ok: boolean };
export type PackFrame = { layer: string; label: string; stamp: string; ok: boolean };
export type FieldRecord = {
  v: 1;
  port: { key: string; name: string };
  savedAt: string;
  pages: PackPage[];
  /** Explore's latest frame of each layer, with its own stamp */
  frames: PackFrame[];
  /** the Research station whose record was saved */
  station: string | null;
  files: number;
  bytes: number;
  /** what could not be saved, in words */
  missing: string[];
};

export type SavePort = { key: string; name: string; lat: number; lon: number };
/** How far a save has got: the part being saved now, in words ("My Coast"). */
export type Progress = { step: string };

// ------------------------------------------------------------------ the record, readable without waiting

const listeners = new Set<() => void>();
let memo: FieldRecord | null | undefined;

function parse(raw: string | null): FieldRecord | null {
  if (!raw) return null;
  try {
    const r = JSON.parse(raw) as FieldRecord;
    return r && r.v === 1 && typeof r.savedAt === "string" && r.port && Array.isArray(r.pages) ? r : null;
  } catch {
    return null;
  }
}

/** The pack on this device, or null. Read from storage (the cache is checked by `verifyRecord`). */
export function readRecord(): FieldRecord | null {
  if (memo !== undefined) return memo;
  try {
    memo = parse(window.localStorage.getItem(RECORD_KEY));
  } catch {
    memo = null;
  }
  return memo;
}

function writeRecord(r: FieldRecord | null) {
  memo = r;
  try {
    if (r) window.localStorage.setItem(RECORD_KEY, JSON.stringify(r));
    else window.localStorage.removeItem(RECORD_KEY);
  } catch {
    // storage blocked: the record lives in the cache and this page view
  }
  listeners.forEach((l) => l());
}

export function subscribeRecord(l: () => void) {
  listeners.add(l);
  const onStorage = (e: StorageEvent) => {
    if (e.key === RECORD_KEY || e.key === null) {
      memo = undefined;
      l();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(l);
    window.removeEventListener("storage", onStorage);
  };
}

/** The cache is the truth: a browser that evicted it (or a private window) forgets the record; a cache whose
 *  record was not in storage (cleared site data, another tab) brings it back. */
export async function verifyRecord(): Promise<FieldRecord | null> {
  if (typeof caches === "undefined") return readRecord();
  try {
    if (!(await caches.has(FIELD_CACHE))) {
      if (readRecord()) writeRecord(null);
      return null;
    }
    const r = await (await caches.open(FIELD_CACHE)).match(FIELD_RECORD);
    const rec = r ? parse(await r.text()) : null;
    if (JSON.stringify(rec) !== JSON.stringify(readRecord())) writeRecord(rec);
    return rec;
  } catch {
    return readRecord();
  }
}

export async function removePack(): Promise<void> {
  try {
    await caches.delete(FIELD_CACHE);
  } catch {
    // nothing to remove
  }
  writeRecord(null);
}

// ------------------------------------------------------------------ saving

/** Same-origin files a page names besides its code: icons, the manifest, images. */
export function publicRefs(html: string): string[] {
  const out = new Set<string>();
  for (const m of html.matchAll(/(?:href|src)="(\/(?!\/)[^"?#]+\.(?:ico|png|webp|svg|jpg|webmanifest|txt))(?:\?[^"]*)?"/g)) out.add(m[1]);
  for (const m of html.matchAll(/<link[^>]+rel="manifest"[^>]+href="(\/[^"]+)"/g)) out.add(m[1]);
  return [...out];
}

/** The baked chart the static tier draws, in every mode, and the land it bakes data frames against. */
export const CHART_FILES = [
  ...["day", "dusk", "night"].flatMap((m) => [`/chart/chart-${m}-ca.webp`, `/chart/chart-${m}-monterey.webp`]),
  "/chart/land.geojson",
  "/chart/opening-canyon.webp",
];

const savedHtml = (html: string, at: string, url: string, label: string) =>
  new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8", "x-neritic-saved-at": at, "x-neritic-saved-url": url, "x-neritic-saved-label": encodeURIComponent(label) } });

/**
 * Save the port's field pack. `researchPath` is the Research address open now (its station), else the port's.
 * Progress is reported per file; a part that fails is named in the record's `missing`, never hidden.
 */
export async function saveFieldPack(port: SavePort, opts: { researchPath?: string | null; onProgress?: (p: Progress) => void } = {}): Promise<{ ok: true; record: FieldRecord } | { ok: false; why: string }> {
  if (typeof window === "undefined" || !("caches" in window)) return { ok: false, why: "this browser cannot keep pages offline." };
  if (!navigator.onLine) return { ok: false, why: "you are offline now; save while you have a connection." };
  const at = new Date().toISOString();
  const place = encodeURIComponent(port.key);
  const missing: string[] = [];
  let bytes = 0;
  let files = 0;
  try {
    if ("serviceWorker" in navigator) {
      await navigator.serviceWorker.register(SW_URL, { scope: "/" });
      await navigator.serviceWorker.ready;
    }
    // a fresh pack: the old one goes only once the new one is written
    const tmpName = `${FIELD_CACHE}-new`;
    await caches.delete(tmpName);
    const c = await caches.open(tmpName);
    const put = async (url: string, r: Response) => {
      const b = await r.clone().blob();
      bytes += b.size;
      files += 1;
      await c.put(url, r);
    };

    // Today keeps its own copy (and the list the offline page names); the pack asks it to refresh it now
    const step = (s: string) => opts.onProgress?.({ step: s });
    step("Today");
    const curl = conditionsUrl(port.lat, port.lon);
    const today = await saveTodayOffline(port.key, port.name, curl);
    if (!today.ok) missing.push(`Today (${today.why.replace(/\.$/, "")})`);

    const researchPath = opts.researchPath && opts.researchPath.startsWith("/research") ? opts.researchPath : `/research?place=${place}`;
    const pageDefs: Array<Omit<PackPage, "ok">> = [
      { id: "today", label: "Today", path: `/today?place=${place}` },
      { id: "notices", label: "Official notices", path: `/notices?place=${place}` },
      { id: "coast", label: "My Coast", path: `/coast?place=${place}` },
      { id: "explore", label: "Explore", path: `/explore?place=${place}` },
      { id: "research", label: "Research", path: researchPath },
    ];
    const plan = await fetch(`/explore/files?place=${place}`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<{ ok: boolean; frames: Array<{ layer: string; label: string; stamp: string; files: string[] }> }>) : null))
      .catch(() => null);
    if (!plan?.ok) missing.push("Explore's frames (the list of files did not load)");
    const frameFiles = plan?.ok ? plan.frames : [];
    const data = [
      `/today/data?place=${place}`,
      curl,
      seaUrl(port.key),
      ...COAST_FISHERIES.map((f) => gatesUrl(port.key, f.id)),
    ];

    // the pages, and every file each one loads
    const code = new Set<string>();
    const pages: PackPage[] = [];
    let station: string | null = null;
    for (const p of pageDefs) {
      step(p.label);
      const r = await fetch(p.path, { credentials: "same-origin", cache: "no-store" }).catch(() => null);
      if (!r?.ok) {
        pages.push({ ...p, ok: false });
        missing.push(`${p.label} (the page did not load)`);
        continue;
      }
      const html = await r.text();
      await put(p.path, savedHtml(html, at, p.path, port.name));
      pages.push({ ...p, ok: true });
      for (const f of [...staticRefs(html), ...publicRefs(html)]) code.add(f);
      if (p.id === "research") station = /data-station="([^"]+)"/.exec(html)?.[1] ?? null;
    }

    // the pages' answers: Today's records and sea, My Coast's gates for every fishery and its sea window
    step("the records and forecasts");
    for (const u of data) {
      const r = await fetch(u, { cache: "no-store" }).catch(() => null);
      if (r?.ok) await put(u, r);
      else if (u.includes("part=sea")) missing.push("My Coast's sea window (the forecast did not load)");
      else if (u.startsWith("/api/conditions")) missing.push("Today's sea conditions (the forecast did not load)");
    }

    // code, styles, fonts (named by the pages and their stylesheets), then the baked chart
    const css: string[] = [];
    const fetchInto = async (f: string) => {
      const r = await fetch(f).catch(() => null);
      if (!r?.ok) return false;
      if (f.endsWith(".css")) css.push(await r.clone().text());
      await put(f, r).catch(() => undefined);
      return true;
    };
    step("the app's code and type");
    // the web app manifest's icons (the browser asks for them on every page)
    if (code.has("/manifest.webmanifest")) {
      const man = await fetch("/manifest.webmanifest").then((r) => (r.ok ? (r.json() as Promise<{ icons?: Array<{ src?: string }> }>) : null)).catch(() => null);
      for (const i of man?.icons ?? []) if (i.src?.startsWith("/")) code.add(i.src);
    }
    await Promise.all([...code].map(fetchInto));
    const fonts = css.flatMap(staticRefs).filter((f) => !code.has(f));
    await Promise.all(fonts.map(fetchInto));
    step("the baked chart");
    const chart = await Promise.all(CHART_FILES.map(fetchInto));
    if (!chart.every(Boolean)) missing.push("part of the baked chart");

    // Explore: each layer's latest frame around the port
    const frames: PackFrame[] = [];
    for (const f of frameFiles) {
      step(`Explore's ${f.layer === "C-HARM" ? f.layer : f.layer.toLowerCase()}`);
      const ok = (await Promise.all(f.files.map(fetchInto))).every(Boolean);
      frames.push({ layer: f.layer, label: f.label, stamp: f.stamp, ok });
      if (!ok) missing.push(`Explore's ${f.layer.toLowerCase()} frame`);
    }

    const record: FieldRecord = { v: 1, port: { key: port.key, name: port.name }, savedAt: at, pages, frames, station, files, bytes, missing };
    await c.put(FIELD_RECORD, new Response(JSON.stringify(record), { headers: { "Content-Type": "application/json" } }));
    // swap: copy the new pack over the old one's name (Cache Storage cannot rename)
    await caches.delete(FIELD_CACHE);
    const dst = await caches.open(FIELD_CACHE);
    for (const req of await c.keys()) {
      const r = await c.match(req);
      if (r) await dst.put(req, r);
    }
    await caches.delete(tmpName);
    writeRecord(record);
    return { ok: true, record };
  } catch (e) {
    return { ok: false, why: e instanceof Error && /quota/i.test(e.message) ? "this device is out of storage." : "the browser refused to store it." };
  }
}

/** What a page says it has offline, and what it cannot have. Pure (tests read it). */
export function packWords(r: FieldRecord | null, page: string): { saved: string | null; notSaved: string } {
  const notSaved: Record<string, string> = {
    today: "Newer tests, notices and forecasts, Alerts and Ask wait for a connection.",
    notices: "Notices issued or changed since then wait for a connection; call the agencies to be sure.",
    coast: "A newer sea window waits for a connection. The worksheet works as usual and stays on this device.",
    explore: "Only each layer's latest frame is saved, as a still picture: earlier frames, temperature lines, currents and tapping the water wait for a connection.",
    research: "Other stations, the native grid and the model's cells wait for a connection. The export works from what is on this page.",
    home: "Newer records wait for a connection.",
  };
  if (!r) return { saved: null, notSaved: notSaved[page] ?? notSaved.home };
  const parts = r.pages.filter((p) => p.ok && p.id !== "notices").map((p) => (p.id === "explore" ? "Explore's latest frames" : p.id === "research" ? "Research's record" : p.label));
  return { saved: parts.length ? `Saved for ${r.port.name}: ${list(parts)}.` : null, notSaved: notSaved[page] ?? notSaved.home };
}

export function list(xs: string[]): string {
  if (xs.length <= 1) return xs.join("");
  return `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}
