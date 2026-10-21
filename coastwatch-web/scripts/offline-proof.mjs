#!/usr/bin/env node
/**
 * Offline field-pack proof (v4), with no network at all:
 *   1. start the built app on fixture data, open the Monterey briefing, save the field pack;
 *   2. close the browser and STOP the app server;
 *   3. reopen the same browser profile offline, with every request (the service worker's too, and
 *      loopback) sent to a dead proxy;
 *   4. the briefing and Explore must open from the pack: the evidence, the saved basemap (PMTiles),
 *      the forecast grid, the offline banner; a page that was not saved must say it is offline.
 *   5. before going offline the service worker's own copy of the site's code is emptied, as after a
 *      deploy whose code files have new names: the saved pages must run on the pack's code alone;
 *   6. every request of the offline run is logged (the pages', their map workers' and the service
 *      worker's): none may leave this device, and Explore is walked through every layer, sensor and
 *      day at several zooms and pans inside the pack's box (and just outside it); every resource the
 *      pages ask for inside the box must be one the pack saved. The result is printed as a table.
 * Usage: npm run build && node scripts/offline-proof.mjs [port]   (exit code 0 = proven)
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

const PORT = Number(process.argv[2] ?? 3470);
const BASE = `http://localhost:${PORT}`;
const VP = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };
const fail = [];
const check = (ok, what) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${what}`);
  if (!ok) fail.push(what);
};

function startServer() {
  const p = spawn("npx", ["next", "start", "-p", String(PORT)], {
    env: { ...process.env, CW_DATA_BASE_URL: "/data/fixture/v1", CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
    stdio: "ignore",
    detached: true,
  });
  return p;
}
async function waitUp() {
  for (let i = 0; i < 120; i++) {
    try {
      if ((await fetch(`${BASE}/about`)).ok) return;
    } catch {
      // not yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("server did not start");
}

// every request of the offline run: the pages', their workers' and the service worker's
const requests = [];
let zone = "inside";
let step = "briefing";
function watch(ctx, phase) {
  ctx.on("request", (r) => {
    const url = r.url();
    if (/^(data|blob|chrome-extension):/.test(url)) return;
    const u = new URL(url);
    const external = u.host !== `localhost:${PORT}`;
    requests.push({ url, path: external ? url : u.pathname + u.search, external, sw: !!r.serviceWorker(), who: r.serviceWorker() ? "service worker" : `page (${phase}, ${step})`, type: r.resourceType(), zone });
  });
}

const dir = mkdtempSync(join(tmpdir(), "neritic-offline-"));
const server = startServer();
try {
  await waitUp();
  let ctx = await chromium.launchPersistentContext(dir, VP);
  let p = ctx.pages()[0] ?? (await ctx.newPage());
  await p.goto(`${BASE}/coast?place=port-550`, { waitUntil: "load" });
  await p.evaluate(() => navigator.serviceWorker.ready);
  await p.reload({ waitUntil: "load" });
  await p.getByTestId("pack-open").click();
  const go = p.getByTestId("pack-download");
  await go.waitFor();
  await p.waitForFunction(() => !document.querySelector('[data-testid="pack-download"]')?.hasAttribute("disabled"), null, { timeout: 90000 });
  await go.click();
  await p.locator('[data-testid="pack-status"][data-complete]').waitFor({ timeout: 120000 });
  check((await p.getByTestId("pack-status").getAttribute("data-complete")) === "true", "the pack is complete and verified online");
  const record = await p.evaluate(async () => (await (await caches.open("neritic-pack-monterey")).match("/__pack__/record.json")).json());
  // as after a deploy with new code file names: nothing but the pack holds the saved pages' code
  // the code cache is named for the build (/sw.js?b=<build>): empty every one
  check(await p.evaluate(async () => { const k = (await caches.keys()).filter((x) => x.startsWith("neritic-static-")); return k.length > 0 && (await Promise.all(k.map((x) => caches.delete(x)))).every(Boolean); }), "the service worker's static code cache is emptied before going offline");
  await ctx.close();

  // no app server, no network
  process.kill(-server.pid, "SIGTERM");
  await new Promise((r) => setTimeout(r, 1500));
  let down = false;
  try {
    await fetch(`${BASE}/about`);
  } catch {
    down = true;
  }
  check(down, "the app server is stopped");

  const OFFLINE = { offline: true, proxy: { server: "http://127.0.0.1:9" }, args: ["--proxy-bypass-list=<-loopback>"] };
  ctx = await chromium.launchPersistentContext(dir, { ...VP, ...OFFLINE });
  watch(ctx, "phone");
  p = ctx.pages()[0] ?? (await ctx.newPage());
  const errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  // every request that failed while offline: a resource the saved pages need must never be missing
  const failed = [];
  // (not counted: requests the page itself cancelled on leaving, and link prefetches (_rsc), which
  // fall back to a full page load that the service worker answers; that fallback is checked below)
  p.on("requestfailed", (r) => {
    if (/ERR_ABORTED/.test(r.failure()?.errorText ?? "") || /[?&]_rsc=/.test(r.url())) return;
    failed.push(r.url());
    if (process.env.DEBUG_OFFLINE) console.log(`  [debug] ${r.url().slice(-30)} frame=${r.frame()?.url()} worker=${!!r.serviceWorker()} online=${String(r.headers()["x-online"] ?? "")}`);
  });
  p.on("response", (r) => r.status() >= 400 && !/[?&]_rsc=/.test(r.url()) && failed.push(`${r.status()} ${r.url()}`));
  await p.goto(`${BASE}/coast?place=port-550`, { waitUntil: "load" });
  await p.waitForTimeout(4000);
  check((await p.getByTestId("coast-title").innerText().catch(() => "")) === "Monterey", "offline: the Monterey briefing opens");
  check((await p.locator('[data-testid="matters"] > li').count()) >= 3, "offline: the briefing's statements are there");
  check(/Offline/.test(await p.getByTestId("offline-banner").innerText().catch(() => "")), "offline: the banner says so, with the saved time");
  // a link tapped offline (client navigation fails, the browser loads the saved page)
  await p.getByRole("link", { name: "Open in Explore" }).first().click();
  await p.waitForURL(/\/\?/, { timeout: 15000 }).catch(() => null);
  await p.waitForTimeout(3000);
  const tapped = await p.evaluate(() => ({ map: !!window.__cwMap, omt: window.__cwMap?.getStyle().sources.omt?.url ?? null, online: navigator.onLine })).catch(() => ({ map: false, omt: null, online: null }));
  if (process.env.DEBUG_OFFLINE) console.log("  [debug] tapped", JSON.stringify(tapped));
  check(tapped.map && /^pmtiles:/.test(tapped.omt ?? ""), "offline: a link tapped on the briefing opens Explore on the saved map");
  await p.goto(`${BASE}/?sel=harbor:550&layer=forecast`, { waitUntil: "load" });
  await p.waitForTimeout(7000);
  const map = await p.evaluate(() => {
    const m = window.__cwMap;
    if (!m) return null;
    let n = 0;
    try {
      n = m.querySourceFeatures("omt", { sourceLayer: "water" }).length;
    } catch {
      n = 0;
    }
    return { omt: m.getStyle().sources.omt.url, water: n, forecast: !!m.getSource("forecast") };
  });
  check(!!map && /^pmtiles:/.test(map.omt), "offline: the map reads the saved PMTiles basemap");
  check(!!map && map.water > 0, "offline: basemap features are drawn from the pack");
  check(!!map && map.forecast, "offline: the C-HARM forecast is drawn from the saved grid");
  check((await p.getByTestId("inspect-title").innerText().catch(() => "")) === "Monterey", "offline: the harbour panel opens");
  // the licences travel with the data: OpenStreetMap's credit on the saved map, the glyphs' licence
  const credit = await p.locator("body").innerText().catch(() => "");
  check(/OpenStreetMap contributors/.test(credit) && /OpenMapTiles/.test(credit), "offline: the map credits OpenStreetMap contributors and OpenMapTiles (ODbL, CC-BY)");
  const ofl = await p.evaluate(() => fetch("/fonts/OFL.txt").then((r) => (r.ok ? r.text() : "")).catch(() => ""));
  check(/SIL OPEN FONT LICENSE Version 1\.1/.test(ofl), "offline: the glyphs' licence (SIL OFL 1.1) is saved with them");
  const sameOrigin = failed.filter((u) => u.includes(`localhost:${PORT}`));
  if (failed.length) console.log(`  offline requests that failed (${failed.length}):\n   ${[...new Set(failed)].slice(0, 30).join("\n   ")}`);
  check(sameOrigin.length === 0, `offline: every resource of the saved pages is in the pack (${sameOrigin.length} same-origin failures)`);
  await p.goto(`${BASE}/research`, { waitUntil: "load" }).catch(() => null);
  check(/offline/i.test(await p.locator("h1").first().innerText().catch(() => "")), "offline: an unsaved page says it is offline");
  check(errors.length === 0, `offline: no page errors (${errors.slice(0, 2).join(" | ")})`);
  await ctx.close();

  // the coverage walk: Explore on a wide screen, every layer, sensor and day, in and around the box
  ctx = await chromium.launchPersistentContext(dir, { viewport: { width: 1440, height: 900 }, ...OFFLINE });
  watch(ctx, "explore");
  p = ctx.pages()[0] ?? (await ctx.newPage());
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto(`${BASE}/?sel=harbor:550&layer=forecast`, { waitUntil: "load" });
  await p.waitForTimeout(5000);
  // v7: the layer list is a popover from the dock's layer button; open it (again) before each choice in it
  const deck = async () => {
    if (!(await p.getByTestId("layer-deck").isVisible())) await p.getByTestId("layer-capsule").click();
    await p.getByTestId("layer-deck").waitFor({ state: "visible" });
  };
  await deck();
  await p.getByTestId("variable-select").waitFor();
  const [w, s, e, n] = record.items.length ? [-122.124, 36.44, -121.676, 36.8] : [0, 0, 0, 0];
  const VIEWS = [
    ["inside z8", (w + e) / 2, (s + n) / 2, 8],
    ["inside z10", -121.9, 36.62, 10],
    ["inside z12 harbour", -121.89, 36.605, 12],
    ["inside z12.5 SW corner", w + 0.06, s + 0.05, 12.5],
    ["inside z12.5 NE corner", e - 0.06, n - 0.05, 12.5],
    ["inside z11 NW", w + 0.08, n - 0.06, 11],
    ["outside z10 N (Santa Cruz)", -122.03, 36.97, 10],
    ["outside z10 S (Big Sur)", -121.9, 36.25, 10],
    ["outside z6 coast", -121.9, 36.62, 6],
  ];
  const walk = async (what) => {
    for (const [name, lon, lat, z] of VIEWS) {
      zone = name.startsWith("outside") ? "outside" : "inside";
      step = `${what} · ${name}`;
      await p.evaluate(([lon, lat, z]) => window.__cwMap?.jumpTo({ center: [lon, lat], zoom: z }), [lon, lat, z]);
      await p.waitForTimeout(1800);
    }
    // back over the harbour before the next choice, so what it asks for is asked for inside the box
    await p.evaluate(() => window.__cwMap?.jumpTo({ center: [-121.9, 36.62], zoom: 11 }));
    await p.waitForTimeout(800);
    zone = "inside";
  };
  await walk("forecast");
  for (const v of await p.getByTestId("variable-select").locator("option").evaluateAll((os) => os.map((o) => o.value))) {
    await deck();
    await p.getByTestId("variable-select").selectOption(v).catch(() => null);
    for (const t of await p.locator('[data-testid^="time-"]:not([disabled])').all()) {
      await t.click().catch(() => null);
      await p.waitForTimeout(700);
    }
  }
  await walk("forecast, last day");
  await deck();
  await p.getByTestId("layer-satellite").click();
  await p.waitForTimeout(2000);
  for (const sensor of await p.locator('[data-testid^="sensor-"]').evaluateAll((bs) => bs.map((b) => b.dataset.testid))) {
    await deck();
    await p.getByTestId(sensor).click();
    await p.waitForTimeout(1500);
    await walk(sensor);
    if (await p.getByTestId("satview-age").count()) {
      await deck();
      await p.getByTestId("satview-age").click();
      await walk(`${sensor} age`);
      await p.getByTestId("satview-chl").click().catch(() => null);
    }
  }
  await deck();
  await p.getByTestId("layer-currents").click();
  await p.waitForTimeout(2000);
  await walk("currents");
  for (const t of await p.locator('[data-testid^="time-"]:not([disabled])').all()) {
    await t.click().catch(() => null);
    await p.waitForTimeout(700);
  }
  await ctx.close();

  // the table: every resource the offline pages asked for, against what the pack saved
  const saved = new Set(record.items.map((i) => i.url));
  const kindOf = (path) =>
    path.startsWith("/_next/static/") ? "App code (/_next/static)"
    : path.startsWith("/vendor/") ? "Map worker"
    : path.endsWith(".pmtiles") ? "Basemap PMTiles (byte ranges)"
    : path.startsWith("/offline/") ? "Pack description (pack.json)"
    : path.startsWith("/fonts/") ? "Glyph ranges"
    : /^\/basemap\/relief/.test(path) ? "Relief rasters, relief.json"
    : path.startsWith("/basemap/isobaths") ? "Isobaths"
    : path.startsWith("/basemap/") ? "Bundled water, county lines"
    : path.includes("/charm/") ? "C-HARM grids"
    : /\/satellite\/.*\/(age-|sensor-)?tiles\//.test(path) ? "Satellite tiles and indexes (chlorophyll, age, sensor)"
    : path.includes("/satellite/") ? "Satellite grids (value, age)"
    : path.includes("/currents/") ? "Currents grids"
    : path.startsWith("/api/") ? "API answers (evidence, conditions, briefing)"
    : path.startsWith("/data/") ? "Other data"
    : "Pages";
  // a map tile is inside the box when it overlaps it (a wide view also shows tiles beyond the box)
  const BOX = [-122.124, 36.44, -121.676, 36.8];
  const tileInside = (path) => {
    const t = /\/(\d+)\/(\d+)\/(\d+)\.(png|pbf)$/.exec(path);
    if (!t) return null;
    const [z, x, y] = [Number(t[1]), Number(t[2]), Number(t[3])];
    const lon = (i) => (i / 2 ** z) * 360 - 180;
    const lat = (j) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * j) / 2 ** z))) * 180) / Math.PI;
    return lon(x) < BOX[2] && lon(x + 1) > BOX[0] && lat(y + 1) < BOX[3] && lat(y) > BOX[1];
  };
  const rows = new Map();
  const unsavedInside = new Set();
  for (const r of requests) {
    if (r.external || r.sw || r.type === "document" || /[?&]_rsc=/.test(r.path) || /^\/sw\.js(\?|$)/.test(r.path)) continue;
    const k = kindOf(r.path);
    const row = rows.get(k) ?? { asked: new Set(), unsaved: new Set(), outsideUnsaved: new Set() };
    row.asked.add(r.path);
    const inside = tileInside(r.path) ?? r.zone === "inside";
    if (!saved.has(r.path)) {
      if (inside) {
        row.unsaved.add(r.path);
        unsavedInside.add(r.path);
      } else row.outsideUnsaved.add(r.path);
    }
    rows.set(k, row);
  }
  console.log("\n  Resources the map and pages asked for offline, against the pack:");
  console.log("  | Resource | asked (distinct) | saved in pack | asked inside box but not saved | asked only outside box, not saved |");
  console.log("  |---|---|---|---|---|");
  for (const [k, r] of [...rows.entries()].sort()) console.log(`  | ${k} | ${r.asked.size} | ${[...r.asked].filter((u) => saved.has(u)).length} | ${r.unsaved.size} | ${r.outsideUnsaved.size} |`);
  for (const u of unsavedInside) console.log(`   not saved: ${u}`);
  const external = requests.filter((r) => r.external);
  for (const u of [...new Set(external.map((r) => `${r.who} ${r.url}`))].slice(0, 20)) console.log(`   left the device: ${u}`);
  check(external.length === 0, `offline: no request leaves the device (${external.length} external requests, the service worker's included)`);
  check(unsavedInside.size === 0, `offline: every resource asked for inside the pack's box is saved in the pack (${unsavedInside.size} not saved)`);
  check(errors.length === 0, `offline: no page errors in the coverage walk (${errors.slice(0, 2).join(" | ")})`);
} catch (e) {
  fail.push(String(e));
  console.error(e);
} finally {
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // already stopped
  }
  rmSync(dir, { recursive: true, force: true });
}
console.log(fail.length ? `\nNOT PROVEN: ${fail.length} check(s) failed` : "\nPROVEN: the Monterey field pack works with no network after a browser restart");
process.exit(fail.length ? 1 : 0);
