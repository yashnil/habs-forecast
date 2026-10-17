#!/usr/bin/env node
/**
 * Offline field-pack proof (v4), with no network at all:
 *   1. start the built app on fixture data, open the Monterey briefing, save the field pack;
 *   2. close the browser and STOP the app server;
 *   3. reopen the same browser profile offline, with every request (the service worker's too, and
 *      loopback) sent to a dead proxy;
 *   4. the briefing and Explore must open from the pack: the evidence, the saved basemap (PMTiles),
 *      the forecast grid, the offline banner; a page that was not saved must say it is offline.
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

  ctx = await chromium.launchPersistentContext(dir, { ...VP, offline: true, proxy: { server: "http://127.0.0.1:9" }, args: ["--proxy-bypass-list=<-loopback>"] });
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
  const sameOrigin = failed.filter((u) => u.includes(`localhost:${PORT}`));
  if (failed.length) console.log(`  offline requests that failed (${failed.length}):\n   ${[...new Set(failed)].slice(0, 30).join("\n   ")}`);
  check(sameOrigin.length === 0, `offline: every resource of the saved pages is in the pack (${sameOrigin.length} same-origin failures)`);
  await p.goto(`${BASE}/research`, { waitUntil: "load" }).catch(() => null);
  check(/offline/i.test(await p.locator("h1").first().innerText().catch(() => "")), "offline: an unsaved page says it is offline");
  check(errors.length === 0, `offline: no page errors (${errors.slice(0, 2).join(" | ")})`);
  await ctx.close();
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
