#!/usr/bin/env node
/**
 * Offline field-pack proof (v12), with no network at all:
 *   1. start the built app on fixture data, open Today at Moss Landing, and save the port's field pack from
 *      the Menu (Offline, "Save Moss Landing");
 *   2. empty the service worker's own copy of the site's code (as after a deploy whose code files have new
 *      names: the saved pages must run on the pack's code alone), close the browser and STOP the app server;
 *   3. reopen the same browser profile offline, with every request (the service worker's too, and loopback)
 *      sent to a dead proxy;
 *   4. `/`, `/today`, `/coast`, `/explore` and `/research` must each open from the pack, at a phone and a desktop
 *      size: the page's own content, the baked chart (the static tier's images, loaded), Explore's baked frame,
 *      and the offline notice saying when the copy was saved and what is and is not available; a tab tapped
 *      offline opens the next saved page; a page that was not saved says it is offline and links to the pack;
 *   5. every request of the offline run is logged (the pages' and the service worker's): none may leave this
 *      device, and none of the saved pages' same-origin requests may fail.
 * Usage: npm run build && node scripts/offline-proof.mjs [port]   (exit code 0 = proven)
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

const PORT = Number(process.argv[2] ?? 3470);
const BASE = `http://localhost:${PORT}`;
const PLACE = "port-592";
const NAME = "Moss Landing";
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };
const DESK = { viewport: { width: 1440, height: 900 } };
const CHANNEL = process.env.CW_PROOF_CHANNEL ?? "chrome";
const fail = [];
const check = (ok, what) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${what}`);
  if (!ok) fail.push(what);
};

function startServer() {
  return spawn("npx", ["next", "start", "-p", String(PORT)], {
    env: { ...process.env, CW_DATA_BASE_URL: "/data/fixture/v1", CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_NWS_API: "1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
    stdio: "ignore",
    detached: true,
  });
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

const requests = [];
let step = "start";
function watch(ctx, phase) {
  ctx.on("request", (r) => {
    const url = r.url();
    if (/^(data|blob|chrome-extension):/.test(url)) return;
    const u = new URL(url);
    requests.push({ url, external: u.host !== `localhost:${PORT}`, who: r.serviceWorker() ? "service worker" : `page (${phase}, ${step})` });
  });
}

const dir = mkdtempSync(join(tmpdir(), "neritic-offline-"));
const server = startServer();
// a step that never settles fails the proof, never hangs it
const watchdog = setTimeout(() => {
  console.log(`FAIL  the proof stopped answering after ${fail.length} failed check(s); last step: ${step}`);
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // already stopped
  }
  console.log("\nNOT PROVEN: the run did not finish");
  process.exit(1);
}, 6 * 60_000);
watchdog.unref();

const RECORD = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: PLACE, fisheries: ["dungeness_commercial"], vessel: null });

try {
  await waitUp();
  // ------------------------------------------------------------ online: save the field pack from Today
  let ctx = await chromium.launchPersistentContext(dir, { ...PHONE, channel: CHANNEL });
  ctx.setDefaultTimeout(60_000);
  let p = ctx.pages()[0] ?? (await ctx.newPage());
  step = "open Today at Moss Landing";
  await p.goto(`${BASE}/about`, { waitUntil: "load" });
  await p.evaluate((rec) => {
    localStorage.setItem("nx-port-v1", rec);
    localStorage.setItem("nx-tour-v1", "dismissed");
    localStorage.setItem("nx-opening-seen-v1", "1");
  }, RECORD);
  await p.goto(`${BASE}/today?place=${PLACE}`, { waitUntil: "load" });
  await p.getByTestId("nx-today").getByRole("heading", { level: 1 }).waitFor();
  step = "save the field pack from the Menu";
  await p.getByTestId("nx-menu-button").click();
  await p.getByTestId("nx-pack-save").click();
  await p.locator('[data-testid="nx-pack"][data-complete]').waitFor({ timeout: 150_000 });
  const rec = await p.evaluate(async () => (await (await caches.open("neritic-pack-field")).match("/__field__/record.json"))?.json());
  check((await p.getByTestId("nx-pack").getAttribute("data-complete")) === "true", `the pack is complete online${rec?.missing?.length ? ` (missing: ${rec.missing.join("; ")})` : ""}`);
  check(rec?.pages?.filter((x) => x.ok).length === 4, "the pack holds Today, My Coast, Explore and Research");
  check((rec?.frames ?? []).filter((f) => f.ok).length >= 2, `the pack holds Explore's latest frames (${(rec?.frames ?? []).map((f) => `${f.layer} ${f.label}`).join(", ")})`);
  check(/Moss Landing saved/.test(await p.getByTestId("nx-pack-status").innerText().catch(() => "")), "the Menu says what is saved and when");
  console.log(`  pack: ${rec?.files} files, ${(rec?.bytes / 1e6).toFixed(1)} MB, station ${rec?.station}`);
  step = "empty the code cache";
  check(
    await p.evaluate(async () => {
      const k = (await caches.keys()).filter((x) => x.startsWith("neritic-static-"));
      return (await Promise.all(k.map((x) => caches.delete(x)))).every(Boolean);
    }),
    "the service worker's own code cache is emptied before going offline",
  );
  await ctx.close();

  // ------------------------------------------------------------ no app server, no network
  process.kill(-server.pid, "SIGTERM");
  await new Promise((r) => setTimeout(r, 1500));
  let down = false;
  try {
    await fetch(`${BASE}/about`);
  } catch {
    down = true;
  }
  check(down, "the app server is stopped");

  const OFFLINE = { offline: true, proxy: { server: "http://127.0.0.1:9" }, args: ["--proxy-bypass-list=<-loopback>"], channel: CHANNEL };
  const errors = [];
  const failed = [];
  const listen = (page) => {
    page.on("pageerror", (e) => errors.push(`${step}: ${e.message}`));
    page.on("requestfailed", (r) => {
      // a page's own cancelled requests on leaving, and link prefetches (_rsc) that fall back to a full load
      if (/ERR_ABORTED/.test(r.failure()?.errorText ?? "") || /[?&]_rsc=/.test(r.url())) return;
      failed.push(`${step}: ${r.url()}`);
    });
    page.on("response", (r) => r.status() >= 400 && !/[?&]_rsc=/.test(r.url()) && failed.push(`${step}: ${r.status()} ${r.url()}`));
  };

  const PAGES = [
    ["/", async (pg) => /Season/.test(await pg.getByTestId("nx-today").getByRole("heading", { level: 1 }).innerText()), "Today at Moss Landing"],
    ["/today", async (pg) => /Season/.test(await pg.getByTestId("nx-today").getByRole("heading", { level: 1 }).innerText()), "Today at Moss Landing"],
    ["/coast", async (pg) => (await pg.getByTestId("coast-worksheet").count()) > 0 && /Moss Landing/.test(await pg.getByTestId("nx-coast").innerText()), "My Coast with the worksheet"],
    ["/explore", async (pg) => (await pg.getByTestId("nx-static-field").count()) > 0 && (await pg.getByTestId("nx-static-field").evaluate((i) => i.complete && i.naturalWidth > 0)), "Explore with its baked frame"],
    ["/research", async (pg) => (await pg.getByTestId("research-title").count()) > 0 && (await pg.getByTestId("research-record").count()) > 0, "Research with the station record"],
  ];
  for (const [size, vp] of [
    ["phone", PHONE],
    ["desktop", DESK],
  ]) {
    ctx = await chromium.launchPersistentContext(dir, { ...vp, ...OFFLINE });
    ctx.setDefaultTimeout(30_000);
    watch(ctx, size);
    p = ctx.pages()[0] ?? (await ctx.newPage());
    listen(p);
    for (const [path, content, what] of PAGES) {
      step = `${size} ${path}`;
      await p.goto(`${BASE}${path}`, { waitUntil: "load" }).catch((e) => failed.push(`${step}: ${e.message}`));
      await p.waitForTimeout(3500);
      check(await content(p).catch(() => false), `offline, ${size}: ${path} opens ${what}`);
      const chart = await p
        .evaluate(() => {
          const imgs = [...document.querySelectorAll('[data-testid="nx-poster"] img')];
          return { tier: document.querySelector('[data-testid="nx-sheet"]')?.getAttribute("data-tier"), n: imgs.length, loaded: imgs.filter((i) => i.complete && i.naturalWidth > 0).length };
        })
        .catch(() => ({ tier: null, n: 0, loaded: 0 }));
      check(chart.tier === "static" && chart.n > 0 && chart.loaded === chart.n, `offline, ${size}: ${path} draws the baked chart (${chart.loaded}/${chart.n} images, tier ${chart.tier})`);
      const notice = await p.getByTestId("nx-offline-notice").innerText().catch(() => "");
      check(/Offline\. Showing what this device saved/.test(notice) && /wait for a connection|waits for a connection/.test(notice), `offline, ${size}: ${path} says it is a saved copy, from when, and what waits for a connection`);
    }
    if (size === "phone") {
      // a tab tapped offline: the client navigation fails and the browser loads the saved page
      step = "phone: tap Explore from Today";
      await p.goto(`${BASE}/today?place=${PLACE}`, { waitUntil: "load" });
      await p.waitForTimeout(2500);
      await p.getByTestId("nx-tabs").getByRole("link", { name: "Explore" }).click();
      await p.waitForURL(/\/explore/, { timeout: 15_000 }).catch(() => null);
      await p.waitForTimeout(3500);
      check((await p.getByTestId("nx-explore").count()) > 0, "offline, phone: a tab tapped on Today opens the saved Explore");
      // a page the pack never held
      step = "phone: an unsaved page";
      await p.goto(`${BASE}/sources`, { waitUntil: "load" }).catch(() => null);
      const off = await p.getByTestId("offline-page").innerText().catch(() => "");
      check(/You are offline/.test(off) && /Explore at Moss Landing/.test(off), "offline: a page that was not saved says so and links to the saved pages");
    }
    await ctx.close();
  }

  const sameOrigin = failed.filter((u) => u.includes(`localhost:${PORT}`));
  if (failed.length) console.log(`  offline requests that failed (${failed.length}):\n   ${[...new Set(failed)].slice(0, 30).join("\n   ")}`);
  check(sameOrigin.length === 0, `offline: every resource of the saved pages is in the pack (${sameOrigin.length} same-origin failures)`);
  const external = requests.filter((r) => r.external);
  for (const u of [...new Set(external.map((r) => `${r.who} ${r.url}`))].slice(0, 20)) console.log(`   left the device: ${u}`);
  check(external.length === 0, `offline: no request leaves the device (${external.length} external requests, the service worker's included)`);
  check(errors.length === 0, `offline: no page errors (${errors.slice(0, 3).join(" | ")})`);
} catch (e) {
  fail.push(`${step}: ${String(e)}`);
  console.error(e);
} finally {
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {
    // already stopped
  }
  rmSync(dir, { recursive: true, force: true });
}
console.log(fail.length ? `\nNOT PROVEN: ${fail.length} check(s) failed` : `\nPROVEN: ${NAME}'s field pack opens every v12 page with no network after a browser restart`);
process.exit(fail.length ? 1 : 0);
