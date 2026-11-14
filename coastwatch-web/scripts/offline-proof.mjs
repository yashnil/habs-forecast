#!/usr/bin/env node
/**
 * Offline field-pack proof (v12), with no network at all:
 *   1. start the built app on fixture data, open Today at Moss Landing, and save the port's field pack from
 *      the Menu (Offline, "Save Moss Landing");
 *   2. empty the service worker's own copy of the site's code (as after a deploy whose code files have new
 *      names: the saved pages must run on the pack's code alone), close the browser and STOP the app server;
 *   3. reopen the same browser profile offline, with every request (the service worker's too, and loopback)
 *      sent to a dead proxy;
 *   4. `/`, `/today`, `/notices`, `/coast`, `/explore` and `/research` must each open from the pack, at a phone and a desktop
 *      size: the page's own content, the baked chart (the static tier's images, loaded), Explore's baked frame,
 *      and the offline notice saying when the copy was saved and what is and is not available; a tab tapped
 *      offline opens the next saved page; a page that was not saved says it is offline and links to the pack;
 *   4b. (v13, journey 6) a trip worked out and saved on My Coast before going offline (the owner's case, net to
 *      boat $2,602.24) is in the saved list after the restart, reopens to the cent, takes an Update with no network
 *      ($2,922.24 at $10/lb), and the updated trip is what a second offline restart opens;
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

// ---------------------------------------------------------------- the trip workspace (v13, journey 6)
const typeIn = async (pg, label, v) => {
  const f = pg.getByRole("textbox", { name: label, exact: true });
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
};
/** The owner's hands-on case: 400 lb at $9, 30 gal at $8.24, bait and ice $100, fees 0, one crew on 20 % after costs, shared. */
async function fillTrip(pg) {
  for (const [l, v] of [["Catch", "400"], ["Price", "9"], ["Fuel", "30"], ["Fuel price", "8.24"], ["Bait and ice", "100"], ["Fees", "0"], ["Crew", "1"]]) await typeIn(pg, l, v);
  await pg.getByRole("radio", { name: "After costs" }).click();
  await typeIn(pg, "Crew share", "20");
  await pg.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await pg.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
  await pg.waitForTimeout(600);
}
const netExact = (pg) => pg.getByTestId("trip-net").locator("[data-figure-exact]").innerText().then((t) => t.trim()).catch(() => "(no figure)");
const said = (pg) => pg.getByTestId("trip-said").innerText().then((t) => t.replace(/\s+/g, " ").trim()).catch(() => "");
/** Offline: My Coast's brief, into the workspace, the saved list, and the one saved trip opened from it. */
async function reopenTrip(pg) {
  await pg.goto(`${BASE}/coast?place=${PLACE}`, { waitUntil: "load" });
  await pg.getByTestId("coast-to-trip").click();
  await pg.getByTestId("trip-inputs").waitFor();
  // start from an empty sheet, so the figure can only come from the saved trip
  if (await pg.getByTestId("trip-clear").count()) await pg.getByTestId("trip-clear").click();
  await pg.getByTestId("trip-saved-open").click();
  const dialog = pg.getByRole("dialog", { name: "Saved trips" });
  const rows = await dialog.getByTestId("trip-saved-row").count();
  await dialog.getByRole("button", { name: /^Open / }).first().click();
  await pg.waitForTimeout(800);
  if (await dialog.isVisible().catch(() => false)) await pg.keyboard.press("Escape");
  return rows;
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
  check(["today", "notices", "coast", "explore", "research"].every((id) => rec?.pages?.some((x) => x.id === id && x.ok)), "the pack holds Today, the official notices, My Coast, Explore and Research");
  check((rec?.frames ?? []).filter((f) => f.ok).length >= 2, `the pack holds Explore's latest frames (${(rec?.frames ?? []).map((f) => `${f.layer} ${f.label}`).join(", ")})`);
  check(/Moss Landing saved/.test(await p.getByTestId("nx-pack-status").innerText().catch(() => "")), "the Menu says what is saved and when");
  console.log(`  pack: ${rec?.files} files, ${(rec?.bytes / 1e6).toFixed(1)} MB, station ${rec?.station}`);
  step = "save a trip on My Coast (the owner's case)";
  await p.evaluate(() => localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" })));
  await p.goto(`${BASE}/coast?place=${PLACE}&view=trip`, { waitUntil: "load" });
  await p.getByTestId("trip-inputs").waitFor();
  await fillTrip(p);
  check((await netExact(p)) === "$2,602.24", `online: the owner's trip works out to $2,602.24 net to boat (${await netExact(p)})`);
  await p.getByTestId("trip-save-new").scrollIntoViewIfNeeded();
  await p.getByTestId("trip-save-new").click();
  check(/^Saved as/.test(await said(p)), `online: the trip is saved on this device (${(await said(p)).slice(0, 80)})`);
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
    [
      "/coast",
      async (pg) => {
        const t = await pg.getByTestId("nx-coast").innerText();
        return /Moss Landing/.test(t) && /Gates/.test(t) && /Sea window/.test(t) && /delayed to at least Jan 1, 2027/.test(await pg.getByTestId("coast-status").innerText()) && (await pg.getByTestId("coast-to-trip").count()) > 0;
      },
      "My Coast's brief: the season status, the gates and the sea window from the pack, and the way into the trip workspace",
    ],
    ["/explore", async (pg) => (await pg.getByTestId("nx-static-field").count()) > 0 && (await pg.getByTestId("nx-static-field").evaluate((i) => i.complete && i.naturalWidth > 0)), "Explore with its baked frame"],
    ["/notices", async (pg) => (await pg.getByTestId("nx-notice").count()) > 0 && /not verified/.test(await pg.getByTestId("nx-notices").innerText()), "every official notice, not verified"],
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
      // a link tapped offline: the client navigation fails and the browser loads the saved page (v14: Explore is
      // reached from the Menu, no longer a destinations tab)
      step = "phone: open Explore from the Menu on Today";
      await p.goto(`${BASE}/today?place=${PLACE}`, { waitUntil: "load" });
      await p.waitForTimeout(2500);
      await p.getByTestId("nx-menu-button").click();
      await p.getByRole("navigation", { name: "More" }).getByRole("link", { name: /Explore/ }).click();
      await p.waitForURL(/\/explore/, { timeout: 15_000 }).catch(() => null);
      await p.waitForTimeout(3500);
      check((await p.getByTestId("nx-explore").count()) > 0, "offline, phone: Explore opened from the Menu on Today opens the saved Explore");
      // a page the pack never held
      step = "phone: an unsaved page";
      await p.goto(`${BASE}/sources`, { waitUntil: "load" }).catch(() => null);
      const off = await p.getByTestId("offline-page").innerText().catch(() => "");
      check(/You are offline/.test(off) && /Explore at Moss Landing/.test(off), "offline: a page that was not saved says so and links to the saved pages");
    }
    // journey 6: the saved trip, with no server and no network. The phone run reopens it and updates it; the
    // desktop run is a second offline restart and must find the update.
    step = `${size}: reopen the saved trip`;
    const rows = await reopenTrip(p).catch((e) => (failed.push(`${step}: ${e.message.split("\n")[0]}`), -1));
    check(rows === 1, `offline, ${size}: the saved list holds the one trip saved before the restart (${rows})`);
    const want = size === "phone" ? "$2,602.24" : "$2,922.24";
    check((await netExact(p)) === want, `offline, ${size}: the saved trip reopens and works out to ${want} net to boat (${await netExact(p)})${size === "desktop" ? ", the update made offline before this restart" : ""}`);
    check(/Hypothetical/.test(await p.locator("[data-nx-desk-header]").innerText().catch(() => "")), `offline, ${size}: the workspace still says the trip is hypothetical`);
    if (size === "phone") {
      step = "phone: update the saved trip offline";
      await typeIn(p, "Price", "10");
      await p.getByTestId("trip-update").scrollIntoViewIfNeeded();
      await p.getByTestId("trip-update").click();
      check(/^Updated/.test(await said(p)), `offline, phone: Update changes the saved trip with no network (${(await said(p)).slice(0, 60)})`);
      check((await netExact(p)) === "$2,922.24", `offline, phone: at $10/lb the trip works out to $2,922.24 (${await netExact(p)})`);
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
console.log(fail.length ? `\nNOT PROVEN: ${fail.length} check(s) failed` : `\nPROVEN: ${NAME}'s field pack opens every page, and the trip saved on My Coast, with no network after a browser restart`);
process.exit(fail.length ? 1 : 0);
