#!/usr/bin/env node
/**
 * Neritic v15.1 demo moments (DIRECTION §8) and a short walkthrough video, from the running app on fixture data:
 *   npm run data:fixture && npm run build
 *   CW_DATA_BASE_URL=/data/fixture/v1 CW_CONDITIONS_FIXTURE=tests/conditions-fixtures.json CW_NWS_API=1 \
 *     CW_FIXED_NOW=2026-10-13T06:00:00Z npx next start -p 7590
 *   node scripts/v151-shots.mjs [baseUrl] [outDir] [qaDir]       # run outside a sandbox (installed Chrome)
 *
 * The ten moments, at 1440×900, 1280×800, 1100×800, 390×844 and 320×640:
 *   01 port-centred Today            02 the transition to statewide (mid-flight and settled)
 *   03 chlorophyll with the time rail 04 C-HARM with its caveats      05 observed HF-radar currents
 *   06 zoom into a coastal region     07 a floating panel moved on its own (desk sizes only)
 *   08 Research where local observations are sparse (Crescent City)
 *   09 About, Privacy and Sources on the new system                10 return to the saved trip
 * A moment whose control is missing in this build is logged as MISSING and not faked. A desk-size video of the flow
 * (Today → statewide → layers → zoom → trip) is recorded with Playwright's recordVideo.
 * The 1440 and 390 frames and the video are copied to docs/coastwatch/v15.1/qa/ at the repository root.
 */
import { copyFileSync, mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:7590";
const OUT = resolve(process.argv[3] ?? "shots/v151");
const QA = resolve(process.argv[4] ?? "../docs/coastwatch/v15.1/qa");
const NOW = "2026-10-13T06:00:00Z";
const MOSS = "port-592";
const CRESCENT = "port-201";
const CHANNEL = process.env.CW_SHOTS_CHANNEL ?? "chrome";
const ONLY = process.env.CW_SHOTS_SIZES?.split(",");
const SIZES = [
  { tag: "1440", viewport: { width: 1440, height: 900 } },
  { tag: "1280", viewport: { width: 1280, height: 800 } },
  { tag: "1100", viewport: { width: 1100, height: 800 } },
  { tag: "390", viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
  { tag: "320", viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true },
].filter((s) => !ONLY || ONLY.includes(s.tag));
const COPY = new Set(["1440", "390"]);
const RETURNING = {
  "nx-opening-seen-v1": "1",
  "nx-tour-v1": "dismissed",
  "nx-port-v1": JSON.stringify({ v: 1, place: MOSS, for: "all", species: "any", lastPort: MOSS, fisheries: ["dungeness_commercial"], vessel: null }),
  "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }),
};
const shots = [];
const missing = [];
const errors = [];
const settle = (p, ms = 1200) => p.waitForTimeout(ms);
const mapReady = (p) => p.locator("[data-testid=nx-sheet][data-map-ready='1'], [data-testid=nx-sheet][data-tier=static]").first().waitFor({ timeout: 20_000 }).catch(() => {});

async function shot(page, size, name, o = {}) {
  await settle(page, o.wait ?? 1200);
  const file = join(OUT, `${size.tag}-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!o.full });
  shots.push(file);
  console.log("shot", file);
}

const miss = (size, name, why) => {
  missing.push(`${size.tag}-${name}: ${why}`);
  console.log("MISSING", size.tag, name, why);
};

/** The first locator of `list` that is visible, or null. */
async function first(...list) {
  for (const l of list) {
    const f = l.first();
    if (await f.isVisible().catch(() => false)) return f;
  }
  return null;
}

// ---------------------------------------------------------------- v15.1 controls (agents A, B)

const exploreEntry = (p) => first(p.getByTestId("nx-explore-entry"), p.getByRole("link", { name: /explore california waters/i }), p.getByRole("button", { name: /explore california waters/i }));
const backToPort = (p) => first(p.getByTestId("nx-explore-port"), p.getByRole("button", { name: /back to moss landing/i }), p.getByRole("link", { name: /back to moss landing/i }));
const layer = (p, id) => first(p.getByTestId(`nx-layer-${id}`));
const panelHandle = (p) => first(p.locator("[data-panel-handle]"), p.locator("[data-testid$=-handle]"), p.getByRole("button", { name: /^(move|drag)\b/i }));

async function caBounds(page) {
  return page.evaluate(() => {
    const m = window.__cwMap;
    if (!m) return null;
    const b = m.getBounds();
    return { s: b.getSouth(), n: b.getNorth() };
  });
}

// ---------------------------------------------------------------- the trip (tests/e2e/v15/helpers.ts planTrip)

async function field(page, label, v) {
  const f = page.getByRole("textbox", { name: label, exact: true }).first();
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
}

async function plan(page) {
  await page.goto(`${BASE}/coast?place=${MOSS}`);
  await page.getByTestId("coast-status").first().waitFor({ timeout: 20_000 });
  await page.getByTestId("coast-fishery").first().selectOption("dungeness_commercial");
  await page.getByTestId("plan-date").fill("2026-10-14");
  await page.getByTestId("plan-more").getByRole("button").click();
  await field(page, "Wind", "12");
  await field(page, "Seas", "6");
  await page.getByTestId("coast-to-trip").click();
  await page.getByTestId("trip-inputs").waitFor();
  for (const [l, v] of [["Catch", "400"], ["Price", "9"], ["Fuel", "30"], ["Fuel price", "8.24"], ["Bait and ice", "100"], ["Fees", "0"], ["Crew", "1"]]) await field(page, l, v);
  await page.getByRole("radio", { name: "After costs" }).click();
  await field(page, "Crew share", "20");
  await page.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await page.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
  await page.getByTestId("trip-plan-save").click();
  await page.waitForURL(/\/coast\/trips\?id=/);
  await page.getByTestId("trip-workspace").waitFor();
  return new URL(page.url()).searchParams.get("id");
}

// ---------------------------------------------------------------- the moments

async function moments(browser, size, o = {}) {
  const ctx = await browser.newContext({ viewport: size.viewport, isMobile: size.isMobile, hasTouch: size.hasTouch, ...(o.video ? { recordVideo: { dir: o.video, size: size.viewport } } : {}) });
  await ctx.addInitScript((kv) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, RETURNING);
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(`[${size.tag}] ${e.message}`));
  await page.clock.setFixedTime(new Date(NOW));
  const desk = size.viewport.width >= 1024;

  // the trip is saved first, so the journey can return to it
  const id = await plan(page);

  // 01 port-centred Today
  await page.goto(`${BASE}/today?place=${MOSS}`);
  await mapReady(page);
  await shot(page, size, "01-today-port", { wait: 2000 });

  // 02 the transition to statewide
  const entry = await exploreEntry(page);
  if (entry) {
    await entry.click();
    await settle(page, 700);
    await page.screenshot({ path: join(OUT, `${size.tag}-02a-transition.png`) });
    shots.push(join(OUT, `${size.tag}-02a-transition.png`));
    await page.waitForFunction(() => window.__cwMap && !window.__cwMap.isMoving(), null, { timeout: 15_000 }).catch(() => {});
    const b = await caBounds(page);
    if (!b || b.n < 41.85 || b.s > 32.7) miss(size, "02-statewide", `the view after the entry is not statewide (${JSON.stringify(b)})`);
    await shot(page, size, "02b-statewide", { wait: 1500 });
  } else {
    miss(size, "02-statewide", "no Explore California Waters entry on Today");
    await page.goto(`${BASE}/explore?place=${MOSS}`);
    await mapReady(page);
  }

  // 03 chlorophyll with the time rail
  const chl = await layer(page, "chl");
  if (chl) {
    await chl.click();
    await shot(page, size, "03-chlorophyll-time", { wait: 2000 });
  } else miss(size, "03-chlorophyll", "no chlorophyll layer control");

  // 04 C-HARM with its caveats
  const charm = await layer(page, "charm");
  if (charm) {
    await charm.click();
    await shot(page, size, "04-charm-caveats", { wait: 2000 });
  } else miss(size, "04-charm", "no C-HARM layer control");

  // 05 observed currents
  const cur = await first(page.getByTestId("nx-layer-currents"), page.getByTestId("nx-currents-toggle"));
  if (cur) {
    if (chl) await chl.click();
    if ((await cur.getAttribute("aria-checked")) !== "true") await cur.click();
    await shot(page, size, "05-currents", { wait: 2500 });
  } else miss(size, "05-currents", "no currents control");

  // 06 zoom into a coastal region: back to the port, then a wheel zoom into Monterey Bay
  const port = await backToPort(page);
  if (port) {
    await port.click();
    await page.waitForFunction(() => window.__cwMap && !window.__cwMap.isMoving(), null, { timeout: 15_000 }).catch(() => {});
  } else miss(size, "06-zoom", "no way back to the port from the statewide map (zoomed by wheel instead)");
  if (desk) {
    await page.mouse.move(size.viewport.width * 0.62, size.viewport.height * 0.5);
    for (let i = 0; i < 3; i++) {
      await page.mouse.wheel(0, -300);
      await settle(page, 250);
    }
  }
  await shot(page, size, "06-coastal-zoom", { wait: 2000 });

  // 07 a floating panel moved on its own (desk sizes: phones use sheets)
  if (desk) {
    await page.goto(`${BASE}/coast?place=${MOSS}`);
    await page.getByTestId("coast-status").first().waitFor({ timeout: 20_000 }).catch(() => {});
    await settle(page, 1200);
    const h = await panelHandle(page);
    if (h) {
      const box = await h.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 260, box.y + box.height / 2 + 120, { steps: 12 });
      await page.mouse.up();
      await shot(page, size, "07-panel-moved");
    } else miss(size, "07-panel", "no draggable panel handle on My Coast");
  }

  // 08 Research where local observations are sparse
  await page.goto(`${BASE}/research?place=${CRESCENT}`);
  await mapReady(page);
  await page.waitForLoadState("networkidle").catch(() => {});
  await shot(page, size, "08-research-sparse", { wait: 3500 });

  // 09 About, Privacy, Sources
  for (const r of ["about", "privacy", "sources"]) {
    await page.goto(`${BASE}/${r}`);
    await shot(page, size, `09-${r}`, { wait: 1200 });
  }

  // 10 return to the saved trip
  await page.goto(`${BASE}/coast/trips?id=${id}`);
  await page.getByTestId("trip-workspace").waitFor({ timeout: 20_000 });
  await shot(page, size, "10-saved-trip", { wait: 2000 });

  await ctx.close();
}

/** A short desk video: Today → statewide → chlorophyll → currents → zoom → the trip. */
async function video(browser) {
  const size = SIZES.find((s) => s.tag === "1440") ?? { tag: "1440", viewport: { width: 1440, height: 900 } };
  const dir = join(OUT, "video-tmp");
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const ctx = await browser.newContext({ viewport: size.viewport });
  await ctx.addInitScript((kv) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    for (const [k, v] of Object.entries(kv)) localStorage.setItem(k, v);
  }, RETURNING);
  // plan the trip off camera, then hand the storage to the recorded context
  const pre = await ctx.newPage();
  await pre.clock.setFixedTime(new Date(NOW));
  const id = await plan(pre);
  const state = await ctx.storageState();
  await ctx.close();
  const rec = await browser.newContext({ viewport: size.viewport, storageState: state, recordVideo: { dir, size: size.viewport } });
  const page = await rec.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${BASE}/today?place=${MOSS}`);
  await mapReady(page);
  await settle(page, 2000);
  const entry = await exploreEntry(page);
  if (entry) await entry.click();
  else await page.goto(`${BASE}/explore?place=${MOSS}`);
  await settle(page, 4000);
  for (const id of ["chl", "charm"]) {
    const l = await layer(page, id);
    if (l) {
      await l.click();
      await settle(page, 2500);
    }
  }
  const play = await first(page.getByTestId("nx-play"));
  if (play) {
    await play.click();
    await settle(page, 4000);
    await play.click().catch(() => {});
  }
  const port = await backToPort(page);
  if (port) {
    await port.click();
    await settle(page, 3000);
  }
  await page.goto(`${BASE}/coast/trips?id=${id}`);
  await page.getByTestId("trip-workspace").waitFor({ timeout: 20_000 }).catch(() => {});
  await settle(page, 2500);
  const v = page.video();
  await rec.close();
  const file = join(OUT, "walkthrough-v151.webm");
  renameSync(await v.path(), file);
  rmSync(dir, { recursive: true, force: true });
  shots.push(file);
  console.log("video", file);
}

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ channel: CHANNEL });
try {
  for (const size of SIZES) {
    try {
      await moments(browser, size);
    } catch (e) {
      errors.push(`[${size.tag}] script: ${e.message.split("\n")[0]}`);
      console.log("FAILED", size.tag, e.message);
    }
  }
  if (!process.env.CW_SHOTS_NO_VIDEO) await video(browser);
} finally {
  await browser.close();
}

mkdirSync(QA, { recursive: true });
for (const f of readdirSync(OUT)) {
  if ((COPY.has(f.split("-")[0]) && f.endsWith(".png")) || f.endsWith(".webm")) copyFileSync(join(OUT, f), join(QA, f));
}
writeFileSync(join(OUT, "report.json"), JSON.stringify({ base: BASE, shots: shots.length, missing, errors }, null, 2));
console.log(`\n${shots.length} files, ${missing.length} missing moments, ${errors.length} errors`);
for (const m of missing) console.log("  missing", m);
for (const e of errors) console.log("  error", e);
