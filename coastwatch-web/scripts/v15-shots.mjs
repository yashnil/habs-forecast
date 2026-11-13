#!/usr/bin/env node
/**
 * Neritic v15 screenshots and a walkthrough video, from the running app on fixture data:
 *   npm run data:fixture && npm run build
 *   CW_DATA_BASE_URL=/data/fixture/v1 CW_CONDITIONS_FIXTURE=tests/conditions-fixtures.json CW_NWS_API=1 \
 *     CW_FIXED_NOW=2026-10-13T06:00:00Z npx next start -p 6590
 *   node scripts/v15-shots.mjs [baseUrl] [outDir] [qaDir]
 *
 * For each size (1440×900, 390×844, 320×640) it plans the owner's trip on My Coast (Moss Landing, commercial
 * Dungeness, a 12 kt / 6 ft limit, the regression worksheet), then photographs: 1 the planned trip's workspace,
 * 2 two departure periods compared, 3 the change feed after a recheck that finds a stronger NWS wind for the departure
 * day (the sea answer is changed in the browser, as the acceptance suite does), 4 the departure briefing opened with
 * no network after a restart of the browser profile, 5 planned against actual after recording an outcome, 6 the
 * history. A desk-size walkthrough video of the same journey is recorded (Playwright recordVideo).
 * The selected frames (desk and 390 px) are copied to docs/coastwatch/v15/qa/ at the repository root.
 */
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:6590";
const OUT = resolve(process.argv[3] ?? "shots/v15");
const QA = resolve(process.argv[4] ?? "../docs/coastwatch/v15/qa");
const NOW = "2026-10-13T06:00:00Z";
const MOSS = "port-592";
const CHANNEL = process.env.CW_SHOTS_CHANNEL ?? "chrome";
const SIZES = [
  { tag: "1440", viewport: { width: 1440, height: 900 } },
  { tag: "390", viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
  { tag: "320", viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true },
];
const RETURNING = {
  "nx-opening-seen-v1": "1",
  "nx-tour-v1": "dismissed",
  "nx-port-v1": JSON.stringify({ v: 1, place: MOSS, for: "all", species: "any", lastPort: MOSS, fisheries: ["dungeness_commercial"], vessel: null }),
  "nx-coast-v1": JSON.stringify({ fishery: "dungeness_commercial" }),
};
const settle = (p, ms = 1200) => p.waitForTimeout(ms);
const shots = [];

async function shot(page, size, name, o = {}) {
  await settle(page, o.wait ?? 1200);
  const file = join(OUT, `${size.tag}-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!o.full });
  shots.push(file);
  console.log("shot", file);
}

async function field(page, label, v) {
  const f = page.getByRole("textbox", { name: label, exact: true }).first();
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
}

/** The owner's planning journey (tests/e2e/v15/helpers.ts planTrip). */
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
  for (const [l, v] of [
    ["Catch", "400"],
    ["Price", "9"],
    ["Fuel", "30"],
    ["Fuel price", "8.24"],
    ["Bait and ice", "100"],
    ["Fees", "0"],
    ["Crew", "1"],
  ])
    await field(page, l, v);
  await page.getByRole("radio", { name: "After costs" }).click();
  await field(page, "Crew share", "20");
  await page.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await page.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
  await page.getByTestId("trip-plan-save").click();
  await page.waitForURL(/\/coast\/trips\?id=/);
  await page.getByTestId("trip-workspace").waitFor();
  return new URL(page.url()).searchParams.get("id");
}

/** NWS raises the departure day's wind (the acceptance suite's Test B change). Routed on the context, which also
 *  catches the service worker's own requests (the trips page registers it). */
async function strongerWind(ctx) {
  await ctx.route("**/coast/data?*part=sea*", async (route) => {
    const res = await route.fetch();
    const j = await res.json();
    const lane = j.sea?.lanes?.find((l) => l.name === "Wed");
    if (lane) lane.text = "NE wind 20 to 25 kt, backing to NW in the afternoon. Seas around 3 ft. Wave Detail: NW 3 ft at 7 seconds and SW 2 ft at 15 seconds.";
    await route.fulfill({ response: res, json: j });
  });
}

async function journey(browser, size, o = {}) {
  const profile = mkdtempSync(join(tmpdir(), "nx-v15-shots-"));
  const opts = { channel: CHANNEL, viewport: size.viewport, isMobile: size.isMobile, hasTouch: size.hasTouch, serviceWorkers: "allow" };
  let ctx = await chromium.launchPersistentContext(profile, { ...opts, ...(o.video ? { recordVideo: { dir: o.video, size: size.viewport } } : {}) });
  let page = ctx.pages()[0] ?? (await ctx.newPage());
  page.on("pageerror", (e) => console.log(`[${size.tag}] pageerror`, e.message));
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${BASE}/about`);
  await page.evaluate((kv) => Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, v)), RETURNING);

  const id = await plan(page);
  await shot(page, size, "1-planned", { wait: 2500 });

  // two departure periods compared: Tuesday and the departure day
  const tue = page.getByTestId("window-2026-10-13|day");
  if (await tue.count()) {
    await tue.scrollIntoViewIfNeeded();
    await tue.click();
    await page.getByTestId("window-2026-10-14|day").click();
    await page.getByTestId("window-compare").first().scrollIntoViewIfNeeded();
    await shot(page, size, "2-departure-comparison");
  }

  // the change feed: photographed in its own browser context (changeFeed below)
  await page.goto(`${BASE}/coast/trips?id=${id}`);
  await page.getByTestId("trip-workspace").waitFor();

  // the departure briefing, offline after a browser restart
  await page.waitForFunction(async () => (await caches.has("neritic-trips-v1")) && !!navigator.serviceWorker.controller, null, { timeout: 30_000 }).catch(() => console.log("trips page not kept yet"));
  await ctx.close();
  ctx = await chromium.launchPersistentContext(profile, { ...opts, offline: true, ...(o.video ? { recordVideo: { dir: o.video, size: size.viewport } } : {}) });
  page = ctx.pages()[0] ?? (await ctx.newPage());
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${BASE}/coast/trips?id=${id}&view=briefing`);
  await page.getByTestId("trip-briefing").waitFor({ timeout: 20_000 });
  await shot(page, size, "4-offline-briefing", { full: size.tag === "1440" });

  // record the outcome (still offline): planned against actual
  await page.goto(`${BASE}/coast/trips?id=${id}`);
  await page.getByTestId("trip-workspace").waitFor();
  // behind More on a wide screen; in line on a phone
  if (await page.getByTestId("trip-more").isVisible().catch(() => false)) await page.getByTestId("trip-more").click();
  await page.getByTestId("trip-complete").first().scrollIntoViewIfNeeded();
  await page.getByTestId("trip-complete").first().click();
  const form = page.getByTestId("outcome-form");
  await form.waitFor();
  for (const [l, v] of [
    ["Species", "Dungeness crab"],
    ["Weight", "450"],
    ["Price received", "8.50"],
    ["Fuel used", "35"],
    ["Fuel price", "8.00"],
    ["What", "Bait and ice"],
    ["Paid", "120"],
    ["Crew paid, in total", "700"],
  ])
    await form.getByLabel(l, { exact: true }).first().fill(v);
  await form.getByTestId("outcome-save").click();
  await page.getByTestId("planned-vs-actual").waitFor();
  await page.getByTestId("planned-vs-actual").scrollIntoViewIfNeeded();
  await shot(page, size, "5-actual-vs-planned", { wait: 1800 });

  // the history
  await page.goto(`${BASE}/coast/trips`);
  await page.getByTestId("history").waitFor();
  // the page may show its reading placeholder again while it settles: photograph the list itself
  await page.waitForFunction(() => !document.body.innerText.includes("Reading the trips kept") && !!document.querySelector('[data-testid="history-row"]'), null, { timeout: 20_000 }).catch(() => console.log(`[${size.tag}] the history did not settle`));
  await shot(page, size, "6-history", { wait: 1500 });
  await ctx.close();
  rmSync(profile, { recursive: true, force: true });
}

/**
 * The change feed after a recheck that finds the stronger wind, in an ordinary browser context (the recheck's motion
 * settles there as in the acceptance suite; a persistent profile is only needed for the offline restart).
 */
async function changeFeed(size) {
  const b = await chromium.launch({ channel: CHANNEL });
  const ctx = await b.newContext({ viewport: size.viewport, isMobile: size.isMobile, hasTouch: size.hasTouch });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${BASE}/about`);
  await page.evaluate((kv) => Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, v)), RETURNING);
  const id = await plan(page);
  await strongerWind(ctx);
  await page.goto(`${BASE}/coast/trips?id=${id}`);
  const feed = page.getByTestId("trip-change-feed");
  await feed.waitFor({ timeout: 20_000 });
  // the feed's own control, or the primary when the recheck is the next action (one recheck control at a time)
  const own = feed.getByTestId("trip-recheck");
  await ((await own.count()) ? own : page.getByTestId("trip-next")).click();
  await page.waitForFunction(() => /change|differ|incomplete|could not/i.test(document.querySelector('[data-testid="trip-change-feed"] h2')?.textContent ?? ""), null, { timeout: 20_000 });
  await page.getByTestId("trip-since").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => {
    const h = document.querySelector('[data-testid="trip-change-feed"] h2');
    return !!h && getComputedStyle(h).opacity === "1";
  }, null, { timeout: 10_000 }).catch(() => console.log(`[${size.tag}] the feed's headline did not settle to full opacity`));
  await shot(page, size, "3-change-feed", { wait: 2500 });
  await b.close();
}

mkdirSync(OUT, { recursive: true });
const browser = null;
for (const size of SIZES) {
  await journey(browser, size);
  await changeFeed(size);
}
// the walkthrough: the desk journey again, recorded
const vdir = join(OUT, "video");
rmSync(vdir, { recursive: true, force: true });
mkdirSync(vdir, { recursive: true });
await journey(browser, SIZES[0], { video: vdir });
const videos = readdirSync(vdir).filter((f) => f.endsWith(".webm")).sort();
videos.forEach((f, i) => renameSync(join(vdir, f), join(vdir, `walkthrough-${i + 1}${i === 0 ? "-online" : "-offline"}.webm`)));

// the selected frames for review, beside the QA notes
mkdirSync(QA, { recursive: true });
for (const f of shots) if (/\/(1440|390)-/.test(f)) copyFileSync(f, join(QA, f.split("/").pop()));
for (const f of readdirSync(vdir)) copyFileSync(join(vdir, f), join(QA, f));
console.log(`done: ${shots.length} shots in ${OUT}; selected copied to ${QA}`);
