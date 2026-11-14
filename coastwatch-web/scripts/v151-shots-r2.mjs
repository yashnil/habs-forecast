#!/usr/bin/env node
/**
 * Neritic v15.1 round 2 (DIRECTION-2 §F): Explore's new states, statewide and at the port, at 1440×900 and 390×844.
 * Run against the STAGING data channel (production lacks the fisheries-status sidecar), so WCOFS and the DINEOF estimate are read at request time (no CW_CONDITIONS_FIXTURE):
 *   CW_DATA_BASE_URL=https://raw.githubusercontent.com/yashnil/habs-forecast/coastwatch-data-staging/v1 npx next start -p 7591
 *   CW_SHOTS_HEADED=1 node scripts/v151-shots-r2.mjs http://localhost:7591 <outDir> [qaDir]    # outside a sandbox
 *
 *   r2a the chart with no chlorophyll (Chart)       r2b observed chlorophyll (one day, gaps kept)
 *   r2c the recent composite (pixel ages)          r2d the gap-filled estimate (DINEOF, labelled)
 *   r2e the latest C-HARM run (expired)            r2f WCOFS model sea temperature and predicted currents
 *   r2g the revised timeline (three families on true times)
 * The device clock is the real one: the moments show the live data as of the run. A control that is missing or
 * disabled is logged (MISSING / DISABLED) with the page's own words for it, and the frame is not faked.
 */
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:7591";
const OUT = resolve(process.argv[3] ?? "shots/v151-r2");
const QA = process.argv[4] ? resolve(process.argv[4]) : null;
const MOSS = "port-592";
const SIZES = [
  { tag: "1440", viewport: { width: 1440, height: 900 } },
  { tag: "390", viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
];
const VIEWS = [
  { tag: "coast", q: `view=coast&place=${MOSS}` },
  { tag: "port", q: `place=${MOSS}` },
];
const RETURNING = { "nx-opening-seen-v1": "1", "nx-tour-v1": "dismissed" };
const log = [];
const note = (s) => {
  log.push(s);
  console.log(s);
};
const settle = (p, ms) => p.waitForTimeout(ms);

async function shot(page, name, wait = 2500) {
  await settle(page, wait);
  await page.waitForFunction(() => !window.__cwMap || !window.__cwMap.isMoving(), null, { timeout: 15_000 }).catch(() => {});
  const file = join(OUT, `${name}.png`);
  await page.screenshot({ path: file });
  note(`shot ${name}`);
}

/** Click a radio/switch by test id when it is there and enabled; otherwise log why not. */
async function choose(page, testid, name) {
  const el = page.getByTestId(testid).first();
  if (!(await el.count())) {
    note(`MISSING ${name}: no ${testid}`);
    return false;
  }
  await el.scrollIntoViewIfNeeded().catch(() => {});
  if (await el.isDisabled()) {
    note(`DISABLED ${name}: ${testid} says "${(await el.innerText()).replace(/\s+/g, " ").trim()}"`);
    return false;
  }
  if ((await el.getAttribute("aria-checked")) !== "true") await el.click();
  return true;
}

async function open(page, view) {
  await page.goto(`${BASE}/explore?${view.q}`);
  await page.locator("[data-testid=nx-sheet][data-map-ready='1']").waitFor({ timeout: 30_000 }).catch(() => note(`map not ready on ${view.q}`));
  await page.getByTestId("nx-explore-panel").waitFor({ timeout: 30_000 });
  await settle(page, 1500);
}

async function moments(browser, size) {
  const ctx = await browser.newContext({ viewport: size.viewport, isMobile: size.isMobile, hasTouch: size.hasTouch });
  await ctx.addInitScript((kv) => Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, v)), RETURNING);
  const page = await ctx.newPage();
  page.on("pageerror", (e) => note(`[${size.tag}] pageerror ${e.message}`));
  for (const view of VIEWS) {
    const at = `${size.tag}-${view.tag}`;
    await open(page, view);
    if (await choose(page, "nx-layer-chart", `${at} r2a`)) await shot(page, `${at}-r2a-chart-only`);
    if (await choose(page, "nx-layer-chl", `${at} chlorophyll`)) {
      if (await choose(page, "nx-chl-observed", `${at} r2b`)) await shot(page, `${at}-r2b-observed`);
      if (await choose(page, "nx-chl-composite", `${at} r2c`)) await shot(page, `${at}-r2c-composite`);
      if (await choose(page, "nx-chl-estimate", `${at} r2d`)) await shot(page, `${at}-r2d-estimate`, 4000);
    }
    if (await choose(page, "nx-layer-charm", `${at} r2e`)) await shot(page, `${at}-r2e-charm-expired`);
    if (await choose(page, "nx-layer-sst-model", `${at} r2f`)) {
      if (await choose(page, "nx-currents-toggle", `${at} r2f currents`)) await choose(page, "nx-currents-predicted", `${at} r2f predicted`);
      await shot(page, `${at}-r2f-wcofs`, 4500);
    }
    // the timeline: open it if it is folded, then photograph it with the chlorophyll composite on
    await choose(page, "nx-layer-chl", `${at} r2g`);
    const toggle = page.getByTestId("nx-rail-toggle").first();
    if ((await toggle.count()) && /^Timeline$/.test((await toggle.innerText()).trim())) await toggle.click();
    const rail = page.getByTestId("nx-timerail").first();
    if (await rail.count()) {
      await rail.scrollIntoViewIfNeeded().catch(() => {});
      await shot(page, `${at}-r2g-timeline`);
    } else note(`MISSING ${at} r2g: no nx-timerail`);
    // what the page says about each live source, for the README
    const words = await page.evaluate(() =>
      ["nx-wcofs-state", "nx-charm-stalled", "nx-rail-axis-note", "nx-model-label"].map((id) => `${id}: ${document.querySelector(`[data-testid=${id}]`)?.textContent?.replace(/\s+/g, " ").trim() ?? "(absent)"}`),
    );
    for (const w of words) note(`  ${at} ${w}`);
  }
  await ctx.close();
}

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ channel: process.env.CW_SHOTS_CHANNEL ?? "chrome", headless: !process.env.CW_SHOTS_HEADED });
try {
  for (const size of SIZES) await moments(browser, size);
} finally {
  await browser.close();
}
writeFileSync(join(OUT, "report.txt"), log.join("\n") + "\n");
if (QA) {
  mkdirSync(QA, { recursive: true });
  for (const f of readdirSync(OUT)) if (f.endsWith(".png")) copyFileSync(join(OUT, f), join(QA, f));
}
console.log(`\n${log.filter((l) => l.startsWith("shot")).length} frames; ${log.filter((l) => /^(MISSING|DISABLED)/.test(l)).length} missing or disabled`);
