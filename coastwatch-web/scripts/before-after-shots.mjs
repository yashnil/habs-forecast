#!/usr/bin/env node
/**
 * v12 stills matched to v11's docs/coastwatch/v11/after/ (same pages, same sizes: 1440 × 900 at 1x, 390 × 844 at
 * 2x), plus 320 × 640 and 844 × 390 for v12. v11's default was dark, so its plain shots pair with v12 in Dusk and
 * its "-day" shots with v12 in Day (v12's default). Fixture data, clock 2026-10-13.
 * Usage: npm run build, start the app on fixture data, then
 *   node scripts/before-after-shots.mjs <baseUrl> <outDir>
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:3410";
const OUT = process.argv[3] ?? "../docs/coastwatch/v12/implementation/before-after";
mkdirSync(OUT, { recursive: true });
const REC = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });

const DESK = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 };
const PHONE = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true };
const SMALL = { viewport: { width: 320, height: 640 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true };
const LAND = { viewport: { width: 844, height: 390 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true };

// [file, path, device, mode, returning reader (port saved)]
const SHOTS = [
  ["today-desktop", "/today?place=port-592", DESK, "dusk", true],
  ["today-desktop-day", "/today?place=port-592", DESK, "day", true],
  ["today-phone", "/today?place=port-592", PHONE, "dusk", true],
  ["today-phone-day", "/today?place=port-592", PHONE, "day", true],
  ["explore-desktop", "/explore", DESK, "dusk", false],
  ["explore-phone", "/explore", PHONE, "dusk", false],
  ["explore-place-desktop", "/explore?place=port-592", DESK, "dusk", true],
  ["explore-place-phone", "/explore?place=port-592", PHONE, "dusk", true],
  ["coast-desktop", "/coast", DESK, "dusk", false],
  ["coast-phone", "/coast", PHONE, "dusk", false],
  ["coast-brief-desktop", "/coast?place=port-592", DESK, "dusk", true],
  ["coast-brief-desktop-day", "/coast?place=port-592", DESK, "day", true],
  ["coast-brief-phone", "/coast?place=port-592", PHONE, "dusk", true],
  ["coast-brief-phone-day", "/coast?place=port-592", PHONE, "day", true],
  ["research-desktop", "/research?station=HABs-SantaCruzWharf", DESK, "dusk", true],
  ["research-phone", "/research?station=HABs-SantaCruzWharf", PHONE, "dusk", true],
  // v12's default mode for the pages v11 only showed dark
  ["explore-place-desktop-day", "/explore?place=port-592", DESK, "day", true],
  ["research-desktop-day", "/research?station=HABs-SantaCruzWharf", DESK, "day", true],
  // v12 only: the smallest phone and a phone on its side
  ...[
    ["home", "/", false],
    ["today", "/today?place=port-592", true],
    ["explore", "/explore?place=port-592", true],
    ["coast", "/coast?place=port-592", true],
    ["research", "/research?place=port-592", true],
    ["notices", "/notices?place=port-592", true],
  ].flatMap(([n, p, seed]) => [
    [`${n}-320x640`, p, SMALL, "day", seed],
    [`${n}-844x390`, p, LAND, "day", seed],
  ]),
];

const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
for (const [name, path, device, mode, seed] of SHOTS) {
  const ctx = await browser.newContext(device);
  await ctx.addInitScript(
    ([r, m, s]) => {
      localStorage.setItem("nx-chart-mode-v1", m);
      localStorage.setItem("nx-tour-v1", "dismissed");
      localStorage.setItem("nx-opening-seen-v1", "1");
      if (s) localStorage.setItem("nx-port-v1", r);
    },
    [REC, mode, seed],
  );
  const page = await ctx.newPage();
  await page.goto(`${BASE}${path}`, { waitUntil: "load" });
  await page.waitForFunction(() => document.querySelector('[data-testid="nx-sheet"]')?.getAttribute("data-map-ready") === "1", null, { timeout: 20_000 }).catch(() => null);
  await page.waitForTimeout(3500);
  await page.screenshot({ path: join(OUT, `${name}.png`) });
  console.log(name);
  await ctx.close();
}
await browser.close();
