#!/usr/bin/env node
/**
 * Motion clips for review (docs/coastwatch/v12/implementation/motion/): the built app on fixture data, recorded
 * with Playwright's recordVideo at 1440 × 900 and 390 × 844. Each clip is one journey's motion: the first-visit
 * opening into Home and a port into Today; the route pans across the four pages and back; a worksheet scenario
 * previewed and returned; an Explore frame cut with its readout; a Research brush. Plus the route clip under
 * reduced motion (final states, no pans).
 * Usage: npm run build && node scripts/motion-clips.mjs [port] [outDir]
 */
import { spawn } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const PORT = Number(process.argv[2] ?? 3480);
const OUT = resolve(process.argv[3] ?? "../docs/coastwatch/v12/implementation/motion");
const BASE = `http://localhost:${PORT}`;
const REC = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const SIZES = { 1440: { width: 1440, height: 900 }, 390: { width: 390, height: 844 } };

const server = spawn("npx", ["next", "start", "-p", String(PORT)], {
  env: { ...process.env, CW_DATA_BASE_URL: "/data/fixture/v1", CW_CONDITIONS_FIXTURE: "tests/conditions-fixtures.json", CW_NWS_API: "1", CW_FIXED_NOW: "2026-10-13T06:00:00Z" },
  stdio: "ignore",
  detached: true,
});
for (let i = 0; i < 120; i++) {
  try {
    if ((await fetch(`${BASE}/about`)).ok) break;
  } catch {
    // not yet
  }
  await new Promise((r) => setTimeout(r, 500));
}
mkdirSync(OUT, { recursive: true });
const tmp = mkdtempSync(join(tmpdir(), "neritic-motion-"));
const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });

async function clip(name, w, { seed = true, reduced = false } = {}, body) {
  const size = SIZES[w];
  const phone = w < 700;
  const ctx = await browser.newContext({ viewport: size, isMobile: phone, hasTouch: phone, reducedMotion: reduced ? "reduce" : "no-preference", recordVideo: { dir: join(tmp, name), size } });
  if (seed)
    await ctx.addInitScript((r) => {
      if (sessionStorage.getItem("__seeded")) return;
      sessionStorage.setItem("__seeded", "1");
      localStorage.setItem("nx-port-v1", r);
      localStorage.setItem("nx-tour-v1", "dismissed");
      localStorage.setItem("nx-opening-seen-v1", "1");
    }, REC);
  const page = await ctx.newPage();
  page.setDefaultTimeout(30_000);
  try {
    await body(page, phone);
  } catch (e) {
    console.log(`  ${name}: ${e.message.split("\n")[0]}`);
  }
  const video = page.video();
  await ctx.close();
  if (video) copyFileSync(await video.path(), join(OUT, `${name}.webm`));
  console.log(`clip ${name}.webm`);
}

const mapReady = (page) => page.waitForFunction(() => document.querySelector('[data-testid="nx-sheet"]')?.getAttribute("data-map-ready") === "1", null, { timeout: 30_000 });
const tabs = (page, phone) => (phone ? page.getByTestId("nx-tabs") : page.getByRole("navigation", { name: "Destinations" }).first());

for (const w of [1440, 390]) {
  // 1. the first visit: the opening, Home, Find your port, a fishery, Today assembling
  await clip(`opening-to-today-${w}`, w, { seed: false }, async (page) => {
    await page.goto(`${BASE}/`);
    await page.waitForFunction(() => !document.documentElement.getAttribute("data-nx-opening"), null, { timeout: 8000 }).catch(() => null);
    await page.waitForTimeout(700);
    await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
    const finder = page.getByTestId("nx-home-finder");
    await finder.getByRole("combobox").pressSequentially("moss", { delay: 110 });
    await page.waitForTimeout(400);
    await finder.getByRole("option", { name: /Moss Landing/ }).click();
    await page.waitForTimeout(1200);
    await page.getByTestId("nx-fishery-chooser").getByRole("button", { name: "Dungeness crab (commercial)" }).click();
    await page.locator("[data-nx-sounding=buoy]").waitFor({ timeout: 10_000 }).catch(() => null);
    await page.waitForTimeout(2600);
  });

  // 2. the route: the chart pans and the margin swaps, on tabs and on back
  for (const reduced of [false, true]) {
    if (reduced && w !== 1440) continue;
    await clip(`route-${reduced ? "reduced-" : ""}${w}`, w, { reduced }, async (page, phone) => {
      await page.goto(`${BASE}/today?place=port-592`);
      await mapReady(page);
      await page.waitForTimeout(1500);
      for (const t of ["Explore", "My Coast", "Research"]) {
        await tabs(page, phone).getByRole("link", { name: t }).click();
        await page.waitForTimeout(1700);
      }
      for (let i = 0; i < 3; i++) {
        await page.goBack();
        await page.waitForTimeout(1700);
      }
    });
  }

  // 3. My Coast: the worksheet from the reader's numbers, a scenario previewed in the waterfall, then back
  await clip(`worksheet-scenario-${w}`, w, {}, async (page, phone) => {
    await page.goto(`${BASE}/coast?place=port-592`);
    await page.getByTestId("coast-worksheet").waitFor();
    await page.waitForTimeout(1200);
    if (phone) for (let i = 0; i < 2; i++) await page.getByRole("button", { name: "Show more of this page" }).tap();
    await page.getByLabel("Catch", { exact: true }).scrollIntoViewIfNeeded();
    const type = async (label, v) => {
      const f = page.getByLabel(label, { exact: true });
      await f.click();
      await f.pressSequentially(v, { delay: 60 });
    };
    await type("Catch", "800");
    await type("Price", "4.75");
    await type("Fuel", "120");
    await type("Fuel price", "4.60");
    await type("Bait and ice", "180");
    await type("Fees", "5");
    await type("Crew", "2");
    await page.getByRole("radio", { name: "Share after costs" }).click();
    await type("Crew share", "30");
    await page.getByRole("radio", { name: "Shared, off the top" }).click();
    await page.waitForTimeout(900);
    const up = page.getByRole("button", { name: /^\+30 % catch, \+20 % price/ });
    const down = page.getByRole("button", { name: /^−30 % catch, −20 % price/ });
    await up.scrollIntoViewIfNeeded();
    await page.waitForTimeout(600);
    if (phone) {
      await up.tap();
      await page.waitForTimeout(1600);
      await up.tap();
      await page.waitForTimeout(1200);
      await down.tap();
      await page.waitForTimeout(1600);
      await down.tap();
    } else {
      await up.hover();
      await page.waitForTimeout(1600);
      await down.hover();
      await page.waitForTimeout(1600);
      await page.getByRole("heading", { name: "Worksheet" }).hover();
    }
    await page.waitForTimeout(1600);
  });

  // 4. Explore: a frame is a cut; tapping the water registers a readout whose figure settles on the next frame
  await clip(`explore-frame-readout-${w}`, w, {}, async (page, phone) => {
    await page.goto(`${BASE}/explore?place=port-592`);
    await mapReady(page);
    await page.waitForTimeout(2200);
    const tap = async (lat, lon) => {
      const p = await page.evaluate(([la, lo]) => window.__cwMap.project([lo, la]), [lat, lon]);
      const box = await page.getByTestId("nx-map").boundingBox();
      await page.mouse.click(box.x + p.x, box.y + p.y);
    };
    await tap(36.87, -122.02);
    await page.getByTestId("nx-readout").getByText("Chlorophyll here").waitFor();
    await page.waitForTimeout(1800);
    const frame = page.getByTestId("nx-frame-olci:2026-10-05");
    if (phone) await frame.scrollIntoViewIfNeeded();
    await frame.click();
    await page.waitForTimeout(2000);
    await page.getByTestId("nx-frame-olci:latest").click();
    await page.waitForTimeout(2000);
  });

  // 5. Research: the crosshair, then the brush moved by key and by hand; every strip follows
  await clip(`research-brush-${w}`, w, {}, async (page, phone) => {
    await page.goto(`${BASE}/research?station=HABs-MontereyWharf&from=2024-01-01&to=2025-12-31&var=pn_seriata`);
    await page.getByTestId("research-title").waitFor();
    await page.waitForTimeout(2400);
    if (!phone) {
      const plot = page.getByTestId("research-strips").locator("[role=group]");
      const box = await plot.boundingBox();
      await page.mouse.move(box.x + box.width * 0.3, box.y + 60);
      await page.mouse.move(box.x + box.width * 0.6, box.y + 60, { steps: 20 });
      await page.waitForTimeout(600);
    }
    const end = page.getByRole("slider", { name: "Window end" });
    await end.scrollIntoViewIfNeeded();
    await end.focus();
    await end.press("Shift+ArrowLeft");
    await page.waitForTimeout(1400);
    const start = page.getByRole("slider", { name: "Window start" });
    const sb = await start.boundingBox();
    await page.mouse.move(sb.x + sb.width / 2, sb.y + sb.height / 2);
    await page.mouse.down();
    await page.mouse.move(sb.x + sb.width / 2 + 60, sb.y + sb.height / 2, { steps: 16 });
    await page.mouse.up();
    await page.waitForTimeout(1600);
  });
}

await browser.close();
try {
  process.kill(-server.pid, "SIGTERM");
} catch {
  // already stopped
}
rmSync(tmp, { recursive: true, force: true });
