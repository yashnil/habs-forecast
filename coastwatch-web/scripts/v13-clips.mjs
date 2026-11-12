#!/usr/bin/env node
/**
 * v13 interaction clips for review (docs/coastwatch/v13/implementation/media/): the six recordings the owner's
 * brief asks for, on the built app with fixture data (clock 2026-10-13), recorded with Playwright's recordVideo
 * and re-encoded small with ffmpeg when it is installed (VP9, 960 px wide for desktop).
 *   1 map-transitions         the chart panning and re-framing as the four pages change (also the page transitions)
 *   2 port-transition         the selected port changed from the bar: Moss Landing → Bodega Bay → back
 *   3 research-select-brush   a variable chosen on the rail, a sample opened, the window brushed
 *   4 economics-sensitivity   the owner's case, then the grid and "one at a time" moving the result
 *   5 explore-time-and-read   a cell read, the days played, the layer changed
 *   6 loading-offline         a cold load at Slow 4G (placeholders, then records), then the same page offline
 * Usage: npm run build, start the app on fixture data, then
 *   node scripts/v13-clips.mjs <baseUrl> [outDir]        (CLIPS=3,4 for a subset)
 */
import { execFileSync, spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:3410";
const OUT = resolve(process.argv[3] ?? "../docs/coastwatch/v13/implementation/media");
const ONLY = process.env.CLIPS?.split(",");
const REC = JSON.stringify({ v: 1, place: "port-592", for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const DESK = { width: 1440, height: 900 };
const PHONE = { width: 390, height: 844 };
mkdirSync(OUT, { recursive: true });
const tmp = mkdtempSync(join(tmpdir(), "neritic-v13-clips-"));
const ffmpeg = spawnSync("ffmpeg", ["-version"]).status === 0;
const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
const wait = (p, ms) => p.waitForTimeout(ms);
const failures = [];

async function clip(n, name, size, body, { seed = true } = {}) {
  if (ONLY && !ONLY.includes(String(n))) return;
  const dir = join(tmp, `${name}-${size.width}`);
  const phone = size.width < 720;
  const ctx = await browser.newContext({ viewport: size, isMobile: phone, hasTouch: phone, recordVideo: { dir, size } });
  if (seed)
    await ctx.addInitScript((rec) => {
      if (sessionStorage.getItem("seeded")) return;
      sessionStorage.setItem("seeded", "1");
      localStorage.setItem("nx-port-v1", rec);
      localStorage.setItem("nx-tour-v1", "dismissed");
      localStorage.setItem("nx-opening-seen-v1", "1");
      localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
    }, REC);
  const page = await ctx.newPage();
  try {
    await body(page, ctx);
  } catch (e) {
    failures.push(`${name}: ${String(e.message).split("\n")[0]}`);
    console.log(`FAILED ${name}: ${String(e.message).split("\n")[0]}`);
  }
  await ctx.close();
  const raw = join(dir, readdirSync(dir).find((f) => f.endsWith(".webm")));
  const out = join(OUT, `${n}-${name}-${size.width}.webm`);
  if (ffmpeg) execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", raw, "-vf", phone ? "scale=390:-2" : "scale=960:-2", "-c:v", "libvpx-vp9", "-crf", "42", "-b:v", "0", "-r", "20", "-an", out]);
  else copyFileSync(raw, out);
  console.log(`${n}-${name}-${size.width}.webm  ${(statSync(out).size / 1024).toFixed(0)} KB`);
}

const type = async (p, label, v, delay = 60) => {
  const f = p.getByRole("textbox", { name: label, exact: true });
  await f.scrollIntoViewIfNeeded();
  await f.click();
  await f.pressSequentially(v, { delay });
};
async function ownerCase(p) {
  for (const [l, v] of [["Catch", "400"], ["Price", "9"], ["Fuel", "30"], ["Fuel price", "8.24"], ["Bait and ice", "100"], ["Fees", "0"], ["Crew", "1"]]) await type(p, l, v);
  await p.getByRole("radio", { name: "After costs" }).click();
  await type(p, "Crew share", "20");
  await p.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
  await wait(p, 500);
  await p.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
}

for (const size of [DESK, PHONE]) {
  // 1 map transitions and page transitions: the four destinations and back
  await clip(1, "map-transitions", size, async (p) => {
    await p.goto(`${BASE}/today?place=port-592`);
    await wait(p, 3500);
    const nav = size === DESK ? p.getByRole("navigation", { name: "Destinations" }).first() : p.getByTestId("nx-tabs");
    for (const name of ["Explore", "My Coast", "Research", "Today"]) {
      await nav.getByRole("link", { name, exact: true }).click();
      await wait(p, 2600);
    }
    await p.goBack();
    await wait(p, 2200);
  });
  // 2 the selected port changes
  await clip(2, "port-transition", size, async (p) => {
    await p.goto(`${BASE}/today?place=port-592`);
    await wait(p, 3000);
    for (const q of ["bodega", "moss"]) {
      await p.getByTestId("nx-port-control").click();
      const box = p.getByRole("combobox").first();
      await box.pressSequentially(q, { delay: 90 });
      await wait(p, 500);
      await p.keyboard.press("Enter");
      await wait(p, 3500);
    }
  });
}

// 3 Research: a variable, a sample, the brush
await clip(3, "research-select-brush", DESK, async (p) => {
  await p.goto(`${BASE}/research?station=HABs-SantaCruzWharf&var=pDA`);
  await p.getByTestId("research-plot").waitFor();
  await wait(p, 2500);
  const plot = await p.getByTestId("research-plot").boundingBox();
  // read along the plot, then open a sample
  for (const fx of [0.3, 0.45, 0.6, 0.72]) {
    await p.mouse.move(plot.x + plot.width * fx, plot.y + plot.height * 0.45, { steps: 12 });
    await wait(p, 450);
  }
  await p.mouse.click(plot.x + plot.width * 0.72, plot.y + plot.height * 0.45);
  await wait(p, 1800);
  await p.getByTestId("research-plot").scrollIntoViewIfNeeded();
  // another variable on the rail
  await p.getByTestId("rail-temp").click();
  await wait(p, 2000);
  await p.getByTestId("rail-pDA").click();
  await wait(p, 1500);
  // the range: a preset, then the brush dragged
  await p.getByTestId("range-1y").click();
  await wait(p, 1500);
  const brush = p.getByTestId("research-brush");
  await brush.scrollIntoViewIfNeeded();
  const b = await brush.boundingBox();
  const frame = await brush.getByRole("slider").first().boundingBox();
  const y = (frame ?? b).y + (frame ?? b).height / 2;
  const x0 = frame ? frame.x + frame.width / 2 : b.x + b.width * 0.9;
  await p.mouse.move(x0, y);
  await p.mouse.down();
  await p.mouse.move(b.x + b.width * 0.55, y, { steps: 30 });
  await wait(p, 400);
  await p.mouse.move(b.x + b.width * 0.7, y, { steps: 20 });
  await p.mouse.up();
  await wait(p, 2200);
  await p.getByTestId("research-plot").scrollIntoViewIfNeeded();
  await wait(p, 1500);
});
await clip(3, "research-select-brush", PHONE, async (p) => {
  await p.goto(`${BASE}/research?station=HABs-SantaCruzWharf&var=pDA&range=1y`);
  await p.getByTestId("research-plot").waitFor();
  await wait(p, 2500);
  const plot = await p.getByTestId("research-plot").boundingBox();
  await p.touchscreen.tap(plot.x + plot.width * 0.6, plot.y + plot.height * 0.4);
  await wait(p, 2200);
  await p.getByRole("button", { name: "Next" }).click().catch(() => {});
  await wait(p, 1200);
  await p.getByRole("button", { name: "Next" }).click().catch(() => {});
  await wait(p, 1800);
});

// 4 economics: the owner's case typed, then what moves the result
await clip(4, "economics-sensitivity", DESK, async (p) => {
  await p.goto(`${BASE}/coast?place=port-592&view=trip`);
  await p.getByTestId("trip-inputs").waitFor();
  await wait(p, 2000);
  await ownerCase(p);
  await wait(p, 2000);
  await p.getByText("If things move").scrollIntoViewIfNeeded();
  await wait(p, 800);
  const cells = p.getByRole("group", { name: /catch and price move|If things move/i }).getByRole("button");
  const n = await cells.count();
  for (const i of [0, 2, n - 1, Math.floor(n / 2)]) {
    if (i < 0 || i >= n) continue;
    await cells.nth(i).hover();
    await wait(p, 1300);
  }
  await p.getByRole("radio", { name: "One at a time" }).click();
  await wait(p, 1500);
  const sliders = p.getByRole("slider");
  if (await sliders.count()) {
    await sliders.first().focus();
    for (let i = 0; i < 6; i++) {
      await p.keyboard.press("ArrowRight");
      await wait(p, 220);
    }
    for (let i = 0; i < 10; i++) {
      await p.keyboard.press("ArrowLeft");
      await wait(p, 220);
    }
  }
  await wait(p, 1500);
});
await clip(4, "economics-sensitivity", PHONE, async (p) => {
  await p.goto(`${BASE}/coast?place=port-592&view=trip`);
  await p.getByTestId("trip-inputs").waitFor();
  await wait(p, 1500);
  await ownerCase(p);
  await wait(p, 1500);
  await p.getByRole("link", { name: "Results" }).or(p.getByRole("button", { name: "Results" })).first().click();
  await wait(p, 2500);
  await p.getByText("If things move").scrollIntoViewIfNeeded();
  await wait(p, 2500);
});

// 5 Explore: a cell read, the days played, the layer changed
for (const size of [DESK, PHONE])
  await clip(5, "explore-time-and-read", size, async (p) => {
    await p.goto(`${BASE}/explore?place=port-592`);
    await wait(p, 5000);
    const map = await p.getByTestId("nx-map").boundingBox();
    await p.mouse.click(map.x + map.width * 0.55, map.y + map.height * (size === DESK ? 0.5 : 0.35));
    await wait(p, 2500);
    // On a phone the layer switch and Play rest below the sheet's visible part (review defect F-01): the clip
    // raises the sheet as a reader has to, and records what it can reach rather than failing.
    const reach = async (id) => {
      for (let i = 0; i < 3; i++) {
        const ok = await p.getByTestId(id).click({ timeout: 2500 }).then(() => true, () => false);
        if (ok) return true;
        await p.getByRole("button", { name: /Show (more|less) of this page/ }).first().click({ timeout: 1500 }).catch(() => {});
        await wait(p, 900);
      }
      console.log(`  (${size.width} px: ${id} could not be reached)`);
      return false;
    };
    await reach("nx-play");
    await wait(p, 6000);
    await reach("nx-layer-sst");
    await wait(p, 3000);
    await reach("nx-layer-charm");
    await wait(p, 3000);
  });

// 6 loading and offline: Slow 4G with the cache off, then no network at all
for (const size of [DESK, PHONE])
  await clip(6, "loading-offline", size, async (p, ctx) => {
    const cdp = await ctx.newCDPSession(p);
    await p.goto(`${BASE}/today?place=port-592`);
    await wait(p, 6000); // this device keeps its copy of Today
    await cdp.send("Network.enable");
    await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
    await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 562.5, downloadThroughput: 1474560 / 8, uploadThroughput: 675000 / 8 });
    await p.goto(`${BASE}/coast?place=port-592`).catch(() => {});
    await wait(p, 9000);
    await cdp.send("Network.emulateNetworkConditions", { offline: true, latency: 0, downloadThroughput: 0, uploadThroughput: 0 });
    await ctx.setOffline(true);
    await p.goto(`${BASE}/`).catch(() => {});
    await wait(p, 5000);
    await p.goto(`${BASE}/alerts?place=port-592`).catch(() => {});
    await wait(p, 3500);
  });

await browser.close();
rmSync(tmp, { recursive: true, force: true });
if (failures.length) {
  console.log(`\n${failures.length} clip(s) did not run to the end:\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
