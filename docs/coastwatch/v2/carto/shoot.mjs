// Screenshots of every treatment at every zoom preset, desktop (1440x900) and phone (390x844).
//   node shoot.mjs                 all shots -> shots/<treatment>-<preset>-<w>.png
//   node shoot.mjs T3 harbor 1440  filters (any order; several of each kind allowed)
//   node shoot.mjs T2 bay classes=20   the 20-class experiment (suffix -fine20)
// Needs coastwatch-web's Playwright and an installed Chrome (channel "chrome"); basemap tiles,
// glyphs and the positron style come from OpenFreeMap (network). Perf lines logged by the page
// ("[carto] raster first paint …") are collected into shots/perf.json.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
import { startServer } from "./serve.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "../../../../coastwatch-web/package.json"));
const { chromium } = require("playwright");

const TREATMENTS = ["T0", "T1", "T2", "T3", "T4"];
const PRESETS = ["regional", "bay", "harbor", "crescent"];
const VIEWPORTS = { 1440: { width: 1440, height: 900 }, 390: { width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 2 } };
const args = process.argv.slice(2);
const pick = (all) => { const f = all.filter((k) => args.includes(k)); return f.length ? f : all; };
const treatments = pick(TREATMENTS), presets = pick(PRESETS), widths = pick(Object.keys(VIEWPORTS));
// "classes=20": the 5 %-class experiment; shots get a "-fine20" suffix
const extraQuery = args.filter((a) => a.includes("=")).map((a) => "&" + a).join("");
const suffix = args.includes("classes=20") ? "-fine20" : "";
const PORT = 3399 + Math.floor(Math.random() * 100);
const OUT = join(here, "shots");
mkdirSync(OUT, { recursive: true });

const server = await startServer(PORT);
const browser = await chromium.launch({ channel: "chrome" });
const perf = [];
try {
  for (const w of widths) {
    const context = await browser.newContext({ viewport: VIEWPORTS[w], deviceScaleFactor: VIEWPORTS[w].deviceScaleFactor ?? 1, isMobile: !!VIEWPORTS[w].isMobile, hasTouch: !!VIEWPORTS[w].hasTouch });
    const page = await context.newPage();
    const logs = [];
    page.on("console", (m) => { const t = m.text(); if (t.startsWith("[carto]")) logs.push(t); if (m.type() === "error") console.error("  page error:", t); });
    page.on("pageerror", (e) => console.error("  page exception:", e.message));
    for (const t of treatments) for (const p of presets) {
      const name = `${t}-${p}-${w}${suffix}`;
      const t0 = Date.now();
      logs.length = 0;
      await page.goto(`http://127.0.0.1:${PORT}/index.html?t=${t}&p=${p}${extraQuery}`, { waitUntil: "load" });
      try {
        await page.waitForFunction(() => window.__carto?.idle === true, null, { timeout: 60_000 });
      } catch (e) {
        console.error(`${name}: page never became idle (${e.message.split("\n")[0]})`);
        await page.screenshot({ path: join(OUT, `${name}-FAILED.png`) });
        continue;
      }
      // let the basemap's labels settle (fade is 0, but glyphs may still arrive)
      await page.evaluate(() => new Promise((res) => { const m = window.__carto.state.map; if (m.loaded() && !m.isMoving()) { m.once("idle", res); m.triggerRepaint(); setTimeout(res, 1500); } else m.once("idle", res); }));
      await page.waitForTimeout(400);
      await page.screenshot({ path: join(OUT, `${name}.png`) });
      const p2 = await page.evaluate(() => window.__carto.perf);
      perf.push({ shot: name, ...p2, log: logs.slice() });
      console.log(`${name}  ${Date.now() - t0} ms  ${logs[0] ?? ""}`);
    }
    await context.close();
  }
} finally {
  await browser.close();
  server.close();
}
writeFileSync(join(OUT, `perf${suffix}.json`), JSON.stringify(perf, null, 1));
console.log(`wrote ${perf.length} shots to ${OUT}`);
