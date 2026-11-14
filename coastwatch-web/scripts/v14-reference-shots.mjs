#!/usr/bin/env node
/**
 * v14 reference stills: the approved synthesis PROTOTYPE (plain HTML/CSS/canvas) rendered at the owner's five
 * review sizes, for the five frames, with motion settled (`?screen=<frame>&shot=1` and reduced motion), device
 * scale 1, saved as WebP into docs/coastwatch/v14/implementation/reference/<frame>-<w>x<h>.webp.
 * These are what scripts/v14-app-shots.mjs pairs the app's stills with.
 *
 * Usage (from coastwatch-web):
 *   node scripts/v14-reference-shots.mjs [outDir]
 *   env: CW_V14_PROTOTYPE=<prototype folder>  CW_V14_PROTO_PORT=7790 (7700–7799)  FRAMES=today,alerts  SIZES=390x844
 * The prototype loads its Motion build and Figtree from public CDNs, so the run needs the network.
 */
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";
import { IMPL_DIR, PROTOTYPE_DIR, contextFor, imageTools, pickFrames, pickSizes, serveFolder, sizeName } from "./v14-lib.mjs";

const OUT = process.argv[2] ?? join(IMPL_DIR, "reference");
const PORT = Number(process.env.CW_V14_PROTO_PORT ?? 7790);
const QUALITY = Number(process.env.CW_V14_WEBP_Q ?? 0.86);
mkdirSync(OUT, { recursive: true });

statSync(join(PROTOTYPE_DIR, "index.html")); // fail early with a clear path
const server = await serveFolder(PROTOTYPE_DIR, PORT);
const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
const tools = await imageTools(browser);
const problems = [];
let total = 0;
try {
  for (const size of pickSizes()) {
    const ctx = await browser.newContext(contextFor(size, { reducedMotion: "reduce" }));
    const page = await ctx.newPage();
    page.on("pageerror", (e) => problems.push(`${sizeName(size)} pageerror ${e.message}`));
    page.on("console", (m) => m.type() === "error" && problems.push(`${sizeName(size)} console ${m.text().slice(0, 200)}`));
    for (const frame of pickFrames()) {
      let ok = false;
      for (let k = 0; k < 3 && !ok; k++) {
        try {
          await page.goto(`http://127.0.0.1:${PORT}/index.html?screen=${frame}&shot=1`, { waitUntil: "networkidle", timeout: 20_000 });
          await page.waitForFunction(() => document.documentElement.dataset.ready === "1", null, { timeout: 12_000 });
          ok = true;
        } catch (e) {
          console.log(`retry ${frame} ${sizeName(size)}: ${String(e.message).slice(0, 80)}`);
        }
      }
      if (!ok) {
        problems.push(`${frame} ${sizeName(size)}: never ready`);
        continue;
      }
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(1200);
      const hscroll = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      if (hscroll > 1) problems.push(`${frame} ${sizeName(size)}: prototype scrolls sideways by ${hscroll}px`);
      const webp = await tools.webp(await page.screenshot(), QUALITY);
      const file = join(OUT, `${frame}-${sizeName(size)}.webp`);
      writeFileSync(file, webp);
      total += webp.length;
      console.log(`${file}  ${(webp.length / 1024).toFixed(0)} kB`);
    }
    await ctx.close();
  }
} finally {
  await tools.close();
  await browser.close();
  server.close();
}
console.log(`total ${(total / 1024 / 1024).toFixed(2)} MB`);
if (problems.length) console.log(`notes:\n  ${problems.join("\n  ")}`);
