// Capture prototype screenshots at the three review viewports.
// usage: node scripts/shoot.mjs [filter]   (serves ../prototype on :8765 itself)
// Playwright is resolved from coastwatch-web/node_modules; nothing is installed.
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../prototype");
const OUT = path.resolve(HERE, "../screenshots");
const require = createRequire(path.resolve(HERE, "../../../../coastwatch-web/package.json"));
const { chromium } = require("@playwright/test");

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".json": "application/json", ".svg": "image/svg+xml" };
const server = createServer(async (req, res) => {
  const p = path.join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  try {
    res.writeHead(200, { "content-type": TYPES[path.extname(p)] ?? "application/octet-stream" });
    res.end(await readFile(p));
  } catch {
    res.writeHead(404).end();
  }
}).listen(8765);

const VIEWPORTS = {
  "desktop-1440": { width: 1440, height: 900 },
  "laptop-1280": { width: 1280, height: 720 },
  "mobile-390": { width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
};

// name, url, viewports, optional action, fullPage
const SHOTS = [
  ["01-map-statewide", "map.html", ["desktop-1440", "laptop-1280", "mobile-390"]],
  ["02-map-port-santa-cruz", "map.html?region=monterey_bay&port=593", ["desktop-1440", "laptop-1280", "mobile-390"]],
  ["03-map-official-drawer", "map.html?region=monterey_bay", ["desktop-1440", "mobile-390"], async (p) => { await p.locator(".official-pill").click(); await p.waitForTimeout(200); }],
  ["04-bloom-santa-cruz", "bloom.html", ["desktop-1440", "laptop-1280", "mobile-390"], null, true],
  ["05-bloom-monterey-wharf", "bloom.html?station=HABs-MontereyWharf", ["desktop-1440", "mobile-390"], null, true],
  ["06-bloom-trinidad-historical", "bloom.html?station=HABs-TrinidadPier", ["desktop-1440"], null, false],
  ["07-fisheries", "fisheries.html", ["desktop-1440", "laptop-1280", "mobile-390"], null, true],
  ["08-fisheries-species-selected", "fisheries.html?group=dungeness_crab", ["desktop-1440", "mobile-390"], async (p) => { await p.locator("#breakdown").scrollIntoViewIfNeeded(); }, false],
];

const filter = process.argv[2];
// CW_CHROME overrides the browser when Playwright's pinned build is not installed.
const browser = await chromium.launch(process.env.CW_CHROME ? { executablePath: process.env.CW_CHROME } : {});
for (const [name, url, vps, action, full] of SHOTS) {
  if (filter && !name.includes(filter)) continue;
  for (const vp of vps) {
    const { isMobile, hasTouch, deviceScaleFactor, ...viewport } = VIEWPORTS[vp];
    const ctx = await browser.newContext({ viewport, isMobile, hasTouch, deviceScaleFactor: deviceScaleFactor ?? 1 });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => console.error(`[${name} ${vp}]`, e.message));
    await page.goto(`http://localhost:8765/${url}`);
    await page.evaluate(() => document.fonts.ready);
    if (url.startsWith("map")) await page.waitForSelector("body[data-ready]", { timeout: 30000 }).catch(() => console.error("map not idle", name));
    await page.waitForTimeout(url.startsWith("map") ? 1500 : 400);
    if (action) await action(page);
    // Full-page captures: pin the mobile tab bar to the end of the page instead of mid-image.
    if (full) await page.addStyleTag({ content: "body{position:relative}.tabbar{position:absolute!important}" });
    const file = path.join(OUT, vp, `${name}.png`);
    await page.screenshot({ path: file, fullPage: !!full });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    console.log(`${vp}/${name}.png${overflow > 1 ? `  OVERFLOW ${overflow}px` : ""}`);
    await ctx.close();
  }
}
await browser.close();
server.close();
