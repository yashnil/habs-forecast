// Capture map-lab comparison screenshots (1440 x 900) from real data.
// usage: node scripts/shoot_lab.mjs   (serves ../lab on :8767; Playwright from coastwatch-web)
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../lab");
const OUT = path.resolve(HERE, "../maps");
const require = createRequire(path.resolve(HERE, "../../../../coastwatch-web/package.json"));
const { chromium } = require("@playwright/test");
const TYPES = { ".html": "text/html", ".png": "image/png", ".json": "application/json", ".geojson": "application/geo+json" };
const server = createServer(async (req, res) => {
  const p = path.join(ROOT, decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  try { res.writeHead(200, { "content-type": TYPES[path.extname(p)] ?? "application/octet-stream" }); res.end(await readFile(p)); }
  catch { res.writeHead(404).end(); }
}).listen(8767);

const SHOTS = [
  ["01-monterey-charm-native", "region=monterey&layer=charm_native&lead=1"],
  ["02-monterey-charm-display", "region=monterey&layer=charm_display&lead=1"],
  ["03-monterey-olci300-0924", "region=monterey&layer=olci300&date=2026-09-24"],
  ["04-monterey-olci300-latest-1007", "region=monterey&layer=olci300&date=2026-10-07"],
  ["05-monterey-viirs750-0925", "region=monterey&layer=viirs750&date=2026-09-25"],
  ["06-monterey-viirs4km-0925", "region=monterey&layer=viirs_n20&date=2026-09-25"],
  ["07-monterey-dineof2km-0927", "region=monterey&layer=dineof&date=2026-09-27"],
  ["08-monterey-olci-currents-drift", "region=monterey&layer=olci300&date=2026-09-24&currents=1&drift=1"],
  ["09-socal-olci300-1006", "region=socal_bight&layer=olci300&date=2026-10-06"],
  ["10-socal-charm-native", "region=socal_bight&layer=charm_native&lead=1"],
  ["11-socal-viirs4km-1006", "region=socal_bight&layer=viirs_n20&date=2026-10-06"],
  ["12-north-olci300-1001", "region=north_coast&layer=olci300&date=2026-10-01"],
  ["13-north-charm-native", "region=north_coast&layer=charm_native&lead=1"],
  ["14-state-charm-display-currents", "region=state&layer=charm_display&lead=1&currents=1"],
];
const browser = await chromium.launch(process.env.CW_CHROME ? { executablePath: process.env.CW_CHROME } : {});
await mkdir(OUT, { recursive: true });
for (const [name, qs] of SHOTS) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", (e) => console.error(name, e.message));
  await page.goto(`http://localhost:8767/?${qs}`);
  await page.waitForSelector("body[data-ready]", { timeout: 45000 }).catch(() => console.error("not idle", name));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1800);
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
  console.log(name);
  await page.close();
}
await browser.close(); server.close();
