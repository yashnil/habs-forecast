// Capture review screenshots of the built app at the design-reset viewports.
// usage: npm run build && node scripts/shoot.mjs <out-dir> [data-base-url]
// The data base defaults to the production GitHub Pages dataset, so screenshots show real
// published data. Starts `next start` itself on :3300.
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const OUT = path.resolve(process.argv[2] ?? "screenshots");
const BASE = process.argv[3] ?? "https://yashnil.github.io/habs-forecast/v1";
const PORT = 3300;
const VIEWPORTS = {
  "desktop-1440": { viewport: { width: 1440, height: 900 } },
  "laptop-1280": { viewport: { width: 1280, height: 720 } },
  "mobile-390": { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
};
const drawer = async (p) => {
  await p.getByTestId(p.viewportSize().width < 768 ? "tab-notices" : "official-pill").click();
  await p.getByRole("dialog").waitFor();
};
// name, path, viewports, full page, action
const ALL = ["desktop-1440", "laptop-1280", "mobile-390"];
const SHOTS = [
  ["01-map-forecast", "/", ALL, false],
  ["02-map-satellite", "/?layer=olci300", ALL, false],
  ["03-map-port", "/?region=monterey_bay&port=593", ALL, false],
  ["04-map-satellite-age", "/?layer=olci300&age=1", ["desktop-1440", "mobile-390"], false],
  ["05-map-statewide-satellite", "/?region=california&layer=olci300", ["desktop-1440", "laptop-1280"], false],
  ["06-map-places", "/", ["mobile-390"], false, async (p) => { await p.getByTestId("place-button").click(); await p.waitForTimeout(300); }],
  ["10-map-satellite-clouded-day", "/?layer=olci300", ["desktop-1440"], false, async (p) => {
    const days = p.locator("[data-testid^=sat-day-2]");
    const n = await days.count();
    // the newest day whose coverage bar is empty (fully clouded or no overpass), if any
    for (let i = n - 1; i >= 0; i--) {
      const t = (await days.nth(i).getAttribute("title")) ?? "";
      if (/: 0% of/.test(t)) { await days.nth(i).click(); break; }
    }
    await p.waitForTimeout(800);
  }],
  // P2: multi-sensor satellite view and observed currents
  ["11-map-multisensor", "/?layer=multi", ALL, false],
  ["12-map-multisensor-sensor", "/?layer=multi&sensor=1", ["desktop-1440", "mobile-390"], false],
  ["13-map-statewide-multisensor", "/?region=california&layer=multi", ["desktop-1440", "laptop-1280"], false],
  ["14-map-currents", "/?layer=currents", ALL, false],
  ["15-map-currents-mean", "/?layer=currents:mean", ["desktop-1440", "mobile-390"], false],
  ["16-map-currents-flow", "/?layer=currents&flow=particles", ["desktop-1440", "laptop-1280"], false, async (p) => { await p.waitForTimeout(2500); }],
  ["17-map-currents-statewide", "/?region=california&layer=currents", ["desktop-1440", "laptop-1280"], false],
  ["18-map-currents-point", "/?region=monterey_bay&port=593&layer=currents", ALL, false],
  ["19-map-currents-north-coast-gap", "/?region=north_coast&layer=currents", ["desktop-1440", "mobile-390"], false],
  ["07-bloom", "/bloom", ["desktop-1440", "mobile-390"], true],
  ["08-fisheries", "/fisheries", ["desktop-1440", "mobile-390"], true],
  ["09-official-drawer", "/", ["desktop-1440", "mobile-390"], false, drawer],
];

const server = spawn("npx", ["next", "start", "-p", String(PORT)], { env: { ...process.env, CW_DATA_BASE_URL: BASE }, stdio: "ignore" });
for (let i = 0; ; i++) {
  try {
    if ((await fetch(`http://localhost:${PORT}/sources`)).ok) break;
  } catch {}
  if (i > 60) throw new Error("server did not start");
  await new Promise((r) => setTimeout(r, 500));
}
const browser = await chromium.launch(process.env.CI ? {} : { channel: "chrome" });
try {
  const only = process.env.SHOTS_FILTER ? new RegExp(process.env.SHOTS_FILTER) : null;
  for (const [name, url, vps, full, action] of SHOTS) {
    for (const vp of vps) {
      if (only && !only.test(`${vp}/${name}`)) continue;
      const ctx = await browser.newContext(VIEWPORTS[vp]);
      const page = await ctx.newPage();
      page.on("pageerror", (e) => console.error(`[${name} ${vp}]`, e.message));
      // SHOTS_NOW freezes the clock (stale/historical states against recorded data)
      if (process.env.SHOTS_NOW) await page.clock.setFixedTime(new Date(process.env.SHOTS_NOW));
      await page.goto(`http://localhost:${PORT}${url}`, { waitUntil: "networkidle" });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(1200);
      if (url.startsWith("/") && !url.startsWith("/bloom") && !url.startsWith("/fisheries") && !url.startsWith("/sources")) {
        await page.waitForFunction(() => !!window.__cwMap && window.__cwMap.loaded(), null, { timeout: 30000 }).catch(() => {});
        await page.waitForTimeout(1500);
      }
      if (action) await action(page);
      // full-page captures: pin the mobile tab bar to the end of the page instead of mid-image
      if (full && !action) await page.addStyleTag({ content: "body{position:relative}[data-testid=tabbar]{position:absolute!important}" });
      const file = path.join(OUT, vp, `${name}.png`);
      await mkdir(path.dirname(file), { recursive: true });
      await page.screenshot({ path: file, fullPage: full && !action });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      console.log(`${vp}/${name}.png${overflow > 1 ? `  OVERFLOW ${overflow}px` : ""}`);
      await ctx.close();
    }
  }
} finally {
  await browser.close();
  server.kill();
}
