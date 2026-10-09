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
const SHOTS = [
  ["01-map", "/", ["desktop-1440", "laptop-1280", "mobile-390"], false],
  ["02-bloom", "/bloom", ["desktop-1440", "laptop-1280", "mobile-390"], true],
  ["03-fisheries", "/fisheries", ["desktop-1440", "laptop-1280", "mobile-390"], true],
  ["04-sources", "/sources", ["desktop-1440", "mobile-390"], false],
  ["05-official-drawer", "/fisheries", ["desktop-1440", "mobile-390"], false, drawer],
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
  for (const [name, url, vps, full, action] of SHOTS) {
    for (const vp of vps) {
      const ctx = await browser.newContext(VIEWPORTS[vp]);
      const page = await ctx.newPage();
      page.on("pageerror", (e) => console.error(`[${name} ${vp}]`, e.message));
      await page.goto(`http://localhost:${PORT}${url}`, { waitUntil: "networkidle" });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(1200);
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
