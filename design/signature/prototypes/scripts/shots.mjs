// Screenshot harness: node scripts/shots.mjs <outDir> <name>=<hash>@<WxH>[:action;action] ...
// Actions: click=<css> | clickmap=<x>,<y> | wait=<ms> | key=<Key> | hover=<css> | eval=<js> | full
// Records console errors and horizontal overflow for every shot in <outDir>/report.json.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const [out, ...specs] = process.argv.slice(2);
const BASE = process.env.BASE ?? "http://127.0.0.1:3460/";
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const report = [];
for (const spec of specs) {
  const m = spec.match(/^([^=]+)=([^@]*)@(\d+)x(\d+)(?::(.*))?$/);
  if (!m) { console.error("bad spec", spec); continue; }
  const [, name, hash, w, h, acts] = m;
  const reduced = name.includes("reduced");
  const ctx = await browser.newContext({ viewport: { width: +w, height: +h }, deviceScaleFactor: 2, reducedMotion: reduced ? "reduce" : "no-preference", hasTouch: +w < 700, isMobile: +w < 700 });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text().slice(0, 300)); });
  page.on("pageerror", (e) => errors.push(String(e).slice(0, 300)));
  await page.goto(BASE + "#/" + hash, { waitUntil: "load" });
  await page.waitForTimeout(2500);
  let full = false;
  for (const a of (acts ?? "").split(";").filter(Boolean)) {
    const [k, ...rest] = a.split("="); const v = rest.join("=");
    if (k === "click") await page.click(v, { timeout: 5000 }).catch((e) => errors.push("action " + a + ": " + e.message.slice(0, 120)));
    else if (k === "clickmap") { const [x, y] = v.split(",").map(Number); await page.mouse.click(x, y); }
    else if (k === "tap") { const [x, y] = v.split(",").map(Number); await page.touchscreen.tap(x, y); }
    else if (k === "wait") await page.waitForTimeout(+v);
    else if (k === "key") await page.keyboard.press(v);
    else if (k === "hover") await page.hover(v).catch(() => {});
    else if (k === "eval") await page.evaluate(v);
    else if (k === "full") full = true;
  }
  await page.waitForTimeout(700);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  const file = path.join(out, `${name}.jpg`);
  await page.screenshot({ path: file, fullPage: full, type: "jpeg", quality: 82 });
  report.push({ name, hash, viewport: `${w}x${h}`, overflow, errors });
  console.log(`${name}: overflow=${overflow} errors=${errors.length}${errors.length ? " — " + errors[0] : ""}`);
  await ctx.close();
}
await browser.close();
fs.writeFileSync(path.join(out, "report.json"), JSON.stringify(report, null, 2));
