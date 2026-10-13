// Interaction performance: coastal journey frame times, day-switch and selection latency.
import { chromium } from "playwright";
const BASE = process.env.BASE ?? "http://127.0.0.1:3461/";
const b = await chromium.launch({ args: process.env.GPU ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"] : [] });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto(BASE + "#/explorer?at=-122.06,36.80&show=forecast&welcome=0");
await p.waitForTimeout(3500);
// 1. journey Santa Cruz → Crescent City (≈ 560 km is flyTo; pick Morro Bay ≈ 190 km for the coast path)
await p.evaluate(() => { window.__f = []; let last = performance.now(); const tick = (t) => { window.__f.push(t - last); last = t; if (window.__f.length < 400) requestAnimationFrame(tick); }; requestAnimationFrame(tick); });
await p.click(".search input"); await p.keyboard.type("Morro"); await p.waitForTimeout(150); await p.keyboard.press("Enter");
await p.waitForTimeout(3200);
const frames = await p.evaluate(() => window.__f.slice(2));
const sorted = [...frames].sort((a, b) => a - b);
const pct = (q) => sorted[Math.floor(q * (sorted.length - 1))];
// 2. day switch latency: click → the time bar says the new day is selected → field source updated (next frame)
await p.goto(BASE + "#/explorer?at=-122.06,36.80&show=forecast&welcome=0"); await p.waitForTimeout(3500);
const dayLat = [];
for (const i of [3, 1, 0, 2]) {
  const t = await p.evaluate(async (i) => { const btn = document.querySelectorAll(".tday")[i]; const t0 = performance.now(); btn.click(); await new Promise((r) => { const chk = () => (btn.getAttribute("aria-checked") === "true" ? requestAnimationFrame(() => r()) : requestAnimationFrame(chk)); chk(); }); return performance.now() - t0; }, i);
  dayLat.push(Math.round(t)); await p.waitForTimeout(600);
}
// 3. selection latency: click on the water → ledger title updated
const selLat = [];
for (const [x, y] of [[900, 500], [1000, 650], [1100, 400]]) {
  const before = await p.textContent(".place-title");
  const t0 = Date.now(); await p.mouse.click(x, y);
  await p.waitForFunction((b) => document.querySelector(".place-coord, .place-head .t-coord")?.textContent !== b, before, { timeout: 3000 }).catch(() => {});
  selLat.push(Date.now() - t0); await p.waitForTimeout(700);
}
console.log(JSON.stringify({ journeyFrames: frames.length, frameP50: pct(0.5).toFixed(1), frameP95: pct(0.95).toFixed(1), frameMax: Math.max(...frames).toFixed(1), over33ms: frames.filter((f) => f > 33.4).length, dayLatencyMs: dayLat, selectLatencyMs: selLat }, null, 1));
await b.close();
