// cold loads in fresh contexts across pages, viewports and saved-preference states; logs component stacks
import { chromium } from "playwright";
import fs from "node:fs";
const [base, n, out] = process.argv.slice(2);
const pages = ["/", "/coast?place=port-201&for=commercial&species=dungeness", "/coast?place=point-34.000,-122.900", "/fisheries", "/bloom", "/about"];
const vps = [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 320, height: 700 }];
const states = {
  none: {},
  dismissed: { "cw-intro-dismissed-v1": "1" },
  research: { "cw-intro-dismissed-v1": "1", "cw-mode-v1": "research" },
  purpose: { "cw-intro-dismissed-v1": "1", "cw-purpose-v1": JSON.stringify({ purpose: "commercial" }), "cw-home-port": "port-550" },
};
const b = await chromium.launch();
let loads = 0, hits = [];
for (let i = 0; i < +n; i++) for (const path of pages) for (const vp of vps) for (const [sn, st] of Object.entries(states)) {
  if (Math.random() > +(process.env.SAMPLE ?? 0.35)) continue; // sample the matrix
  const ctx = await b.newContext({ viewport: vp, isMobile: vp.width < 500, hasTouch: vp.width < 500 });
  await ctx.addInitScript((st) => { try { for (const [k, v] of Object.entries(st)) localStorage.setItem(k, v); } catch {} }, st);
  const p = await ctx.newPage(); const msgs = [];
  p.on("console", (m) => m.text().includes("CW-HYDRATION-DEBUG") && msgs.push(m.text()));
  p.on("pageerror", (e) => /418|hydrat/i.test(String(e)) && msgs.push("PAGEERROR " + String(e).slice(0, 100)));
  await p.goto(base + path, { waitUntil: "networkidle" }).catch(() => null); await p.waitForTimeout(800);
  loads++;
  if (msgs.length) { hits.push({ path, vp: vp.width, state: sn, msgs }); console.log("HIT", path, vp.width, sn); }
  await ctx.close();
}
fs.writeFileSync(out, JSON.stringify(hits, null, 1));
console.log(`loads ${loads}, hits ${hits.length}`);
await b.close();
