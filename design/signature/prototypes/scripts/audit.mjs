// Accessibility + performance + first-screen audit across routes and viewports.
// node scripts/audit.mjs <out.json>   (BASE defaults to the preview server, http://127.0.0.1:3461/)
import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";

const BASE = process.env.BASE ?? "http://127.0.0.1:3461/";
const VIEWPORTS = [[1440, 900], [1280, 720], [1024, 768], [390, 844], [320, 640]];
const ROUTES = [
  ["map-first", "explorer"],
  ["map-selected", "explorer?at=-122.06,36.80&show=forecast&welcome=0"],
  ["map-research", "explorer?at=-122.06,36.80&mode=research&show=forecast&welcome=0"],
  ["station", "station"],
  ["states", "states"],
  ["directions-c", "directions?d=c"],
];
const out = [];
const browser = await chromium.launch();
for (const [name, hash] of ROUTES) for (const [w, h] of VIEWPORTS) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: w < 700, isMobile: w < 700 });
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    window.__cls = 0; window.__long = []; window.__lcp = 0;
    new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) window.__cls += e.value; }).observe({ type: "layout-shift", buffered: true });
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push(Math.round(e.duration)); }).observe({ type: "longtask", buffered: true });
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lcp = e.startTime; }).observe({ type: "largest-contentful-paint", buffered: true });
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e).slice(0, 200)));
  const t0 = Date.now();
  await page.goto(BASE + "#/" + hash, { waitUntil: "load" });
  await page.waitForTimeout(3500);
  const metrics = await page.evaluate(() => {
    const vis = (sel) => { const el = document.querySelector(sel); if (!el) return null; const r = el.getBoundingClientRect(); return r.height > 0 && r.top >= 0 && r.bottom <= innerHeight && r.left >= 0 && r.right <= innerWidth + 1; };
    const small = [...document.querySelectorAll("body *")].filter((el) => { if (!el.childNodes.length || ![...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) return false; const cs = getComputedStyle(el); const r = el.getBoundingClientRect(); return r.width > 0 && cs.visibility !== "hidden" && parseFloat(cs.fontSize) < 12 && !el.closest(".maplibregl-ctrl-attrib, .sr-only"); }).length;
    const smallTargets = [...document.querySelectorAll("button, a[href], [role=radio], select, input")].filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && (r.width < 24 || r.height < 24) && !el.closest(".maplibregl-ctrl-attrib"); }).length;
    return {
      overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      cls: +window.__cls.toFixed(4), longTasks: window.__long, lcp: Math.round(window.__lcp),
      placeTitleVisible: vis(".place-title, h1"),
      officialLinkVisible: [...document.querySelectorAll("a")].some((a) => /cdph|wildlife\.ca\.gov/.test(a.href) && (() => { const r = a.getBoundingClientRect(); return r.height > 0 && r.top >= 0 && r.bottom <= innerHeight; })()),
      textUnder12px: small, targetsUnder24px: smallTargets,
    };
  });
  const axe = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
  const v = axe.violations.map((x) => ({ id: x.id, impact: x.impact, n: x.nodes.length, sample: x.nodes[0]?.target?.join(" ") }));
  out.push({ name, viewport: `${w}x${h}`, ms: Date.now() - t0, ...metrics, axe: v, errors });
  console.log(`${name} ${w}x${h} axe=${v.map((x) => `${x.id}(${x.impact},${x.n})`).join(",") || "0"} cls=${metrics.cls} lcp=${metrics.lcp} long=${metrics.longTasks.length}/${Math.max(0, ...metrics.longTasks)}ms title=${metrics.placeTitleVisible} official=${metrics.officialLinkVisible} small=${metrics.textUnder12px} tiny=${metrics.targetsUnder24px} ox=${metrics.overflowX} err=${errors.length}`);
  await ctx.close();
}
await browser.close();
fs.writeFileSync(process.argv[2] ?? "audit.json", JSON.stringify(out, null, 2));
