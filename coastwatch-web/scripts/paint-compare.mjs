#!/usr/bin/env node
/**
 * First paint, first contentful paint and LCP of two builds of the same page, alternated on every
 * iteration so drift affects both equally; a fresh context per load, service workers blocked.
 *   node scripts/paint-compare.mjs <baseA> <baseB> <route> [runs=10] [phone]
 * e.g. node scripts/paint-compare.mjs http://localhost:3461 http://localhost:3460 "/coast?place=port-550" 10 phone
 * "phone" = 390x844, DPR 3, 4x CPU throttling. Prints p50/p90 per build and every run's FCP.
 */
import { chromium } from "playwright";
const [A, B, route, runsArg, phone] = process.argv.slice(2);
const RUNS = Number(runsArg ?? 10);
const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))]; };
await (async () => {
  const b = await chromium.launch();
  const out = { A: [], B: [] };
  for (let i = 0; i < RUNS; i++) {
    for (const [label, base] of i % 2 ? [["B", B], ["A", A]] : [["A", A], ["B", B]]) {
      const ctx = await b.newContext({ serviceWorkers: "block", ...(phone ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 } : { viewport: { width: 1440, height: 900 } }) });
      await ctx.addInitScript(() => {
        window.__p = { lcp: 0 };
        new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__p.lcp = e.startTime; }).observe({ type: "largest-contentful-paint", buffered: true });
      });
      const p = await ctx.newPage();
      if (phone) { const cdp = await ctx.newCDPSession(p); await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 }); }
      await p.goto(`${base}${route}`, { waitUntil: "load" });
      await p.waitForTimeout(1200);
      const r = await p.evaluate(() => {
        const n = performance.getEntriesByType("navigation")[0];
        return { dcl: n.domContentLoadedEventEnd, fp: performance.getEntriesByName("first-paint")[0]?.startTime ?? null, fcp: performance.getEntriesByName("first-contentful-paint")[0]?.startTime ?? null, lcp: window.__p.lcp };
      });
      out[label].push(r);
      await ctx.close();
    }
  }
  await b.close();
  const sum = (k, l) => { const v = out[l].map((r) => r[k]).filter((x) => x != null); return `${Math.round(q(v, 0.5))}/${Math.round(q(v, 0.9))}`; };
  const raw = (l) => out[l].map((r) => Math.round(r.fcp)).join(" ");
  console.log(JSON.stringify({ A, B, route, phone: !!phone, runs: RUNS }));
  for (const l of ["A", "B"]) console.log(`${l}  DCL p50/p90 ${sum("dcl", l)}  FP ${sum("fp", l)}  FCP ${sum("fcp", l)}  LCP ${sum("lcp", l)}   FCP runs: ${raw(l)}`);
})();
