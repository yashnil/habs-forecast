#!/usr/bin/env node
/**
 * Does a page go idle? For each route, after it has settled (8 s), over the next 6 s: animation-frame callbacks
 * requested by the page, main-thread task and script time, style recalculations, layouts, requests, and frames
 * the compositor actually produced (CDP screencast, the same source Playwright's video uses). A page that is at
 * rest should ask for no frames and spend ~0 ms; one that does not costs a phone its battery and makes video
 * recording (and its finalisation at context close) expensive.
 * Usage: node scripts/v13-idle-probe.mjs <baseUrl> [out.json]   (the app on fixture data)
 */
import { writeFileSync } from "node:fs";
import { chromium } from "playwright";
const BASE = process.argv[2] ?? "http://localhost:3410";
const REC = JSON.stringify({ v: 1, place: "port-592", for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const ROUTES = [["home (first visit)", "/", false], ["today", "/", true], ["explore", "/explore?place=port-592", true], ["explore currents", "/explore?place=port-592&layer=sst&cur=1", true], ["coast", "/coast?place=port-592", true], ["trip", "/coast?place=port-592&view=trip", true], ["research", "/research?station=HABs-SantaCruzWharf&var=pDA", true], ["alerts", "/alerts?place=port-592", true]];
const SIZES = { desktop: { viewport: { width: 1440, height: 900 } }, phone: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } };
const out = [];
const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
for (const [sizeName, vp] of Object.entries(SIZES))
  for (const [name, path, seed] of ROUTES) {
    const ctx = await browser.newContext(vp);
    await ctx.addInitScript(
      ([rec, seed]) => {
        if (seed) {
          localStorage.setItem("nx-port-v1", rec);
          localStorage.setItem("nx-tour-v1", "dismissed");
          localStorage.setItem("nx-opening-seen-v1", "1");
          localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
        }
        const raf = window.requestAnimationFrame.bind(window);
        window.__raf = 0;
        window.__rafBy = {};
        window.requestAnimationFrame = (cb) => {
          window.__raf++;
          if (window.__rafTrace) {
            const at = (new Error().stack ?? "").split("\n").slice(2, 5).map((l) => l.trim().replace(/^at /, "").replace(/https?:\/\/[^/]+/, "").slice(0, 90)).join(" < ");
            window.__rafBy[at] = (window.__rafBy[at] ?? 0) + 1;
          }
          return raf(cb);
        };
        const si = window.setInterval.bind(window);
        window.__intervals = [];
        window.setInterval = (fn, ms, ...a) => (window.__intervals.push(ms), si(fn, ms, ...a));
      },
      [REC, seed],
    );
    const page = await ctx.newPage();
    const cdp = await ctx.newCDPSession(page);
    await cdp.send("Performance.enable");
    let requests = 0, counting = false, frames = 0;
    const urls = {};
    page.on("request", (r) => counting && (requests++, (urls[new URL(r.url()).pathname] = (urls[new URL(r.url()).pathname] ?? 0) + 1)));
    await page.goto(BASE + path);
    await page.waitForTimeout(8000);
    const m = async () => Object.fromEntries((await cdp.send("Performance.getMetrics")).metrics.map((x) => [x.name, x.value]));
    cdp.on("Page.screencastFrame", (e) => (frames++, cdp.send("Page.screencastFrameAck", { sessionId: e.sessionId }).catch(() => {})));
    await cdp.send("Page.startScreencast", { format: "jpeg", quality: 30, everyNthFrame: 1 });
    await page.waitForTimeout(500);
    frames = 0;
    const a = await m();
    await page.evaluate(() => ((window.__raf = 0), (window.__rafBy = {}), (window.__rafTrace = true)));
    counting = true;
    await page.waitForTimeout(6000);
    const b = await m();
    const js = await page.evaluate(() => ({ raf: window.__raf, by: Object.entries(window.__rafBy).sort((x, y) => y[1] - x[1]).slice(0, 3), intervals: window.__intervals, anims: document.getAnimations().filter((x) => x.playState === "running").map((x) => `${x.effect?.target?.className?.toString().slice(0, 40) ?? "?"}:${x.animationName ?? x.transitionProperty ?? "waapi"}:${x.effect?.getTiming().iterations}`).slice(0, 6) }));
    const row = { size: sizeName, route: name, rafPerSec: +(js.raf / 6).toFixed(1), framesPerSec: +(frames / 6).toFixed(1), taskMsPerSec: Math.round(((b.TaskDuration - a.TaskDuration) * 1000) / 6), scriptMsPerSec: Math.round(((b.ScriptDuration - a.ScriptDuration) * 1000) / 6), layoutsPerSec: +((b.LayoutCount - a.LayoutCount) / 6).toFixed(1), recalcPerSec: +((b.RecalcStyleCount - a.RecalcStyleCount) / 6).toFixed(1), requests, urls, rafBy: js.by, intervals: js.intervals, runningAnimations: js.anims };
    out.push(row);
    console.log(JSON.stringify(row));
    const t0 = Date.now();
    await ctx.close();
    row.closeMs = Date.now() - t0;
  }
await browser.close();
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify(out, null, 1));
