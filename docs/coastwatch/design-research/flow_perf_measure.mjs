// usage: node measure.mjs [video]  → frame stats for arrows / canvas / gl, desktop and 4x-throttled phone
import { chromium } from "/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/759d9f4e-81da-4534-a2e9-43771e479e60/scratchpad/wt/coastwatch-web/node_modules/@playwright/test/index.mjs";
import { writeFile, mkdir } from "node:fs/promises";
const OUT = "/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/759d9f4e-81da-4534-a2e9-43771e479e60/scratchpad/v10-viz";
const video = process.argv[2] === "video";
const HOUR = "20261021T01Z";
const VP = { desktop: { viewport: { width: 1440, height: 900 } }, phone: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 3 } };
const MODES = video ? [["gl", "desktop"], ["canvas", "desktop"], ["gl", "phone"]] : [["arrows", "desktop"], ["canvas", "desktop"], ["gl", "desktop"], ["arrows", "phone"], ["canvas", "phone"], ["gl", "phone"]];
const b = await chromium.launch({ channel: "chrome", args: ["--ignore-gpu-blocklist", "--enable-gpu-rasterization"] });
const res = {};
const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
for (const [mode, dev] of MODES) {
  const runs = [];
  for (let r = 0; r < (video ? 1 : 3); r++) {
    if (video) await mkdir(`${OUT}/video-${mode}-${dev}`, { recursive: true });
    const ctx = await b.newContext({ ...VP[dev], ...(video ? { recordVideo: { dir: `${OUT}/video-${mode}-${dev}`, size: VP[dev].viewport } } : {}) });
    await ctx.addInitScript(() => {
      try { localStorage.setItem("nx-splash-seen-v2", "1"); } catch {}
      window.__lt = []; new PerformanceObserver((l) => l.getEntries().forEach((e) => window.__lt.push([e.startTime, e.duration]))).observe({ type: "longtask", buffered: true });
    });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => console.error("PAGEERR", e.message));
    const cdp = await ctx.newCDPSession(page);
    await cdp.send("Performance.enable");
    if (dev === "phone") await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
    const busy = async () => { const { metrics } = await cdp.send("Performance.getMetrics"); const g = (n) => metrics.find((m) => m.name === n)?.value ?? 0; return { task: g("TaskDuration"), script: g("ScriptDuration"), layout: g("LayoutDuration") + g("RecalcStyleDuration") }; };
    const flow = mode === "arrows" ? "" : `&flow=${mode}`;
    await page.goto(`http://localhost:3431/?layer=currents:${HOUR}&region=monterey_bay${flow}`, { waitUntil: "networkidle" }).catch(() => {});
    if (mode !== "arrows") await page.waitForFunction(() => window.__flowStats && window.__flowStats.frames > 30, null, { timeout: 45000 });
    else await page.waitForFunction(() => window.__cwMap?.getLayer("currents-arrows-1"), null, { timeout: 45000 });
    await page.waitForTimeout(2500);
    const sample = async (ms, pan) => page.evaluate(async ({ ms, pan }) => {
      const m = window.__cwMap; const st0 = window.__flowStats ? { ...window.__flowStats } : null;
      const t0 = performance.now(); const iv = []; let last = t0;
      const lt0 = window.__lt.length;
      if (pan) { m.panBy([pan, pan / 3], { duration: ms * 0.45 }); setTimeout(() => m.panBy([-pan, -pan / 3], { duration: ms * 0.45 }), ms * 0.5); }
      await new Promise((res) => { const f = (t) => { iv.push(t - last); last = t; if (t - t0 < ms) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
      const st1 = window.__flowStats;
      const lts = window.__lt.slice(lt0).filter(([s]) => s >= t0);
      iv.sort((a, b) => a - b);
      const dur = performance.now() - t0;
      return { fps: +(iv.length / (dur / 1000)).toFixed(1), p50: +iv[Math.floor(iv.length * 0.5)].toFixed(1), p95: +iv[Math.floor(iv.length * 0.95)].toFixed(1), worst: +iv[iv.length - 1].toFixed(1), longTasks: lts.length, longTaskMs: Math.round(lts.reduce((a, [, d]) => a + Math.max(0, d - 50), 0)),
        rendererMsPerFrame: st0 && st1 && st1.frames > st0.frames ? +(((st1.simMs - st0.simMs) + ((st1.drawMs ?? 0) - (st0.drawMs ?? 0))) / (st1.frames - st0.frames)).toFixed(2) : null, rendererFrames: st0 && st1 ? st1.frames - st0.frames : null };
    }, { ms, pan });
    const withBusy = async (ms, pan) => { const b0 = await busy(); const t0 = Date.now(); const r = await sample(ms, pan); const b1 = await busy(); const s = (Date.now() - t0) / 1000; return { ...r, mainBusyMsPerS: Math.round(((b1.task - b0.task) * 1000) / s), scriptMsPerS: Math.round(((b1.script - b0.script) * 1000) / s) }; };
    const idle = await withBusy(6000, 0);
    const pan = await withBusy(4000, dev === "phone" ? 220 : 500);
    if (video) await page.screenshot({ path: `${OUT}/frame-${mode}-${dev}.png` });
    runs.push({ idle, pan });
    await ctx.close();
  }
  const pick = (k, f) => med(runs.map((x) => x[k][f]));
  const keys = Object.keys(runs[0].idle);
  res[`${mode}/${dev}`] = Object.fromEntries(["idle", "pan"].map((k) => [k, Object.fromEntries(keys.map((f) => [f, pick(k, f)]))]));
  console.log(mode, dev, JSON.stringify(res[`${mode}/${dev}`]));
}
await b.close();
if (!video) await writeFile(`${OUT}/perf-flow.json`, JSON.stringify(res, null, 2));
