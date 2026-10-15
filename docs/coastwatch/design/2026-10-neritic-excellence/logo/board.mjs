import { chromium } from "../../../../../coastwatch-web/node_modules/@playwright/test/index.mjs";
import { writeFileSync, mkdirSync } from "node:fs";
import { FINAL, PAL } from "./final.mjs";
import { wordmark, FONTS } from "./wm.mjs";
// wordmark per direction: base face, size of outline units, tracking, dot treatment
const WM = {
  A: { font: FONTS.spectral500, track: 0.004, pairs: { ne: -0.004, ri: -0.006 }, label: "Spectral 500, re-spaced; i-dots redrawn as true circles" },
  B: { font: FONTS.sora500, track: -0.012, pairs: { ne: -0.006, ti: 0.004 }, label: "Sora 500, tightened; i-dots redrawn as circles at the mark's dot height" },
  C: { font: FONTS.schib500, track: 0.045, pairs: {}, label: "Schibsted Grotesk 500, letterspaced like a chart label; i-dots redrawn" },
};
const U = 100; // wordmark outline size (px) in its own units
const WMS = Object.fromEntries(Object.entries(WM).map(([k, w]) => [k, { ...w, o: wordmark(w.font, "neritic", U, w) }]));
function wordSvg(k, color, h) { // h = rendered x-height in px
  const o = WMS[k].o; const s = h / o.xh; const pad = 2; const top = o.asc * 0.9; const vbH = o.asc * 0.9 + 4;
  const dots = o.dots.map((d) => `<circle cx="${d.cx.toFixed(2)}" cy="${d.cy.toFixed(2)}" r="${(d.r * 1.04).toFixed(2)}"/>`).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" height="${(vbH * s).toFixed(1)}" viewBox="${(o.minX - pad).toFixed(1)} ${(-top).toFixed(1)} ${(o.width + 2 * pad).toFixed(1)} ${vbH.toFixed(1)}" fill="${color}"><path d="${o.body}"/>${dots}</svg>`;
}
const sym = (k, pal, s) => `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 48 48">${FINAL[k].sym(pal)}</svg>`;
// lockup: symbol sized to ~2.3 x-heights, gap 0.55 symbol; baseline-aligned optically
// lockup: the symbol cropped to its ink, its bottom on the wordmark's baseline, its height set per mark in x-heights
const K = { A: 1.36, B: 1.62, C: 1.5 };
const cropSym = (k, pal, hpx) => { const [x, y, w, h] = FINAL[k].crop; return `<svg xmlns="http://www.w3.org/2000/svg" height="${hpx.toFixed(1)}" width="${(hpx * w / h).toFixed(1)}" viewBox="${x} ${y} ${w} ${h}">${FINAL[k].sym(pal)}</svg>`; };
const lockup = (k, pal, xh, ink) => { const o = WMS[k].o; const s = xh / o.xh; return `<span style="display:inline-flex;align-items:flex-end;gap:${(xh * 0.78).toFixed(1)}px"><span style="display:block;line-height:0;margin-bottom:${(4 * s).toFixed(2)}px">${cropSym(k, pal, xh * K[k])}</span><span style="display:block;line-height:0">${wordSvg(k, ink, xh)}</span></span>`; };
mkdirSync("out", { recursive: true });
for (const k of Object.keys(FINAL)) {
  writeFileSync(`out/${k}-symbol.svg`, sym(k, PAL.light, 48)); writeFileSync(`out/${k}-symbol-dark.svg`, sym(k, PAL.dark, 48)); writeFileSync(`out/${k}-symbol-mono.svg`, sym(k, PAL.mono, 48));
  writeFileSync(`out/${k}-wordmark.svg`, wordSvg(k, "#0b2238", 40));
}
const css = `*{box-sizing:border-box}body{margin:0;background:#eef0f2;font:13px/1.4 "IBM Plex Sans",system-ui;color:#33414f}
.card{background:#fff;margin:0 0 28px;padding:28px 32px 30px;display:grid;grid-template-columns:repeat(6,1fr);gap:18px 18px;align-items:center}
.h{grid-column:1/-1;display:flex;align-items:baseline;gap:14px;border-bottom:1px solid #e3e6ea;padding-bottom:12px}.h b{font:600 20px system-ui;color:#0b2238}.h span{color:#5d6b79}
.cell{border:1px solid #e6e8eb;height:150px;display:grid;place-items:center;position:relative}.cell i{position:absolute;left:8px;bottom:6px;font:10px/1 ui-monospace,monospace;color:#7b8794;font-style:normal}
.dark{background:#071423;border-color:#071423}.navy{background:#0b2238;border-color:#0b2238}.paper{background:#f6f7f8}
.w2{grid-column:span 2}.w3{grid-column:span 3}.mast{height:56px;width:100%;display:flex;align-items:center;padding:0 16px;gap:14px}
.fav{display:flex;gap:14px;align-items:end}.px{image-rendering:pixelated}`;
let html = `<style>${css}</style><body style="padding:28px">`;
for (const k of Object.keys(FINAL)) {
  const f = FINAL[k];
  html += `<section class="card" id="${k}"><div class="h"><b>${k} · ${f.name}</b><span>${f.line}</span></div>
  <div class="cell w2" style="height:230px">${sym(k, PAL.light, 168)}<i>symbol</i></div>
  <div class="cell w2 dark" style="height:230px">${sym(k, PAL.dark, 168)}<i>symbol, dark</i></div>
  <div class="cell w2" style="height:230px;display:flex;gap:26px;justify-content:center">${sym(k, PAL.mono, 84)}<div class="navy" style="padding:14px">${sym(k, PAL.monoW, 56)}</div><i>monochrome</i></div>
  <div class="cell w3">${lockup(k, PAL.light, 30, "#0b2238")}<i>horizontal lockup</i></div>
  <div class="cell w3 dark">${lockup(k, PAL.dark, 30, "#eef3f8")}<i>horizontal lockup, dark</i></div>
  <div class="cell w2 navy" style="height:92px"><div class="mast">${lockup(k, PAL.dark, 9.2, "#ffffff")}<span style="color:#a9b8c9;font-size:12px;margin-left:auto">My Coast&nbsp;&nbsp;Map&nbsp;&nbsp;Blooms</span></div><i style="color:#7f93ab">header, 56 px bar</i></div>
  <div class="cell w2 paper" style="height:92px"><div class="mast" style="border-bottom:1px solid #dfe3e7;background:#fff">${lockup(k, PAL.light, 9.2, "#0b2238")}<span style="color:#5d6b79;font-size:12px;margin-left:auto">My Coast&nbsp;&nbsp;Map&nbsp;&nbsp;Blooms</span></div><i>header, light</i></div>
  <div class="cell w2" style="height:92px"><div class="fav"><span class="px" style="display:inline-block">${sym(k, PAL.light, 16)}</span>${sym(k, PAL.light, 32)}<span class="px" style="display:inline-block;transform:scale(3);transform-origin:bottom left;margin:0 40px 0 0">${sym(k, PAL.light, 16)}</span><span style="display:inline-block;background:#202124;padding:6px 8px;border-radius:8px 8px 0 0;display:flex;gap:6px;align-items:center;color:#e8eaed;font-size:11px">${sym(k, PAL.dark, 16)} Neritic</span></div><i>favicon 16 / 32, 16 at 3x, browser tab</i></div>
  <div class="cell w3" style="height:110px">${wordSvg(k, "#0b2238", 34)}<i>${WMS[k].label}</i></div>
  <div class="cell w3" style="height:110px;display:flex;gap:26px;justify-content:center"><div style="width:112px;height:112px;border-radius:25px;background:#0b2238;display:grid;place-items:center;transform:scale(.8)">${sym(k, PAL.dark, 76)}</div><div style="width:112px;height:112px;border-radius:25px;background:#fff;box-shadow:0 0 0 1px #e3e6ea;display:grid;place-items:center;transform:scale(.8)">${sym(k, PAL.light, 76)}</div><i>app icon</i></div>
  </section>`;
}
writeFileSync("board.html", `<!doctype html><meta charset="utf-8"><title>Neritic logo board</title>` + html);
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
await p.goto("file://" + process.cwd() + "/board.html"); await p.waitForTimeout(500);
for (const k of Object.keys(FINAL)) await p.locator(`#${k}`).screenshot({ path: `out/board-${k}.png` });
await p.screenshot({ path: "out/board.png", fullPage: true }); await b.close();
