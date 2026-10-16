// Screenshots of board.html: the full board, per-direction crops at 1x and 2x, and pixel-exact favicon renders at 16 and 32 px.
// Run from coastwatch-web:  node ../docs/coastwatch/v2/logo/shots.mjs
import { chromium } from "../../../../coastwatch-web/node_modules/playwright/index.mjs";
import { mkdirSync, readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
const HERE = dirname(fileURLToPath(import.meta.url)); const SH = join(HERE, "shots"); mkdirSync(SH, { recursive: true });
const KEYS = ["1b", "2", "2b", "3", "R"];
const b = await chromium.launch({ channel: "chrome" });
for (const dpr of [1, 2]) {
  const p = await b.newPage({ viewport: { width: 1800, height: 900 }, deviceScaleFactor: dpr });
  await p.goto("file://" + join(HERE, "board.html")); await p.waitForTimeout(300);
  if (dpr === 1) await p.screenshot({ path: join(SH, "board.png"), fullPage: true });
  for (const k of KEYS) await p.locator(`#d${k}`).screenshot({ path: join(SH, `${k}-${dpr}x.png`) });
  await p.close();
}
// favicon test renders: exact 16 and 32 px rasters of each symbol (light on white, dark on navy), plus 8x nearest-neighbour blow-ups
const p = await b.newPage({ viewport: { width: 400, height: 120 }, deviceScaleFactor: 1 });
for (const k of KEYS) for (const [v, bg] of [["symbol", "#ffffff"], ["symbol-dark", "#0b2238"]]) for (const s of [16, 32]) {
  const favFile = v === "symbol" ? `${k}-favicon.svg` : `${k}-favicon-dark.svg`;
  const file = s === 16 && existsSync(join(HERE, "out", favFile)) ? favFile : `${k}-${v}.svg`;
  const svg = readFileSync(join(HERE, "out", file), "utf8").replace(/width="48" height="48"/, `width="${s}" height="${s}"`);
  await p.setContent(`<body style="margin:0;background:${bg}"><div style="position:absolute;left:0;top:0;line-height:0">${svg}</div></body>`);
  await p.screenshot({ path: join(SH, `fav-${k}-${v}-${s}.png`), clip: { x: 0, y: 0, width: s, height: s } });
}
// contact sheet of the favicon renders at 6x nearest neighbour
let sheet = `<body style="margin:0;background:#e9ebee;font:12px ui-monospace;padding:16px"><table style="border-collapse:collapse">`;
for (const k of KEYS) { sheet += `<tr><td style="padding:8px 12px">${k}</td>`; for (const [v, bg] of [["symbol", "#fff"], ["symbol-dark", "#0b2238"]]) for (const s of [16, 32]) { const png = readFileSync(join(SH, `fav-${k}-${v}-${s}.png`)).toString("base64"); sheet += `<td style="padding:8px 12px;background:${bg}"><img src="data:image/png;base64,${png}" style="width:${s * 6}px;height:${s * 6}px;image-rendering:pixelated;display:block"></td>`; } sheet += `</tr>`; }
sheet += `</table></body>`;
const q = await b.newPage({ viewport: { width: 900, height: 900 }, deviceScaleFactor: 1 }); await q.setContent(sheet); await q.screenshot({ path: join(SH, "favicons.png"), fullPage: true });
await b.close(); console.log("shots written to", SH);
