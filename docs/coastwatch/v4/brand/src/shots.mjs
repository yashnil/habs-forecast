// Renders board-desktop.png (1440 wide, full page) and board-favicons.png (favicon strips at 1x, blown up 8x nearest-neighbour).
import { chromium } from '/Users/yashnilmohanty/Desktop/habs-forecast/coastwatch-web/node_modules/playwright/index.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const dir = join(dirname(fileURLToPath(import.meta.url)), '..');
const tmp = join(dir, 'src', '.shots'); mkdirSync(tmp, { recursive: true });
const b = await chromium.launch({ args: ['--disable-gpu'] });
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await p.goto('file://' + join(dir, 'board.html'));
await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(500);
await p.screenshot({ path: join(dir, 'board-desktop.png'), fullPage: true });
const rows = [];
for (const k of ['a', 'b', 'c']) {
  const f = join(tmp, `fav-${k}.png`);
  await p.locator(`#fav-${k}`).screenshot({ path: f });
  const box = await p.locator(`#fav-${k}`).boundingBox();
  rows.push({ k, f, w: Math.round(box.width), h: Math.round(box.height) });
}
const html = `<!doctype html><body style="margin:0;padding:24px;background:#fff;font:600 28px system-ui">` + rows.map(r =>
  `<div style="margin:0 0 24px"><div style="margin:0 0 8px">${r.k.toUpperCase()} &nbsp;<span style="font-weight:400;font-size:18px;color:#667">16 · 24 · 32 px on light and dark tabs, 8× nearest-neighbour</span></div>
   <img src="file://${r.f}" style="width:${r.w * 8}px;height:${r.h * 8}px;image-rendering:pixelated;display:block"></div>`).join('') + `</body>`;
writeFileSync(join(tmp, 'fav.html'), html);
const q = await b.newPage({ viewport: { width: 1200, height: 800 } });
await q.goto('file://' + join(tmp, 'fav.html')); await q.waitForTimeout(300);
await q.screenshot({ path: join(dir, 'board-favicons.png'), fullPage: true });
await b.close(); console.log('shots ok');
