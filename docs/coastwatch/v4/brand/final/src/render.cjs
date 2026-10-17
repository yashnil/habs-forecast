// Neritic final comparison (R2 Bight vs C2 Chart label): real-browser renders.
// 1) every [data-shot] in src/parts.html at device scale 1 and 2 -> shots/<name>.png, shots/<name>@2x.png
// 2) board.html -> board.png (full page); summary.html -> summary.png
// Run: node docs/coastwatch/v4/brand/final/src/render.cjs [parts|board|summary]
// Needs network for Google Fonts (Atkinson Hyperlegible Next, Source Serif 4, IBM Plex Mono: the v4 faces).
const PW = process.env.PW || '/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/66a5feca-240e-4ae7-98f5-149a623ea13b/scratchpad/v4/coastwatch-web/node_modules/playwright';
const { chromium } = require(PW);
const path = require('path');
const F = path.join(__dirname, '..');
(async () => {
  const b = await chromium.launch({ args: ['--disable-gpu', '--allow-file-access-from-files'] });
  const only = process.argv[2];
  if (!only || only === 'parts') {
    for (const dsf of [1, 2]) {
      const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: dsf });
      await p.goto('file://' + path.join(F, 'src', 'parts.html'));
      await p.waitForLoadState('load'); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(500);
      const names = await p.$$eval('[data-shot]', els => els.map(e => e.dataset.shot));
      for (const n of names) {
        await p.locator(`[data-shot="${n}"]`).screenshot({ path: path.join(F, 'shots', n + (dsf === 2 ? '@2x' : '') + '.png') });
      }
      await p.close();
      console.log(`parts @${dsf}x: ${names.length}`);
    }
  }
  for (const [src, out, w] of [['board.html', 'board.png', 1760], ['summary.html', 'summary.png', 1600]]) {
    if (only && only !== out.split('.')[0]) continue;
    const p = await b.newPage({ viewport: { width: w, height: 900 }, deviceScaleFactor: 1 });
    await p.goto('file://' + path.join(F, src));
    await p.waitForLoadState('load'); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(600);
    await p.screenshot({ path: path.join(F, out), fullPage: true });
    await p.close(); console.log(out);
  }
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
