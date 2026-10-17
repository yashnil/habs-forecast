// Neritic round 2: real-browser renders. 1) every [data-shot] in render/strips.html at device scale 1 and 2
// -> shots/<name>.png and shots/<name>@2x.png; 2) board.html -> board.png; 3) render/favicons.html -> favicons.png.
// Run: node round2/src/render.cjs   (PW env var may point at another playwright install)
const PW = process.env.PW || '/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/66a5feca-240e-4ae7-98f5-149a623ea13b/scratchpad/v4/coastwatch-web/node_modules/playwright';
const { chromium } = require(PW);
const path = require('path');
const R2 = path.join(__dirname, '..');
(async () => {
  const b = await chromium.launch({ args: ['--disable-gpu'] });
  const only = process.argv[2];
  if (!only || only === 'strips') {
    for (const dsf of [1, 2]) {
      const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: dsf });
      await p.goto('file://' + path.join(R2, 'render', 'strips.html'));
      await p.waitForLoadState('load'); await p.waitForTimeout(300);
      const names = await p.$$eval('[data-shot]', els => els.map(e => e.dataset.shot));
      for (const n of names) {
        await p.locator(`[data-shot="${n}"]`).screenshot({ path: path.join(R2, 'shots', n + (dsf === 2 ? '@2x' : '') + '.png') });
      }
      await p.close();
      console.log(`strips @${dsf}x: ${names.length}`);
    }
  }
  for (const [src, out, w] of [['board.html', 'board.png', 1440], ['render/favicons.html', 'favicons.png', 1900]]) {
    if (only && only !== out.split('.')[0]) continue;
    const p = await b.newPage({ viewport: { width: w, height: 900 }, deviceScaleFactor: 1 });
    await p.goto('file://' + path.join(R2, src));
    await p.waitForLoadState('load'); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(600);
    await p.screenshot({ path: path.join(R2, out), fullPage: true });
    await p.close(); console.log(out);
  }
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
