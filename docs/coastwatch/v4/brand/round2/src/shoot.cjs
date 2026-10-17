// Usage: node shoot.cjs <html> <out.png> [dsf=1] [width=1200] [selector]
// Screenshots a local HTML file with a real Chromium (Playwright) at the given device scale factor.
const PW = process.env.PW || '/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/66a5feca-240e-4ae7-98f5-149a623ea13b/scratchpad/v4/coastwatch-web/node_modules/playwright';
const { chromium } = require(PW);
const path = require('path');
(async () => {
  const [html, out, dsf = '1', width = '1200', sel] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--disable-gpu', '--font-render-hinting=none'] });
  const p = await b.newPage({ viewport: { width: +width, height: 800 }, deviceScaleFactor: +dsf });
  await p.goto('file://' + path.resolve(html));
  await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(400);
  if (sel) await p.locator(sel).screenshot({ path: out });
  else await p.screenshot({ path: out, fullPage: true });
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
