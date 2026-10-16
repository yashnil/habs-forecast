import { chromium } from "../../../../../coastwatch-web/node_modules/@playwright/test/index.mjs";
const dir = new URL(".", import.meta.url).pathname;
const only = process.argv[2]?.split(",");
const b = await chromium.launch();
const report = [];
for (const [w, h] of [[1440, 900], [390, 844], [320, 700]]) {
  const ctx = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: w === 1440 ? 1 : 2, isMobile: w < 760, hasTouch: w < 760 });
  const p = await ctx.newPage(); const errs = []; p.on("pageerror", e => errs.push(e.message)); p.on("console", m => m.type() === "error" && errs.push(m.text()));
  for (const s of ["mycoast", "evidence", "map", "station"]) {
    if (only && !only.includes(s)) continue;
    await p.goto(`file://${dir}/${s}.html`); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(s === "map" ? 6000 : 800);
    const ov = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    const wide = await p.evaluate(() => [...document.querySelectorAll("body *")].filter(e => { const r = e.getBoundingClientRect(); return r.right > innerWidth + 1 && r.width > 0 && getComputedStyle(e).position !== "fixed"; }).slice(0, 4).map(e => e.tagName + "." + e.className));
    report.push(`${w} ${s} overflow=${ov} ${wide.join(" ")} errs=${errs.splice(0).join("|").slice(0, 200)}`);
    if (w === 320) continue;
    await p.screenshot({ path: `${dir}/shots/${s}-${w}.png` });
    if (s !== "map") {
      await p.screenshot({ path: `${dir}/shots/${s}-${w}-full.png`, fullPage: true });
      if (w === 390) { await p.evaluate(() => scrollTo(0, innerHeight * 1.6)); await p.waitForTimeout(300); await p.screenshot({ path: `${dir}/shots/${s}-390-scrolled.png` }); }
    }
  }
  await ctx.close();
}
await b.close(); console.log(report.join("\n"));
