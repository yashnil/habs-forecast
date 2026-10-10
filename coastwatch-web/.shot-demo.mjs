import { chromium } from "@playwright/test";
const [out, base, ...specs] = process.argv.slice(2);
const b = await chromium.launch();
const errs = [];
for (const s of specs) {
  const [name, path, vp] = s.split("|");
  const mobile = vp === "m";
  const ctx = await b.newContext(mobile ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 } : { viewport: { width: 1440, height: 900 } });
  const p = await ctx.newPage();
  p.on("console", (m) => m.type() === "error" && errs.push(`${name}: ${m.text()}`));
  p.on("pageerror", (e) => errs.push(`${name}: PAGEERROR ${e.message}`));
  await p.goto(base + path, { waitUntil: "networkidle" }).catch((e) => errs.push(`${name}: ${e.message}`));
  await p.waitForTimeout(2500);
  await p.screenshot({ path: `${out}/${name}.png` });
  await ctx.close();
}
await b.close();
console.log(errs.join("\n") || "no console errors");
