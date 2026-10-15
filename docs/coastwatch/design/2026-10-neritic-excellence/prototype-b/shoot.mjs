import { chromium } from "../../../../../coastwatch-web/node_modules/@playwright/test/index.mjs";
const pages = process.argv[2] ? process.argv[2].split(",") : ["mycoast", "evidence", "map", "station"];
const b = await chromium.launch();
for (const pg of pages) for (const [w, h, mob] of [[1440, 900, false], [390, 844, true], [320, 700, true]]) {
  const p = await b.newPage({ viewport: { width: w, height: h }, isMobile: mob, hasTouch: mob, deviceScaleFactor: 2 });
  const errs = []; p.on("pageerror", (e) => errs.push(e.message)); p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
  await p.goto(`http://localhost:3521/${pg}.html`, { waitUntil: "networkidle" }).catch(() => {});
  await p.waitForFunction(() => window.__ready, null, { timeout: 30000 }).catch(() => errs.push("not ready"));
  await p.waitForTimeout(2500);
  const ov = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  const tag = w === 320 ? "320" : w === 390 ? "390" : "1440";
  await p.screenshot({ path: `shots/${pg}-${tag}.png` });
  if (mob && w === 390) { await p.evaluate(() => { const r = document.querySelector(".rail"); r.style.setProperty("--sheet-y", "0px"); r.classList.add("full"); }); await p.waitForTimeout(700); await p.screenshot({ path: `shots/${pg}-390-sheet.png` });
    await p.evaluate(() => { const r = document.querySelector(".rail"); r.scrollTop = r.scrollHeight * 0.45; }); await p.waitForTimeout(300); await p.screenshot({ path: `shots/${pg}-390-sheet2.png` }); }
  if (!mob) { const full = await p.evaluate(() => { const r = document.querySelector(".rail"); return r ? r.scrollHeight : 0; }); if (full > 860) { await p.evaluate(() => { const r = document.querySelector(".rail"); r.scrollTop = r.scrollHeight; }); await p.waitForTimeout(300); await p.screenshot({ path: `shots/${pg}-1440-scrolled.png` }); } }
  if (pg === "map" && w !== 320) for (const l of ["satellite", "currents"]) { await p.goto(`http://localhost:3521/map.html?l=${l}`, { waitUntil: "networkidle" }).catch(() => {}); await p.waitForFunction(() => window.__ready, null, { timeout: 30000 }).catch(() => {}); await p.waitForTimeout(3000); await p.screenshot({ path: `shots/map-${tag}-${l}.png` }); }
  console.log(pg, w, "overflow", ov, errs.slice(0, 3).join(" | "));
  await p.close();
}
await b.close();
