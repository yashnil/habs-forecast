#!/usr/bin/env node
/**
 * v13 visual gate (the owner's brief): the twelve states at 1440 × 900, 390 × 844 and 320 × 640, with the same
 * file names as the v12 baseline so each still pairs with its "before" (docs/coastwatch/v13/implementation/
 * before-after.md). It is also a hand-driven walk: what each page says is written to notes.json beside the stills.
 * Usage: npm run build, start the app (fixture or staging data), then
 *   node scripts/v13-gate-shots.mjs <baseUrl> <outDir>      (SIZES=390,320 for a subset)
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://localhost:3410";
const OUT = process.argv[3] ?? "gate-shots";
const ONLY = process.env.SIZES?.split(",");
mkdirSync(OUT, { recursive: true });
const REC = JSON.stringify({ v: 1, place: "port-592", for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
const SIZES = {
  1440: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  390: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true },
  320: { viewport: { width: 320, height: 640 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true },
};
const notes = {};
const note = (k, v) => ((notes[k] ??= []).push(v), console.log(k, (typeof v === "string" ? v : JSON.stringify(v)).slice(0, 400)));

async function ctxFor(browser, size, { returning = true } = {}) {
  const ctx = await browser.newContext(SIZES[size]);
  if (returning)
    await ctx.addInitScript((rec) => {
      if (sessionStorage.getItem("seeded")) return;
      sessionStorage.setItem("seeded", "1");
      localStorage.setItem("nx-port-v1", rec);
      localStorage.setItem("nx-tour-v1", "dismissed");
      localStorage.setItem("nx-opening-seen-v1", "1");
      localStorage.setItem("nx-coast-v1", JSON.stringify({ fishery: "dungeness_commercial" }));
    }, REC);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text().slice(0, 160)}`));
  return { ctx, page, errors };
}
const shot = (page, size, name, full = false) => page.screenshot({ path: join(OUT, `${name}-${size}.png`), fullPage: full });
const settle = (page, ms = 2500) => page.waitForTimeout(ms);
const text = async (loc) => ((await loc.count()) ? (await loc.first().innerText()).replace(/\s+/g, " ").trim() : "(absent)");
const inView = async (page, loc) => {
  const b = await loc.first().boundingBox().catch(() => null);
  const v = page.viewportSize();
  return !!b && b.y >= 0 && b.y + b.height <= (v.width <= 720 ? v.height - 56 : v.height) + 0.5;
};
async function step(name, size, fn) {
  try {
    await fn();
  } catch (e) {
    note(`ERROR ${name} ${size}`, String(e.message).split("\n").slice(0, 3).join(" | "));
  }
}
const type = async (page, label, v) => {
  const f = page.getByRole("textbox", { name: label, exact: true });
  await f.scrollIntoViewIfNeeded();
  await f.fill(v);
};

const browser = await chromium.launch({ channel: process.env.CW_PROOF_CHANNEL ?? "chrome" });
for (const size of Object.keys(SIZES).filter((s) => !ONLY || ONLY.includes(s))) {
  // 1 first-time Today: the front door, the port finder, the fishery, Today assembled
  await step("01", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size, { returning: false });
    await page.goto(`${BASE}/`);
    await page.waitForFunction(() => document.documentElement.getAttribute("data-nx-opening") === null, null, { timeout: 12000 });
    await settle(page, 1200);
    await shot(page, size, "01a-first-home");
    note(`01 home ${size}`, await text(page.getByTestId("nx-home")));
    await page.getByTestId("nx-home").getByRole("button", { name: "Find your port" }).click();
    await page.getByTestId("nx-home-finder").getByRole("combobox").pressSequentially("moss", { delay: 40 });
    await page.keyboard.press("Enter");
    await settle(page, 1200);
    await shot(page, size, "01b-first-fishery");
    await page.getByTestId("nx-fishery-chooser").getByRole("button", { name: /Dungeness crab \(commercial\)/ }).click();
    await page.getByTestId("nx-today").getByRole("heading", { level: 1 }).waitFor({ timeout: 15000 });
    await settle(page, 3500);
    await shot(page, size, "01c-first-today");
    note(`01 today ${size}`, await text(page.getByTestId("nx-today")));
    if (errors.length) note(`01 errors ${size}`, errors);
    await ctx.close();
  });
  // 2 returning Today
  await step("02", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    await page.goto(`${BASE}/today?place=port-592`);
    await page.getByTestId("nx-today").waitFor({ timeout: 15000 });
    await settle(page, 3500);
    await shot(page, size, "02-returning-today");
    await shot(page, size, "02-returning-today-full", true);
    note(`02 today ${size}`, await text(page.getByTestId("nx-today")));
    if (errors.length) note(`02 errors ${size}`, errors);
    await ctx.close();
  });
  // 3 Explore with a layer, 4 Explore at a harbour with a cell read
  await step("03", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    await page.goto(`${BASE}/explore?layer=chl`);
    await settle(page, 5000);
    await shot(page, size, "03-explore-layer-chl");
    note(`03 explore ${size}`, await text(page.getByTestId("nx-explore")));
    await page.goto(`${BASE}/explore?layer=sst`);
    await settle(page, 4000);
    await shot(page, size, "03b-explore-layer-sst");
    if (errors.length) note(`03 errors ${size}`, errors);
    await ctx.close();
  });
  await step("04", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    await page.goto(`${BASE}/explore?place=port-592`);
    await settle(page, 5000);
    await shot(page, size, "04a-explore-harbour");
    await page.goto(`${BASE}/explore?place=port-592&layer=chl&at=36.80,-121.90`);
    await settle(page, 5000);
    await shot(page, size, "04b-explore-harbour-inspect");
    note(`04 readout ${size}`, await text(page.getByTestId("nx-readout")));
    if (errors.length) note(`04 errors ${size}`, errors);
    await ctx.close();
  });
  // 5 My Coast start, 6 inputs, 7 results (the owner's case), 11a saved trips
  await step("05-07,11a", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    await page.goto(`${BASE}/coast?place=port-592`);
    await page.getByTestId("nx-coast").waitFor({ timeout: 15000 });
    await settle(page, 3500);
    await shot(page, size, "05-coast-start");
    note(`05 coast ${size}`, await text(page.getByTestId("nx-coast")));
    await page.goto(`${BASE}/coast?place=port-592&view=trip`);
    await page.getByTestId("trip-inputs").waitFor({ timeout: 15000 });
    await settle(page, 2500);
    await shot(page, size, "05b-trip-empty");
    await type(page, "Catch", "400");
    await type(page, "Price", "9");
    await type(page, "Fuel", "30");
    await type(page, "Fuel price", "8.24");
    await settle(page, 900);
    await shot(page, size, "06-economics-inputs");
    await type(page, "Bait and ice", "100");
    await type(page, "Fees", "0");
    await type(page, "Crew", "1");
    await page.getByRole("radio", { name: "After costs" }).click();
    await type(page, "Crew share", "20");
    await page.getByRole("radiogroup", { name: "Who pays fuel" }).getByRole("radio", { name: "Shared" }).click();
    await page.getByRole("radiogroup", { name: "Who pays bait and ice" }).getByRole("radio", { name: "Shared" }).click();
    await settle(page, 1800);
    const fig = {};
    for (const id of ["trip-net", "trip-gross", "trip-operating", "trip-crew", "trip-be-catch", "trip-be-price", "trip-netbar-value"]) fig[id] = await text(page.getByTestId(id));
    note(`07 figures ${size}`, fig);
    note(`07 result in view with the last input focused ${size}`, await inView(page, page.getByTestId(size === "1440" ? "trip-net" : "trip-netbar-value")));
    await shot(page, size, "07a-economics-results-at-inputs");
    await page.getByTestId("trip-hero").scrollIntoViewIfNeeded();
    await page.getByTestId("trip-results").evaluate((el) => el.scrollIntoView({ block: "start" }));
    await settle(page, 1200);
    await shot(page, size, "07-economics-results");
    await shot(page, size, "07-economics-results-full", true);
    // 11a: save, change the price, save again, open the list
    await page.getByTestId("trip-save-new").scrollIntoViewIfNeeded();
    await page.getByTestId("trip-save-new").click();
    await settle(page, 500);
    note(`11 said ${size}`, await text(page.getByTestId("trip-said")));
    await type(page, "Price", "10");
    await page.getByTestId("trip-save-new").scrollIntoViewIfNeeded();
    await page.getByTestId("trip-save-new").click();
    await settle(page, 500);
    await page.getByTestId("trip-saved-open").click();
    await settle(page, 900);
    const dialog = page.getByRole("dialog", { name: "Saved trips" });
    note(`11 saved rows ${size}`, await dialog.getByTestId("trip-saved-row").allInnerTexts());
    await shot(page, size, "11a-saved-scenarios");
    if (errors.length) note(`05 errors ${size}`, errors);
    await ctx.close();
  });
  // 11b alerts
  await step("11b", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    await page.goto(`${BASE}/alerts?place=port-592`);
    await page.getByTestId("nx-alerts").waitFor({ timeout: 15000 });
    await settle(page, 3000);
    await shot(page, size, "11b-alerts");
    await shot(page, size, "11b-alerts-full", true);
    note(`11 alerts ${size}`, await text(page.getByTestId("nx-alerts")));
    if (errors.length) note(`11b errors ${size}`, errors);
    await ctx.close();
  });
  // 8 Research with a measured pDA sample, 9 nothing in range, 10 a sample in the inspector
  await step("08-10", size, async () => {
    const { ctx, page, errors } = await ctxFor(browser, size);
    for (const st of ["HABs-SantaCruzWharf", "HABs-MontereyWharf"]) {
      await page.goto(`${BASE}/research?station=${st}&var=pDA`);
      await page.getByTestId("research-plot").waitFor({ timeout: 20000 });
      await settle(page, 3500);
      await shot(page, size, `08-research-pDA-${st.replace("HABs-", "")}`);
      note(`08 ${st} status ${size}`, await text(page.getByTestId("research-status")));
      note(`08 ${st} rail ${size}`, await text(page.getByTestId("research-rail")));
      note(`08 ${st} plot in first screen ${size}`, await inView(page, page.getByTestId("research-plot")));
    }
    await page.getByTestId("range-90d").click();
    await settle(page, 2500);
    await shot(page, size, "09-research-90d-MontereyWharf");
    note(`09 empty range ${size}`, await text(page.getByTestId("research-main")));
    await page.getByTestId("range-latest").click();
    await settle(page, 2500);
    await shot(page, size, "10-research-inspect-latest");
    note(`10 inspector ${size}`, await text(page.getByTestId("research-inspector")));
    note(`10 inspector in view ${size}`, await inView(page, page.getByTestId("research-inspector")));
    await page.goto(`${BASE}/research?station=HABs-SantaCruzWharf&var=pDA`);
    await page.getByTestId("research-plot").waitFor({ timeout: 20000 });
    await settle(page, 2500);
    const plot = page.getByTestId("research-plot");
    await plot.scrollIntoViewIfNeeded();
    await plot.focus().catch(() => {});
    await page.keyboard.press("End");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Enter");
    await settle(page, 1000);
    await shot(page, size, "10-research-inspect-keyboard");
    note(`10 keyboard inspector ${size}`, await text(page.getByTestId("research-inspector")));
    if (size === "1440") {
      const box = await plot.boundingBox();
      await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.4);
      await settle(page, 800);
      await shot(page, size, "10b-research-inspect-hover");
    }
    if (errors.length) note(`08 errors ${size}`, errors);
    await ctx.close();
  });
}
await browser.close();
writeFileSync(join(OUT, "notes.json"), JSON.stringify(notes, null, 1));
