import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";
import { coveragePct } from "../../src/lib/layers";

/**
 * Feature parity with v2, in v4's interaction architecture (the owner's list): home port, waters
 * chosen on My Coast's map, currents compared with chlorophyll, regional satellite coverage, notice
 * counts for a place, per-variable harbour summaries, and the trip checklist's own ticks. Each keeps
 * its provenance: where, when, which product, and what it does not mean.
 */
const NOW = "2026-10-08T20:00:00Z";
const BRIEF = "2026-10-13T06:00:00Z";
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const intel = JSON.parse(readFileSync(path.join(FIX, manifest.port_intel_url), "utf8"));
type M = { loaded(): boolean; getSource(id: string): unknown; jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number } };
const map = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());

test("home port: set it from the title's place menu with the trip, it survives a reload, another harbour offers the way back, and it can be forgotten", async ({ page }) => {
  await page.clock.setFixedTime(new Date(BRIEF));
  await page.goto(`${OK}/coast?place=port-550&for=commercial`);
  // the title is the place control
  await page.getByTestId("coast-place-button").click();
  await expect(page.getByTestId("coast-place-button")).toHaveAttribute("aria-expanded", "true");
  await page.getByTestId("coast-home-set").click();
  await expect(page.getByTestId("coast-home")).toContainText("Your home port");
  // stored with its trip, on this device only
  expect(JSON.parse((await page.evaluate(() => localStorage.getItem("cw-home-port")))!)).toEqual({ place: "port-550", for: "commercial", species: "any" });
  await page.reload();
  await expect(page.getByTestId("coast-home")).toContainText("Your home port");
  // change harbour from the place menu: the trip is kept, the home port is one tap away
  await page.getByTestId("coast-place-button").click();
  await page.getByTestId("coast-place-picker").selectOption("port-201");
  await expect(page).toHaveURL(/place=port-201/);
  await expect(page).toHaveURL(/for=commercial/);
  await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
  await page.getByTestId("coast-home-open").click();
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  await expect(page).toHaveURL(/for=commercial/);
  // the landing lists it first
  await page.goto(`${OK}/coast`);
  await expect(page.getByTestId("coast-landing")).toContainText("Your home port");
  await page.goto(`${OK}/coast?place=port-550`);
  await page.getByTestId("coast-place-button").click();
  await page.getByTestId("coast-home-clear").click();
  await expect(page.getByTestId("coast-home-set")).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("cw-home-port"))).toBeNull();
});

test("an older home port (a plain place key) still opens as home", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("cw-home-port", "port-550"));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-home")).toContainText("Your home port");
  await page.goto(`${OK}/coast?place=port-201`);
  await expect(page.getByTestId("coast-home-open")).toContainText("Home: Monterey");
});

test("My Coast's map: tapping the water briefs that point, with the trip kept", async ({ page }) => {
  await page.clock.setFixedTime(new Date(BRIEF));
  await page.goto(`${OK}/coast?place=port-550&for=recreational`);
  const loc = page.getByTestId("locator-caption");
  await loc.scrollIntoViewIfNeeded();
  await expect(loc).toContainText("Tap a harbour or the water");
  await map(page);
  await page.waitForTimeout(500);
  const box = (await page.locator(".maplibregl-canvas").first().boundingBox())!;
  const pt = await page.evaluate(() => (window as unknown as { __cwMap: M }).__cwMap.project([-121.95, 36.72]));
  await page.mouse.click(box.x + pt.x, box.y + pt.y);
  await expect(page).toHaveURL(/place=point-36\.7\d\d,-121\.9\d\d/);
  await expect(page).toHaveURL(/for=recreational/);
  await expect(page.getByTestId("coast-subtitle")).toContainText("Point chosen on the map");
  // a point is not a home port
  await page.getByTestId("coast-place-button").click();
  await expect(page.getByTestId("coast-place-bar")).toContainText("cannot be a home port");
  await expect(page.getByTestId("coast-home-set")).toHaveCount(0);
});

test("currents compared with chlorophyll: both drawn, each with its own time, never one combined value", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=currents`);
  await map(page);
  await page.getByTestId("currents-chl-toggle").check();
  await expect(page).toHaveURL(/chl=1/);
  await page.waitForFunction(() => { const m = (window as unknown as { __cwMap: M }).__cwMap; return !!m.getSource("satellite"); });
  await expect(page.getByTestId("stamp-currents").first()).toContainText("Currents, HF radar · observed");
  await expect(page.getByTestId("stamp-satellite").first()).toContainText("Chlorophyll");
  await expect(page.getByTestId("legend-compare").first()).toContainText("not toxin");
  await expect(page.getByTestId("currents-chl")).toContainText("not combined into one value");
  // a link keeps the comparison, and stepping through hours keeps it too
  await page.reload();
  await map(page);
  await expect(page.getByTestId("currents-chl-toggle")).toBeChecked();
  await page.getByTestId("time-mean").click();
  await expect(page).toHaveURL(/chl=1/);
});

test("satellite coverage: the share actually seen, what is counted, the dates and the region", async ({ page }) => {
  const l = manifest.layers.find((x: { layer_id: string }) => x.layer_id === "olci300_chl_latest");
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=olci300`);
  await map(page);
  const cov = page.getByTestId("sat-coverage").first();
  await expect(cov).toContainText(`Seen clearly: ${coveragePct(l.coverage.domain_observed_fraction)}`);
  await expect(cov).toContainText(`${l.coverage.domain_reference_cells} ocean cells across the California domain`);
  await expect(cov).toContainText("not low chlorophyll");
  await expect(cov).toContainText(/pixels observed \w{3} \d+–\w{3} \d+/);
  await expect(page.getByTestId("sat-coverage-regions").first()).toContainText(`Monterey Bay ${coveragePct(l.coverage.regions[0].observed_fraction)}`);
  await cov.locator("summary").click();
  await expect(cov).toContainText(l.coverage.reference.slice(0, 40));
});

test("notices for a place: the count of those covering its waters leads, the rest follow, all unverified", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await map(page);
  await expect(page.getByTestId("notices-here")).toBeVisible();
  const here = Number(await page.getByTestId("notices-count").innerText());
  expect(here).toBeGreaterThan(0);
  await page.getByTestId("notices-button").click();
  const local = page.getByTestId("notices-local");
  await expect(local).toContainText(`Cover Monterey’s waters · ${here}`);
  await expect(local).toContainText("Not verified");
  await expect(local.locator('li[data-testid^="notice-"]')).toHaveCount(here);
  await expect(local.getByTestId("notice-cdph-2026-sn26-019-anchovy-central-coast")).toBeVisible();
  await expect(page.getByTestId("notices-panel")).toContainText("Elsewhere in California");
});

test("advanced harbour summary: each product's neighbourhood statistic with its cells or pixels, size and dates", async ({ page }) => {
  const p = intel.ports.find((x: { port_code: number }) => x.port_code === 550);
  const lead1 = p.charm.leads.find((l: { lead_days: number }) => l.lead_days === 1).variables.particulate_domoic;
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&var=particulate_domoic&lead=1&sel=harbor:550`);
  await map(page);
  const sum = page.getByTestId("harbor-summary");
  await sum.locator("summary").click();
  const row = sum.getByTestId("harbor-summary-particulate_domoic");
  await expect(row).toContainText(`${Math.round(lead1.median * 100)}%`);
  await expect(row).toContainText(String(lead1.n));
  await expect(sum.getByTestId("harbor-summary-charm")).toContainText(`${p.charm.cells_in_radius}`);
  await expect(sum.getByTestId("harbor-summary-charm")).toContainText("not at the harbour, not a toxin measurement");
  await expect(sum.getByTestId("harbor-summary-chl")).toContainText(`${p.chlorophyll.latest.n}`);
  await expect(sum.getByTestId("harbor-summary-chl")).toContainText("VIIRS 750 m");
  await expect(sum.getByTestId("harbor-summary-chl")).toContainText("not toxin");
});

test("trip checklist: the reader's own ticks, kept per trip on this device, never a verification", async ({ page }) => {
  await page.clock.setFixedTime(new Date(BRIEF));
  await page.goto(`${OK}/coast?place=port-201&for=commercial&species=dungeness`);
  const first = page.getByTestId("checks").locator(":scope > li").first();
  const id = (await first.getAttribute("data-testid"))!;
  await expect(page.getByTestId("checks-progress")).toContainText("0 of");
  await expect(page.getByTestId("checks-progress")).toContainText("not a check by Neritic or by any agency");
  await page.getByTestId(`${id}-box`).check();
  await expect(first).toHaveAttribute("data-done", "true");
  await expect(page.getByTestId("checks-progress")).toContainText("1 of");
  await page.reload();
  await expect(page.getByTestId(`${id}-box`)).toBeChecked();
  // another trip has its own list
  await page.goto(`${OK}/coast?place=port-201&for=recreational`);
  await expect(page.getByTestId("checks-progress")).toContainText("0 of");
});

/* Defects found by the final scripted walkthrough (2026-10-17), each held here. */

test("walkthrough H1: a radar hour's chip is dated by its Pacific day, as its stamp is", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=currents`);
  await map(page);
  // 01:00 UTC Oct 8 is 6 PM PDT Oct 7
  const chip = page.getByTestId("time-20261008T01Z").first();
  await expect(chip).toContainText("6:00 PM");
  await expect(chip).toContainText("Oct 7");
  await expect(page.getByTestId("time-20261008T07Z").first()).toContainText("Oct 8");
  // each chip holds its whole label ("10:00 AM PDT" did not fit an 84 px chip and ran into the next)
  const overflow = await page.locator('[data-testid^="time-"][role="radio"]').evaluateAll((els) => els.filter((e) => [...e.children].some((c) => c.scrollWidth > e.clientWidth)).map((e) => e.textContent));
  expect(overflow).toEqual([]);
});

test("walkthrough H2: the multi-sensor coverage names the days its pixels come from, not one overpass", async ({ page }) => {
  const l = manifest.layers.find((x: { layer_id: string }) => x.layer_id === "multisensor_chl_latest");
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=multi`);
  await map(page);
  const cov = page.getByTestId("sat-coverage").first();
  await expect(cov).toContainText(`Seen clearly: ${coveragePct(l.coverage.domain_observed_fraction)}`);
  await expect(cov).toContainText("pixels observed Oct 1–Oct 6, each its latest clear day");
  await expect(cov).not.toContainText("overpass of");
});

for (const width of [390, 320]) {
  test(`walkthrough H3: at ${width} px the notices panel opens above the harbour sheet and its wording can be read`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width, height: width === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
    await map(page);
    await page.getByTestId("notices-button").click();
    const panel = page.getByTestId("notices-panel");
    await expect(panel).toBeVisible();
    // the panel is the top-most thing at its own centre (not the sheet, not the map credits)
    const top = await panel.evaluate((el) => { const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return !!hit && el.contains(hit); });
    expect(top).toBe(true);
    const words = panel.locator("summary", { hasText: "The agency’s wording, area and sources" }).first();
    await words.scrollIntoViewIfNeeded();
    await words.click({ timeout: 5000 });
    await expect(words.locator("..")).toHaveAttribute("open", "");
    await ctx.close();
  });
}

test("walkthrough H4: on a phone the briefing offers the map at its top, and a tapped point opens at the top of its briefing", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(BRIEF));
  await page.goto(`${OK}/coast?place=port-550&for=recreational`);
  // the title's place menu offers the map, at the top of the page
  await page.getByTestId("coast-place-button").click();
  const jump = page.getByTestId("coast-map-jump");
  await expect(jump).toBeInViewport();
  await jump.click();
  await expect(page.getByTestId("locator-caption")).toBeInViewport();
  await map(page);
  await page.waitForTimeout(500);
  const box = (await page.locator(".maplibregl-canvas").first().boundingBox())!;
  const pt = await page.evaluate(() => (window as unknown as { __cwMap: M }).__cwMap.project([-121.95, 36.72]));
  await page.mouse.click(box.x + pt.x, box.y + pt.y);
  await expect(page).toHaveURL(/place=point-/);
  await expect(page.getByTestId("coast-subtitle")).toContainText("Point chosen on the map");
  await expect(page.getByTestId("coast-title")).toBeInViewport();
  await ctx.close();
});

for (const width of [1440, 320]) {
  test(`'Show on map' from a briefing: at ${width} px the way back is uncovered and returns to it`, async ({ browser }) => {
    const ctx = await browser.newContext(width === 320 ? { viewport: { width, height: 640 }, isMobile: true, hasTouch: true } : { viewport: { width, height: 900 } });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=port-550`);
    await page.getByTestId("locator-explore").click();
    const back = page.getByTestId("back-to-briefing");
    await expect(back).toBeInViewport();
    const top = await back.evaluate((el) => { const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return !!hit && el.contains(hit); });
    expect(top).toBe(true);
    await back.click();
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
    await ctx.close();
  });
}
