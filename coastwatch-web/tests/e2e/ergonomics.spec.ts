import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Fixes from the v2/v4 task walkthrough (scratchpad tasks-audit): the phone workspace never scrolls
 * as a page (the dock stays at the bottom), layers are one tap from the sheet's peek, a data layer at
 * state scale says how to read it, the empty search offers the common places, a station's panel says
 * when it was last sampled and offers its data, and the briefing puts "Before you go" before the evidence.
 */
const NOW = "2026-10-08T20:00:00Z";
const phone = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

test("phone: scrolling the selection into view never slides the workspace or the dock", async ({ browser }) => {
  const ctx = await browser.newContext(phone);
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&var=particulate_domoic&lead=1&sel=cell:36.8,-121.95`);
  await expect(page.getByTestId("cell-leads")).toBeAttached();
  await page.getByTestId("cell-leads").scrollIntoViewIfNeeded();
  await page.evaluate(() => document.querySelector('[data-testid="inspect-panel"] > :last-child')?.scrollIntoView());
  expect(await page.evaluate(() => [document.scrollingElement!.scrollTop, document.querySelector('[data-testid="explore"]')!.scrollTop])).toEqual([0, 0]);
  const dock = (await page.getByTestId("nav-dock").boundingBox())!;
  expect(dock.y + dock.height).toBeGreaterThan(844 - 4);
  await ctx.close();
});

test("phone: the layers are one tap from the sheet's peek", async ({ browser }) => {
  const ctx = await browser.newContext(phone);
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=none`);
  await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "peek");
  await page.getByTestId("chip-forecast").click();
  await expect(page.getByTestId("chip-forecast")).toHaveAttribute("aria-checked", "true");
  await expect(page).toHaveURL(/layer=forecast/);
  await ctx.close();
});

test("a data layer at state scale says how to read it, and goes to Monterey Bay in one tap", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  await expect(page.getByTestId("read-hint").first()).toContainText("tap the water");
  await page.getByTestId("zoom-monterey").first().click();
  await expect.poll(() => page.evaluate(() => (window as unknown as { __cwMap: { getZoom(): number } }).__cwMap.getZoom())).toBeGreaterThan(8);
  await expect(page.getByTestId("read-hint")).toHaveCount(0);
});

test("the empty search offers the common places; a station's panel says when it was sampled and offers its data", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=none`);
  await page.getByTestId("place-search-input").click();
  const list = page.getByTestId("place-search-list");
  await expect(list).toContainText("Monterey");
  await expect(list).toContainText("Santa Cruz Wharf");
  await list.getByText("Santa Cruz Wharf").click();
  await expect(page.getByTestId("inspect-title")).toHaveText("Santa Cruz Wharf");
  await expect(page.getByTestId("inspect-panel")).toContainText(/last sampled \w{3} \d+/);
  await expect(page.getByTestId("station-csv")).toHaveAttribute("href", /erddap\.sccoos\.org\/erddap\/tabledap\/HABs-SantaCruzWharf\.csv/);
  await expect(page.getByTestId("station-research")).toHaveAttribute("href", "/research?station=HABs-SantaCruzWharf");
});

test("the briefing: what to do comes right after what matters and what changed", async ({ page }) => {
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.goto(`${OK}/coast?place=port-201&for=commercial&species=dungeness`);
  const order = await page.locator('[data-testid^="brief-"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
  expect(order.slice(0, 4)).toEqual(["brief-glance", "brief-now", "brief-changes", "brief-checks"]);
  expect(order.indexOf("brief-checks")).toBeLessThan(order.indexOf("brief-measured"));
});

test("phone: a place picked in search lands in the map strip above the sheet, not behind it", async ({ browser }) => {
  const ctx = await browser.newContext(phone);
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  await page.getByTestId("place-search-input").click();
  await page.getByTestId("place-search-list").getByText("Moss Landing", { exact: true }).click();
  await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "half");
  await page.waitForFunction(() => { const m = (window as unknown as { __cwMap: { isMoving(): boolean } }).__cwMap; return !m.isMoving(); });
  await page.waitForTimeout(700);
  const sheetTop = (await page.getByTestId("mobile-sheet").boundingBox())!.y;
  const y = await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: { project(p: number[]): { y: number } } }).__cwMap;
    return m.project([-121.7867, 36.8044]).y;
  });
  // centred in the visible strip (between the search bar's band, ~128 px, and the sheet), not at its edge
  expect(Math.abs(y - (128 + sheetTop) / 2)).toBeLessThan(60);
  await ctx.close();
});
