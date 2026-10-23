import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * The six usability issues the final scripted walkthrough left open (M2, M4–M8), each held here:
 * Research's map follows a new station; one station's latest HAB sample is dated the same way, with its
 * year, everywhere; what the map shows and when is above the fold; on a phone the layers stay reachable
 * with a place open and a statewide notice count says so; every listed notice carries its date; and
 * Research's map tools are one tap from the top on a phone.
 */
const NOW = "2026-10-08T20:00:00Z";
type M = { loaded(): boolean; getCenter(): { lng: number; lat: number } };
const mapReady = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());
const phone = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

test("M7: on a phone, Research's map tools are one tap from the top of the page", async ({ browser }) => {
  const ctx = await browser.newContext(phone);
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  const jump = page.getByTestId("research-jump-r-tools");
  await expect(jump).toBeInViewport();
  await expect(page.getByTestId("research-grid")).not.toBeInViewport();
  await jump.click();
  await expect(page.getByTestId("research-grid")).toBeInViewport();
  await expect(page.getByTestId("tool-cell")).toBeInViewport();
  await ctx.close();
});

test("M2: choosing another station moves Research's map to it and clears the old selection", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await page.getByTestId("research-grid").scrollIntoViewIfNeeded();
  await mapReady(page);
  const before = await page.evaluate(() => (window as unknown as { __cwMap: M }).__cwMap.getCenter());
  expect(before.lat).toBeGreaterThan(36.5);
  await page.getByTestId("research-stations").getByRole("button", { name: /Scripps Pier/ }).click();
  await expect(page.getByTestId("research-title")).toContainText("Scripps Pier");
  await expect.poll(async () => (await page.evaluate(() => (window as unknown as { __cwMap: M }).__cwMap.getCenter())).lat, { timeout: 5000 }).toBeLessThan(33.5);
});

test("M4: a station's latest HAB sample is the same date, with its year, in Research and in Explore", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  const inResearch = (await page.getByTestId("research-stations").getByRole("button", { name: /Monterey Wharf/ }).innerText()).match(/last HAB sample [A-Z][a-z]{2} \d+(, \d{4})?/)?.[0];
  // Monterey Wharf's newest toxin or cell sample is from 2025: the year is said, never read as this year's
  expect(inResearch).toMatch(/, 2025$/);
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await mapReady(page);
  const row = page.getByTestId("harbor-stations").locator("li", { hasText: "Monterey Wharf" });
  await expect(row.getByTestId("harbor-station-last")).toHaveText(inResearch!);
});

test("M5: comparing currents with chlorophyll, both times are above the fold on a laptop screen", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=currents&chl=1`);
  await mapReady(page);
  for (const id of ["stamp-currents", "stamp-satellite"]) await expect(page.getByTestId("layer-deck").getByTestId(id).first()).toBeInViewport();
});

test("M6: on a phone with a harbour open the layers are one tap away, and a statewide notice count says it is statewide", async ({ browser }) => {
  const ctx = await browser.newContext(phone);
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await mapReady(page);
  await expect(page.getByTestId("notices-here")).toBeVisible();
  const chips = page.getByTestId("sheet-layer-chips");
  await expect(chips).toBeVisible();
  await chips.getByTestId("chip-currents").click();
  await expect(page).toHaveURL(/layer=currents/);
  await expect(page).toHaveURL(/sel=harbor(:|%3A)550/);
  await expect(page.getByTestId("notices-here")).toBeVisible();
  await page.goto(`${OK}/?layer=forecast`);
  await mapReady(page);
  await expect(page.getByTestId("notices-statewide")).toHaveText("in CA");
  await expect(page.getByTestId("notices-here")).toHaveCount(0);
  await ctx.close();
});

test("M8: when several notices cover a place, each listed notice carries the date it took effect", async ({ request }) => {
  const g = await (await request.get(`${OK}/api/evidence?place=port-550`)).json();
  const n = g.claims.find((c: { id: string }) => c.id === "notices");
  expect(n.headline).toMatch(/^\d+ notices cover these waters/);
  for (const part of (n.detail as string).split(" · ").filter((p: string) => !/more, all in Notices/.test(p))) expect(part).toMatch(/\(since [A-Z][a-z]{2} \d+\)$/);
});
