import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * My Coast after the v4 final pass: "Before you go" stays open right after the answer; the evidence,
 * official notices and gaps below it are folded, each with its count and one line; the title is the
 * place control. Icons are always sized; the phone sheet reads cool chart paper, never cream.
 */
const NOW = "2026-10-13T06:00:00Z";

test("below 'Before you go' the sections are folded with a count and a line; opening one shows all of it", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("matters")).toBeVisible();
  await expect(page.getByTestId("checks")).toBeVisible();
  for (const id of ["measured", "official", "unknown"]) {
    const t = page.getByTestId(`brief-${id}-toggle`);
    await expect(t).toHaveAttribute("aria-expanded", "false");
    await expect(t).toContainText(/\d/);
  }
  await expect(page.getByTestId("unknowns")).toBeHidden();
  await page.getByTestId("brief-unknown-toggle").click();
  await expect(page.getByTestId("unknowns")).toBeVisible();
  await expect(page.getByTestId("unknowns")).toContainText("Toxin in seafood is not measured by Neritic");
  await page.getByTestId("brief-official-toggle").click();
  await expect(page.getByTestId("brief-official")).toContainText("Not verified");
  // nothing removed: a station's sample against its own record is said once, in the statement it supports
  const record = page.locator('[data-testid="brief-context"] [data-node^="record:"]');
  const said = await page.locator('[data-testid="matters"]').innerText();
  if (/Compared with earlier years/.test(said)) await expect(record).toHaveCount(0);
  const axe = await new AxeBuilder({ page }).include('[data-testid="my-coast"] main').analyze();
  expect(axe.violations.map((v) => v.id)).toEqual([]);
});

test("the title is the place control on a phone: one button instead of three pills; icons are sized", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-place-bar")).toHaveCount(0);
  const btn = page.getByTestId("coast-place-button");
  await btn.click();
  await expect(page.getByTestId("coast-place-picker")).toBeFocused();
  await expect(page.getByTestId("coast-map-jump")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("coast-place-bar")).toHaveCount(0);
  // the answer (v5: the at-a-glance block, then "What matters") still starts within the first screen
  expect((await page.getByTestId("brief-glance").boundingBox())!.y).toBeLessThan(560);
  // every v4 icon has a size of its own (none at the SVG default of 300 x 150)
  const big = await page.locator('svg[viewBox="0 0 24 24"]').evaluateAll((els) => els.filter((e) => e.getBoundingClientRect().width > 64).length);
  expect(big).toBe(0);
  // the landing's chevrons too
  await page.goto(`${OK}/coast`);
  const w = await page.getByTestId("coast-start-port-550").locator("svg").evaluate((e) => e.getBoundingClientRect().width);
  expect(w).toBe(20);
  await ctx.close();
});

test("the phone sheet over the map is cool chart paper, not cream", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(`${OK}/?sel=harbor:550&layer=forecast`);
  const sheet = page.getByTestId("mobile-sheet");
  await expect(sheet).toBeVisible({ timeout: 20_000 });
  expect(await sheet.evaluate((e) => getComputedStyle(e).backgroundColor)).toBe("rgb(255, 255, 255)");
  await ctx.close();
});
