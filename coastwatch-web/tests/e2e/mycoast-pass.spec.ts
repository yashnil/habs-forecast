import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * My Coast after the v4 final pass: "Before you go" stays open right after the answer; the evidence,
 * official notices and gaps below it are folded, each with its count and one line; the title is the
 * place control. Icons are always sized; the phone sheet reads cool chart paper, never cream.
 */
const NOW = "2026-10-13T06:00:00Z";

test("v7: below what matters, every section is a folded note with a count and a line; opening one shows all of it", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("matters")).toBeVisible();
  for (const id of ["sea", "seasons", "checks", "evidence", "official", "unknown", "map"]) await expect(page.getByTestId(`brief-${id}-toggle`)).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("checks")).toBeHidden();
  await page.getByTestId("brief-checks-toggle").click();
  await expect(page.getByTestId("checks")).toBeVisible();
  for (const id of ["evidence", "official", "unknown"]) {
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
  // the landing's search icon too
  await page.goto(`${OK}/coast`);
  const w = await page.getByTestId("coast-place-search").locator("svg").first().evaluate((e) => e.getBoundingClientRect().width);
  expect(w).toBe(20);
  await ctx.close();
});

test("the phone sheet over the map is the instrument's own dark surface (v7), never cream or a white card", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(`${OK}/?sel=harbor:550&layer=forecast`);
  const sheet = page.getByTestId("mobile-sheet");
  await expect(sheet).toBeVisible({ timeout: 20_000 });
  // the env-instrument surface token (#0c1921)
  expect(await sheet.evaluate((e) => getComputedStyle(e).backgroundColor)).toBe("rgb(12, 25, 33)");
  await ctx.close();
});

test("v6 P0 4: offline three days later, the saved copy is dated 'as saved' and makes no 'Tonight' or 'Today' claim", async ({ browser }) => {
  test.setTimeout(180_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await page.getByTestId("pack-open").click();
  await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId("pack-download").click();
  await expect(page.getByTestId("pack-status")).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
  await page.reload();
  // online, on the day it was issued, the glance reads Monday night's NWS period
  await expect(page.getByTestId("glance-sea-link")).toContainText("(NWS), Tonight");

  await ctx.route("**/*", (r) => r.abort("internetdisconnected"));
  await ctx.setOffline(true);
  await page.clock.setFixedTime(new Date(Date.parse(NOW) + 3 * 86_400_000 + 3_600_000));
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey", { timeout: 20_000 });
  await expect(page.getByTestId("matters-heading")).toContainText("What mattered when saved");
  const glance = page.getByTestId("brief-glance");
  await expect(glance.locator("#glance-h")).toContainText(/as saved \w{3} \d{1,2}, \d{1,2}:\d{2} [AP]M/);
  // no saved NWS period starts on this day: said, never the first saved period
  await expect(page.getByTestId("glance-sea-link")).toHaveText("No NWS period saved for this day");
  await expect(page.getByTestId("sea-no-period")).toHaveText("No NWS period saved for this day.");
  // the water temperature carries its own measured date and where
  await expect(page.getByTestId("glance-water")).toContainText(/measured Oct 1\d, .*\((CO-OPS|NDBC)/);
  // the saved periods are dated by their own day, never "Tonight"
  await page.getByTestId("brief-sea-toggle").click();
  await page.getByTestId("sea-periods-more").locator("summary").first().click();
  await expect(page.getByTestId("sea-period").first()).toContainText(/Mon, Oct 12, night/);
  for (const id of ["brief-glance", "brief-sea", "brief-now"]) expect(await page.getByTestId(id).innerText(), id).not.toMatch(/\b(tonight|today)\b(?!'s price)/i); // "Not today's price" is the dock-value caveat, not a claim
  await ctx.unrouteAll({ behavior: "ignoreErrors" });
  await ctx.setOffline(false);
  await ctx.close();
});
