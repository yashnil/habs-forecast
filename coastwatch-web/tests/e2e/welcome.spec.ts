import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "./ports";

/** The first-visit welcome: optional purpose, then the waters or a starting point; defaults only, never facts. */
const NOW = "2026-10-08T20:00:00Z";
const FIRST = { cookies: [], origins: [] };
const DESKTOP = { viewport: { width: 1440, height: 900 } };
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

async function fresh(browser: Browser, opts: object, url = `${OK}/`) {
  const ctx = await browser.newContext({ ...opts, storageState: FIRST });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
  return { ctx, page };
}
const stored = (page: Page, k: string) => page.evaluate((key) => localStorage.getItem(key), k);

test.describe("welcome", () => {
  test("commercial: choose, go Back, choose again, pick waters, land in My Coast filtered for commercial fishing", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    const card = page.getByTestId("intro-card");
    await expect(card).toHaveAttribute("data-step", "1");
    await expect(card).toContainText("not an official source");
    await page.getByTestId("intro-purpose-commercial").click();
    await expect(page.getByTestId("intro-progress")).toContainText("2 of 2");
    await page.getByTestId("intro-back").click();
    await expect(card).toHaveAttribute("data-step", "1");
    expect(await stored(page, "cw-purpose-v1")).toBeNull(); // looking is not choosing
    await page.getByTestId("intro-purpose-commercial").click();
    await page.getByTestId("place-input").fill("Santa Cruz");
    await page.keyboard.press("Enter");
    await page.getByTestId("intro-to-coast").click();
    await expect(page).toHaveURL(/\/coast\?place=port-593&for=commercial$/);
    await expect(page.getByTestId("coast-for-commercial")).toHaveAttribute("aria-pressed", "true");
    expect(JSON.parse((await stored(page, "cw-purpose-v1"))!)).toEqual({ purpose: "commercial" });
    // the map on a later plain visit: no welcome, one link back to those waters
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("purpose-return")).toHaveAttribute("href", "/coast?place=port-593&for=commercial");
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await ctx.close();
  });

  test("Skip and the X never force a choice, and the welcome does not come back", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-dismiss").click();
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    expect(await stored(page, "cw-purpose-v1")).toBeNull();
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await ctx.close();
    const second = await fresh(browser, DESKTOP);
    await second.page.getByTestId("intro-purpose-ports").click();
    await second.page.getByTestId("intro-close").click();
    await expect(second.page.getByTestId("intro-card")).toHaveCount(0);
    expect(await stored(second.page, "cw-purpose-v1")).toBeNull();
    await second.ctx.close();
  });

  test("just exploring starts the existing guided tour (no second introduction)", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-tour").click();
    await expect(page.getByTestId("tour")).toBeVisible();
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await expect(page.getByTestId("tour-progress")).toHaveText("Step 1 of 9");
    expect(JSON.parse((await stored(page, "cw-purpose-v1"))!).purpose).toBe("exploring");
    await page.getByTestId("tour-close").click();
    await page.getByTestId("tour-open").click(); // the tour can always be reopened
    await expect(page.getByTestId("tour")).toBeVisible();
    await ctx.close();
  });

  test("research puts measurements and forecast coverage first; the same notices are listed for everyone", async ({ browser }) => {
    const before = await fresh(browser, DESKTOP, `${OK}/coast?place=port-593`);
    const ids = async (p: Page) => (await p.locator('[data-testid^="coast-notice-"][data-tier]').evaluateAll((els) => els.map((e) => `${e.getAttribute("data-testid")}:${e.getAttribute("data-tier")}`))).sort();
    await expect(before.page.getByTestId("coast-official")).toBeVisible();
    const noPurpose = await ids(before.page);
    await before.ctx.close();

    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-purpose-research").click();
    await page.getByTestId("intro-research-coast").click();
    await expect(page).toHaveURL(/\/coast$/);
    await page.goto(`${OK}/coast?place=port-593`);
    const order = await page.locator('section[data-testid^="coast-"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
    expect(order.slice(0, 4)).toEqual(["coast-measured", "coast-model", "coast-observed", "coast-official"]);
    await expect(page.getByTestId("glance-official")).toBeVisible(); // official information stays at the top
    expect(await ids(page)).toEqual(noPurpose);
    await ctx.close();
  });

  test("a shared link takes precedence over the welcome and over the saved purpose", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP, `${OK}/?region=monterey_bay&port=593`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await page.evaluate(() => localStorage.setItem("cw-purpose-v1", JSON.stringify({ purpose: "commercial" })));
    await page.goto(`${OK}/coast?place=port-201&for=recreational`);
    await expect(page.getByTestId("coast-for-recreational")).toHaveAttribute("aria-pressed", "true");
    await page.goto(`${OK}/coast?place=port-201`);
    await expect(page.getByTestId("coast-for-commercial")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    await ctx.close();
  });

  test("purpose can be changed or cleared later, and the welcome shown again", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-dismiss").click();
    await page.goto(`${OK}/about`);
    await page.getByTestId("purpose-select").selectOption("recreational:shellfish");
    await page.goto(`${OK}/coast?place=site-clam-beach`);
    await expect(page.getByTestId("coast-for-shellfish")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("purpose-select").selectOption("none");
    await expect(page.getByTestId("coast-for-all")).toHaveAttribute("aria-pressed", "true");
    await page.goto(`${OK}/about`);
    await page.getByTestId("purpose-reset").click();
    // the map then writes its view into the address; the welcome is back either way
    await expect(page).toHaveURL(new RegExp(`^${OK}/(\\?.*)?$`));
    await expect(page.getByTestId("intro-card")).toBeVisible({ timeout: 15000 });
    await ctx.close();
  });

  test("phone: shellfish harvester picks a CDPH site and lands on it; the card fits the screen", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, PHONE);
    const card = page.getByTestId("intro-card");
    await expect(card).toBeVisible();
    const box = (await card.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390);
    await page.getByTestId("intro-purpose-recreational").click();
    await page.getByTestId("intro-harvest-shellfish").click();
    await page.getByTestId("place-input").fill("Clam");
    await page.getByTestId("place-option-site-clam-beach").click();
    await page.getByTestId("intro-to-coast").click();
    await expect(page).toHaveURL(/\/coast\?place=site-clam-beach&for=shellfish$/);
    await expect(page.getByTestId("coast-glance")).toBeVisible();
    await ctx.close();
  });

  for (const [name, opts] of [["desktop", DESKTOP], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing in either step of the welcome`, async ({ browser }) => {
      const { ctx, page } = await fresh(browser, opts);
      await expect(page.getByTestId("intro-card")).toBeVisible();
      let r = await new AxeBuilder({ page }).include('[data-testid="intro-card"]').analyze();
      expect(r.violations.map((v) => v.id)).toEqual([]);
      await page.getByTestId("intro-purpose-commercial").click();
      r = await new AxeBuilder({ page }).include('[data-testid="intro-card"]').analyze();
      expect(r.violations.map((v) => v.id)).toEqual([]);
      await ctx.close();
    });
  }
});
