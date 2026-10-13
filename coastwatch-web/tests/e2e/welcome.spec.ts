import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "./ports";

/** The front door: "Explore California's coast" and three paths; defaults only, never facts. */
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
  test("plan a fishing trip: choose the kind of fishing, go Back, pick waters, land in My Coast's trip plan", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    const card = page.getByTestId("intro-card");
    await expect(card).toHaveAttribute("data-step", "1");
    await expect(card).toContainText("Explore California's coast");
    await expect(card).toContainText("not an official source");
    for (const p of ["fish", "conditions", "research"]) await expect(page.getByTestId(`welcome-${p}`)).toBeVisible();
    await page.getByTestId("welcome-fish").click();
    await expect(card).toHaveAttribute("data-step", "2");
    await page.getByTestId("intro-back").click();
    await expect(card).toHaveAttribute("data-step", "1");
    expect(await stored(page, "cw-purpose-v1")).toBeNull(); // looking is not choosing
    await page.getByTestId("welcome-fish").click();
    await page.getByTestId("intro-purpose-commercial").click();
    await page.getByTestId("place-input").fill("Santa Cruz");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/coast\?place=port-593&for=commercial$/);
    await expect(page.getByTestId("journey")).toBeVisible();
    expect(JSON.parse((await stored(page, "cw-purpose-v1"))!)).toEqual({ purpose: "commercial" });
    // a later plain visit: no welcome, one link back to those waters
    await page.goto(`${OK}/`);
    await expect(page.getByTestId("purpose-return")).toHaveAttribute("href", "/coast?place=port-593&for=commercial");
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await ctx.close();
  });

  test("Skip never forces a choice, and the welcome does not come back", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-dismiss").click();
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    expect(await stored(page, "cw-purpose-v1")).toBeNull();
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await ctx.close();
  });

  test("explore coastal conditions: the map takes over, with one line on how to use it; the tour stays one click away", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("welcome-conditions").click();
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await expect(page.getByTestId("map-hint")).toContainText("Tap the water");
    await expect(page.getByTestId("layer-switch")).toBeVisible();
    expect(JSON.parse((await stored(page, "cw-purpose-v1"))!).purpose).toBe("exploring");
    await page.getByTestId("tour-open").click();
    await expect(page.getByTestId("tour")).toBeVisible();
    await ctx.close();
  });

  test("analyze scientific data: Research presentation, station analysis; the same notices are listed for everyone", async ({ browser }) => {
    const before = await fresh(browser, DESKTOP, `${OK}/coast?place=port-593`);
    const ids = async (p: Page) => (await p.locator('[data-testid^="coast-notice-"][data-tier]').evaluateAll((els) => els.map((e) => `${e.getAttribute("data-testid")}:${e.getAttribute("data-tier")}`))).sort();
    await expect(before.page.getByTestId("coast-official")).toBeAttached();
    const noPurpose = await ids(before.page);
    await before.ctx.close();

    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("welcome-research").click();
    await expect(page).toHaveURL(/\/bloom/);
    expect(await stored(page, "cw-mode-v1")).toBe("research");
    await expect(page.getByTestId("mode-switch-navy").getByTestId("mode-research")).toHaveAttribute("aria-pressed", "true");
    await page.goto(`${OK}/coast?place=port-593`);
    // the sections appear once, already in the researcher's order
    await expect(page.getByTestId("coast-official")).toBeVisible();
    const order = await page.locator('section[data-testid^="coast-"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
    expect(order.slice(0, 4)).toEqual(["coast-measured", "coast-model", "coast-observed", "coast-official"]);
    await expect(page.getByTestId("brief-official")).toBeVisible(); // official information stays at the top
    expect(await ids(page)).toEqual(noPurpose);
    await ctx.close();
  });

  test("a shared link takes precedence over the welcome and over the saved purpose", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP, `${OK}/?region=monterey_bay&port=593`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await page.evaluate(() => localStorage.setItem("cw-purpose-v1", JSON.stringify({ purpose: "commercial" })));
    await page.goto(`${OK}/coast?place=port-201&for=recreational`);
    // a link naming the activity opens on what officially applies; the first chapter keeps the choice
    await expect(page.getByTestId("chapter-target")).toContainText("Recreational");
    await page.getByTestId("chapter-head-target").click();
    await expect(page.getByTestId("coast-for-recreational")).toHaveAttribute("aria-checked", "true");
    await page.goto(`${OK}/coast?place=port-201`);
    await page.getByTestId("chapter-head-target").click();
    await expect(page.getByTestId("coast-for-commercial")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    await ctx.close();
  });

  test("purpose can be changed or cleared later, and the welcome shown again", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-dismiss").click();
    await page.goto(`${OK}/about`);
    await page.getByTestId("purpose-select").selectOption("recreational:shellfish");
    await page.goto(`${OK}/coast?place=site-clam-beach`);
    await expect(page.getByTestId("coast-for-shellfish")).toHaveAttribute("aria-checked", "true");
    await page.getByTestId("purpose-select").selectOption("none");
    await expect(page.getByTestId("coast-for-all")).toHaveAttribute("aria-checked", "true");
    await page.goto(`${OK}/about`);
    await page.getByTestId("purpose-reset").click();
    await expect(page).toHaveURL(new RegExp(`^${OK}/(\\?.*)?$`));
    await expect(page.getByTestId("intro-card")).toBeVisible({ timeout: 15000 });
    await ctx.close();
  });

  test("phone: a shellfish harvester picks a CDPH site and lands on it; the welcome fits the screen", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, PHONE);
    const card = page.getByTestId("intro-card");
    await expect(card).toBeVisible();
    const box = (await card.boundingBox())!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390);
    await page.getByTestId("welcome-fish").click();
    await page.getByTestId("intro-purpose-shellfish").click();
    await page.getByTestId("place-input").fill("Clam");
    await page.getByTestId("place-option-site-clam-beach").click();
    await expect(page).toHaveURL(/\/coast\?place=site-clam-beach&for=shellfish$/);
    await page.getByTestId("chapter-head-official").click();
    await expect(page.getByTestId("brief-official")).toBeVisible();
    await ctx.close();
  });

  test("small phones: the three paths are reachable and the notices stay one tap away", async ({ browser }) => {
    for (const viewport of [{ width: 360, height: 740 }, { width: 320, height: 640 }]) {
      const { ctx, page } = await fresh(browser, { ...PHONE, viewport });
      await expect(page.getByTestId("intro-card")).toBeVisible();
      for (const p of ["fish", "conditions", "research"]) {
        await page.getByTestId(`welcome-${p}`).scrollIntoViewIfNeeded();
        await expect(page.getByTestId(`welcome-${p}`)).toBeInViewport();
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
      await page.getByTestId("tab-notices").click();
      await expect(page.getByTestId("official-drawer")).toBeVisible();
      await ctx.close();
    }
  });

  for (const [name, opts] of [["desktop", DESKTOP], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing in either step of the welcome`, async ({ browser }) => {
      const { ctx, page } = await fresh(browser, opts);
      // measure the settled welcome: mid fade-in, text contrast is not what a reader sees
      const settled = () => page.getByTestId("intro-card").evaluate((el) => Promise.all(el.getAnimations({ subtree: true }).map((a) => a.finished)));
      await expect(page.getByTestId("intro-card")).toBeVisible();
      await page.waitForTimeout(1600);
      await settled();
      let r = await new AxeBuilder({ page }).include('[data-testid="intro-card"]').analyze();
      expect(r.violations.map((v) => v.id)).toEqual([]);
      await page.getByTestId("welcome-fish").click();
      await page.waitForTimeout(800);
      await settled();
      r = await new AxeBuilder({ page }).include('[data-testid="intro-card"]').analyze();
      expect(r.violations.map((v) => v.id)).toEqual([]);
      await ctx.close();
    });
  }
});
