import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "./ports";

/** The guided tour of the Ocean Map: real controls, never trapping, never in front of the official notices. */
const NOW = "2026-10-08T20:00:00Z";
const FIRST_VISIT = { cookies: [], origins: [] };
const DESKTOP = { viewport: { width: 1440, height: 900 } };
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

async function fresh(browser: Browser, opts: object, url = OK) {
  const ctx = await browser.newContext({ ...opts, storageState: FIRST_VISIT });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
  return { ctx, page };
}

async function inViewport(page: Page, testid: string) {
  const b = (await page.getByTestId(testid).boundingBox())!;
  const vp = page.viewportSize()!;
  return b.x >= 0 && b.y >= 0 && b.x + b.width <= vp.width && b.y + b.height <= vp.height;
}

const progress = (page: Page) => page.getByTestId("tour-progress");

test.describe("guided tour", () => {
  test("starts from the first-visit card and walks all nine steps with the real controls", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-tour").click();
    const tour = page.getByTestId("tour");
    await expect(tour).toBeVisible();
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await expect(progress(page)).toHaveText("Step 1 of 9");
    await expect(page.locator("#tour-h")).toBeFocused();
    await expect(tour).toContainText("not an official source");

    const next = page.getByTestId("tour-next");
    await next.click();
    await expect(tour).toHaveAttribute("data-step", "place");
    await expect(page.getByTestId("tour-highlight")).toBeVisible();

    await next.click();
    await expect(tour).toHaveAttribute("data-step", "forecast");
    await expect(page.getByTestId("group-forecast").first()).toHaveAttribute("aria-selected", "true");
    await expect(tour).toContainText("a low probability does not mean an area is safe");

    await next.click();
    await expect(tour).toHaveAttribute("data-step", "satellite");
    await expect(page.getByTestId("group-satellite").first()).toHaveAttribute("aria-selected", "true");
    await expect(tour).toContainText("Chlorophyll is not toxin");

    await next.click();
    await expect(tour).toHaveAttribute("data-step", "gaps");
    await expect(page.getByTestId("group-forecast").first()).toHaveAttribute("aria-selected", "true");
    await expect(tour).toContainText("never fills these in");

    await next.click();
    await expect(tour).toHaveAttribute("data-step", "port");
    await expect(page.getByTestId("detail-panel")).toBeVisible();

    // Back returns to the previous step
    await page.getByTestId("tour-back").click();
    await expect(tour).toHaveAttribute("data-step", "gaps");
    await next.click();
    await expect(tour).toHaveAttribute("data-step", "port");

    await next.click();
    await expect(tour).toHaveAttribute("data-step", "official");
    await next.click();
    await expect(tour).toHaveAttribute("data-step", "pages");
    await next.click();
    await expect(tour).toHaveAttribute("data-step", "limits");
    await expect(progress(page)).toHaveText("Step 9 of 9");
    await expect(tour).toContainText("does not say that seafood is safe or that an area is open");
    await expect(page.getByTestId("tour-about")).toHaveAttribute("href", "/about");
    await expect(page.getByTestId("tour-skip")).toHaveCount(0);
    await next.click(); // Finish
    await expect(tour).toHaveCount(0);

    // seen: neither the card nor the tour comes back on the next visit
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await page.waitForTimeout(300);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await expect(page.getByTestId("tour")).toHaveCount(0);
    await ctx.close();
  });

  test("never traps: the notices stay one click away, and Escape or X closes it at any step", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP);
    await page.getByTestId("intro-tour").click();
    await page.getByTestId("tour-next").click();
    // the masthead's notices pill is still usable while the tour is open
    await page.getByTestId("official-pill").click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
    await page.keyboard.press("Escape"); // closes the drawer, not the tour
    await expect(page.getByTestId("official-drawer")).toHaveCount(0);
    await expect(page.getByTestId("tour")).toBeVisible();
    // Tab moves on out of the card (no focus trap)
    await page.locator("#tour-h").focus();
    for (let k = 0; k < 8; k++) await page.keyboard.press("Tab");
    expect(await page.evaluate(() => !!document.activeElement?.closest('[data-testid="tour"]'))).toBe(false);
    // Escape closes the tour
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("tour")).toHaveCount(0);
    // reopened from the rail, the X closes it; focus returns to the opener
    await page.getByTestId("tour-open").click();
    await expect(progress(page)).toHaveText("Step 1 of 9");
    await page.getByTestId("tour-next").click();
    await page.getByTestId("tour-next").click();
    await page.getByTestId("tour-close").click();
    await expect(page.getByTestId("tour")).toHaveCount(0);
    await expect(page.getByTestId("tour-open")).toBeFocused();
    await ctx.close();
  });

  test("the official step opens the real notices list", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP, `${OK}/?tour=1`);
    await expect(page.getByTestId("tour")).toBeVisible();
    for (let k = 0; k < 6; k++) await page.getByTestId("tour-next").click();
    await expect(page.getByTestId("tour")).toHaveAttribute("data-step", "official");
    await page.getByTestId("tour-open-notices").click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
    await ctx.close();
  });

  test("Skip ends it and remembers; About links to it", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, DESKTOP, `${OK}/about`);
    await expect(page.getByTestId("about-tour")).toHaveAttribute("href", "/?tour=1");
    await page.getByTestId("about-tour").click();
    await expect(page.getByTestId("tour")).toBeVisible();
    await page.getByTestId("tour-skip").click();
    await expect(page.getByTestId("tour")).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem("cw-intro-dismissed-v1"))).toBe("1");
    await ctx.close();
  });

  test("phone: docked under the place button, every step in view, tabs still usable", async ({ browser }) => {
    const { ctx, page } = await fresh(browser, PHONE);
    await page.getByTestId("intro-tour").click();
    for (let k = 1; k <= 9; k++) {
      await expect(progress(page)).toHaveText(`Step ${k} of 9`);
      await page.waitForTimeout(250);
      expect(await inViewport(page, "tour"), `step ${k}`).toBe(true);
      const tour = (await page.getByTestId("tour").boundingBox())!;
      const place = (await page.getByTestId("place-button").boundingBox())!;
      expect(tour.y).toBeGreaterThanOrEqual(place.y + place.height);
      if (k < 9) await page.getByTestId("tour-next").click();
    }
    await expect(page.getByTestId("tab-notices")).toBeVisible();
    await page.getByTestId("tab-notices").click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
    await page.keyboard.press("Escape");
    await page.getByTestId("tour-close").click();
    await expect(page.getByTestId("tour")).toHaveCount(0);
    // reopened from the help button beside the place button
    await page.getByTestId("tour-open").click();
    await expect(progress(page)).toHaveText("Step 1 of 9");
    await ctx.close();
  });

  test("reduced motion: the highlight and the card do not animate", async ({ browser }) => {
    const ctx = await browser.newContext({ ...DESKTOP, storageState: FIRST_VISIT, reducedMotion: "reduce" });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?tour=1`);
    await page.getByTestId("tour-next").click();
    await expect(page.getByTestId("tour-highlight")).toBeVisible();
    for (const id of ["tour", "tour-highlight"]) expect(await page.getByTestId(id).getAttribute("class")).not.toContain("transition");
    await ctx.close();
  });

  for (const [name, opts] of [["desktop", DESKTOP], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing with the tour open`, async ({ browser }) => {
      const { ctx, page } = await fresh(browser, opts, `${OK}/?tour=1`);
      await expect(page.getByTestId("tour")).toBeVisible();
      await page.getByTestId("tour-next").click();
      const r = await new AxeBuilder({ page }).include('[data-testid="tour"]').analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
      await ctx.close();
    });
  }
});
