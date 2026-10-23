import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The launch screen (src/v4/shell/Splash.tsx): once per browser session, never for automated browsers
 * unless asked (every other suite runs without it). Forced on here: it appears over the page, the mark
 * draws, it steps aside within the 2.5 s cap, and the page underneath is usable at once.
 */
const force = () => localStorage.setItem("nx-splash-force-v1", "1");

test("the launch screen shows on a session's first load, steps aside, and the page is usable", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto(`${OK}/?layer=forecast`);
  const splash = page.getByTestId("splash");
  await expect(splash).toBeVisible();
  // the page renders underneath at the same time (never blocked by the launch screen)
  await expect(page.getByTestId("explore")).toBeAttached();
  await expect(splash.getByRole("status")).toHaveText("Loading Neritic");
  // the mark is animating (its own CSS, no script needed)
  expect(await page.evaluate(() => document.getAnimations().length)).toBeGreaterThan(0);
  await expect(splash).toBeHidden({ timeout: 4000 });
  expect(Date.now() - t0).toBeLessThan(6000);
  // usable: the first click lands on the page
  await page.getByTestId("more-menu").click();
  await expect(page.getByRole("navigation", { name: "More pages" })).toBeVisible();
  // not again in the same session (a reload, a navigation)
  await page.reload();
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/);
  await expect(splash).toBeHidden();
  await ctx.close();
});

test("a reading page opens without the launch screen: its content is in the first HTML and is never hidden", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  expect(await page.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
  await expect(page.getByTestId("splash")).toBeHidden();
  await ctx.close();
});

test("Explore: the launch screen leaves within its cap and takes no clicks while fading", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await expect(page.getByTestId("splash")).toBeVisible();
  // the moment it starts to leave: after the mark's first beat, before the cap, and already click-through
  const out = await page.evaluate(
    () =>
      new Promise<{ at: number; pe: string }>((done) => {
        const d = document.documentElement;
        const check = () => d.classList.contains("nx-splash-out") && (done({ at: performance.now(), pe: getComputedStyle(document.querySelector(".nx-splash")!).pointerEvents }), true);
        if (check()) return;
        const mo = new MutationObserver(() => check() && mo.disconnect());
        mo.observe(d, { attributes: true, attributeFilter: ["class"] });
      }),
  );
  expect(out.pe).toBe("none");
  expect(out.at).toBeGreaterThanOrEqual(700);
  expect(out.at).toBeLessThanOrEqual(2600);
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 2000 });
  await ctx.close();
});

test("reduced motion: a still mark that fades", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  await page.goto(`${OK}/`);
  await expect(page.getByTestId("splash")).toBeVisible();
  // (the overlay's own 2.5 s cap is a fade; nothing inside it draws or moves)
  const moving = await page.evaluate(() => document.getAnimations().filter((a) => ((a.effect as KeyframeEffect | null)?.target as Element | null)?.parentElement?.closest(".nx-splash")).length);
  expect(moving).toBe(0);
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 4000 });
  await ctx.close();
});

test("automated browsers do not see it unless they ask", async ({ page }) => {
  await page.goto(`${OK}/about`);
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/);
  await expect(page.getByTestId("splash")).toBeHidden();
});
