import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The launch screen (src/v4/shell/Splash.tsx): a device's first visit ever only (v6), never on a saved
 * copy reopened from a field pack, never for a returning reader, never for automated browsers unless
 * asked (every other suite runs without it). Forced on here (each test opens a fresh context, so a first
 * visit): it appears over the page, the canyon and the mark draw, it holds at least 1.8 s and steps aside
 * by the 3 s cap, a tap, click or key skips it at once, and the page underneath is usable at once.
 */
const force = () => localStorage.setItem("nx-splash-force-v1", "1");

test("the launch screen shows on a device's first visit, steps aside, and the page is usable", async ({ browser }) => {
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
  await expect(splash).toBeHidden({ timeout: 5000 });
  expect(Date.now() - t0).toBeLessThan(7000);
  // usable: the first click lands on the page
  await page.getByTestId("more-menu").click();
  await expect(page.getByRole("navigation", { name: "More pages" })).toBeVisible();
  // never again on this device: a reload, and a new session (a new tab: its own sessionStorage)
  await page.reload();
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/);
  await expect(splash).toBeHidden();
  const again = await ctx.newPage();
  await again.goto(`${OK}/?layer=forecast`, { waitUntil: "commit" });
  await again.waitForLoadState("domcontentloaded");
  expect(await again.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
  expect(await again.evaluate(() => localStorage.getItem("nx-splash-seen-v2"))).toBe("1");
  await ctx.close();
});

test("never on a saved copy reopened from a field pack, and never after the first visit, whatever page it opened", async ({ browser }) => {
  // a saved copy: the service worker's mark is in the page before the launch screen's script runs
  const saved = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await saved.addInitScript(force);
  await saved.addInitScript(() => {
    (self as unknown as { __NERITIC_SAVED__: unknown }).__NERITIC_SAVED__ = { at: "2026-10-13T05:00:00.000Z", url: "/?layer=forecast" };
  });
  const p1 = await saved.newPage();
  await p1.goto(`${OK}/?layer=forecast`, { waitUntil: "commit" });
  await p1.waitForLoadState("domcontentloaded");
  expect(await p1.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
  await expect(p1.getByTestId("splash")).toBeHidden();
  await saved.close();
  // a device whose first visit was a reading page: Explore later opens at once, with no launch screen and no hold
  const back = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await back.addInitScript(force);
  const p2 = await back.newPage();
  await p2.goto(`${OK}/about`);
  expect(await p2.evaluate(() => localStorage.getItem("nx-splash-seen-v2"))).toBe("1");
  await p2.goto(`${OK}/?layer=forecast`, { waitUntil: "commit" });
  await p2.waitForLoadState("domcontentloaded");
  expect(await p2.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
  await expect(p2.getByTestId("explore")).toBeVisible();
  await back.close();
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
  // assert from the first paint: on a slow runner the load event can come after the 3 s screen has left
    await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeVisible();
  // the moment it starts to leave: not before the 1.8 s hold, before the 3 s cap, and already click-through
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
  expect(out.at).toBeGreaterThanOrEqual(1780);
  // it begins to leave before its 3 s cap (scheduled at cap − exit; a busy main thread can fire the timer late)
  expect(out.at).toBeLessThanOrEqual(3000);
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 2000 });
  await ctx.close();
});

test("a tap, a click or a key skips it at once", async ({ browser }) => {
  for (const how of ["tap", "click", "key"] as const) {
    const ctx = await browser.newContext(how === "tap" ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } : { viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(force);
    // in-page clocks: when the input arrived (its own event time) and when the screen began to leave
    await ctx.addInitScript(() => {
      const w = window as unknown as { __skip?: { input?: number; out?: number } };
      w.__skip = {};
      const mark = (e: Event) => (w.__skip!.input ??= e.timeStamp);
      for (const t of ["pointerdown", "keydown", "touchstart"]) addEventListener(t, mark, { capture: true });
      new MutationObserver(() => {
        const c = document.documentElement.classList;
        if ((c.contains("nx-splash-out") || !c.contains("nx-splash-on")) && w.__skip!.input != null) w.__skip!.out ??= performance.now();
      }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    });
    const page = await ctx.newPage();
    // assert from the first paint: on a slow runner the load event can come after the 3 s screen has left
    await page.goto(`${OK}/`, { waitUntil: "commit" });
    const splash = page.getByTestId("splash");
    await expect(splash).toBeVisible();
    // it says so, quietly (the words for the pointer in use)
    await expect(splash.getByText(how === "tap" ? "Tap to skip" : "Click or press any key to skip")).toBeVisible();
    const before = await page.evaluate(() => performance.now());
    if (how === "tap") await page.touchscreen.tap(200, 400);
    else if (how === "click") await page.mouse.click(700, 450);
    else await page.keyboard.press("Space");
    // leaving at once (well before the 1.8 s hold, when the skip came early enough), gone in its 450 ms exit
    const leaving = await page.evaluate(() => [document.documentElement.classList.contains("nx-splash-out") || !document.documentElement.classList.contains("nx-splash-on"), performance.now()] as const);
    expect(leaving[0], how).toBe(true);
    await expect(splash).toBeHidden({ timeout: 1000 });
    // it acted on the input itself, not on the hold ending: measured in the page, from the input's own
    // timestamp to the moment the screen began to leave (a busy runner delays both equally)
    const t = await page.evaluate(() => (window as unknown as { __skip: { input?: number; out?: number } }).__skip);
    if (before < 1200 && t.input != null && t.out != null) expect(t.out - t.input, how).toBeLessThan(400);
    // the skip did not land on the map underneath: the page is simply there
    await expect(page.getByTestId("explore")).toBeVisible();
    await ctx.close();
  }
});

test("reduced motion: a still mark that fades", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  // assert from the first paint: on a slow runner the load event can come after the 3 s screen has left
    await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeVisible();
  // (the overlay's own 3 s cap is a fade; nothing inside it draws or moves: the composed final frame)
  const moving = await page.evaluate(() => document.getAnimations().filter((a) => ((a.effect as KeyframeEffect | null)?.target as Element | null)?.parentElement?.closest(".nx-splash")).length);
  expect(moving).toBe(0);
  await expect(page.getByTestId("splash").locator(".nx-splash-word")).toBeVisible();
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 5000 });
  await ctx.close();
});

test("automated browsers do not see it unless they ask", async ({ page }) => {
  await page.goto(`${OK}/about`);
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/);
  await expect(page.getByTestId("splash")).toBeHidden();
});
