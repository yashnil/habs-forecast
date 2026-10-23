import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The launch screen, "Sounding" (src/v4/shell/Splash.tsx, v9): a device's first visit ever only (v6),
 * never on a saved copy reopened from a field pack, never for a returning reader, never for automated
 * browsers unless asked (every other suite runs without it). Forced on here (each test opens a fresh
 * context, so a first visit): soundings join into the canyon's contours, the mark and lockup form, and
 * once the map is ready (never before 3.3 s) an aperture opens on the live map from Monterey Canyon and
 * the lockup lands in the header's brand (about 4.4 s); without that, the CSS cap fades it by 4.7 s. A tap, click or key skips it in
 * 200 ms; the header stays usable throughout.
 */
const force = () => localStorage.setItem("nx-splash-force-v1", "1");

test("the launch screen shows on a device's first visit, steps aside, and the page is usable", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  const t0 = Date.now();
  // assert from the first paint: on a busy runner the load event can come after the screen has left
  await page.goto(`${OK}/?layer=forecast`, { waitUntil: "commit" });
  const splash = page.getByTestId("splash");
  await expect(splash).toBeVisible();
  // the page renders underneath at the same time (never blocked by the launch screen)
  await expect(page.getByTestId("explore")).toBeAttached();
  await expect(splash.getByRole("status")).toHaveText("Loading Neritic. Press any key to skip.");
  // the soundings are animating (its own CSS, no script needed)
  expect(await page.evaluate(() => document.getAnimations().length)).toBeGreaterThan(0);
  // it plays: still there at 1.5 s (the mark is still forming), gone soon after the handoff or the cap
  await page.waitForTimeout(Math.max(0, 1500 - (Date.now() - t0)));
  if (Date.now() - t0 < 2300) await expect(splash).toBeVisible();
  await expect(splash).toBeHidden({ timeout: 9000 });
  expect(Date.now() - t0).toBeLessThan(11000);
  // the header's brand is back once the mark has landed in it
  await expect(page.getByTestId("brand-home").locator("svg").first()).toHaveCSS("opacity", "1");
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

test("Explore: the handoff starts once the lockup has formed and the map is ready, takes no clicks, and is gone by the cap", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  // assert from the first paint: on a slow runner the load event can come after the screen has left
  await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeVisible();
  // the moment it starts to leave: the handoff (the map was ready) or, on a slow runner, the cap's fade
  const out = await page.evaluate(
    () =>
      new Promise<{ how: string; at: number; pe: string }>((done) => {
        const d = document.documentElement;
        const el = document.querySelector<HTMLElement>(".nx-splash")!;
        const check = () => {
          const how = d.classList.contains("nx-splash-land") ? "land" : getComputedStyle(el).opacity !== "1" || !d.classList.contains("nx-splash-on") ? "cap" : null;
          if (how) done({ how, at: performance.now(), pe: getComputedStyle(el).pointerEvents });
          return !!how;
        };
        if (check()) return;
        const mo = new MutationObserver(() => check() && mo.disconnect());
        mo.observe(d, { attributes: true, attributeFilter: ["class"] });
        const t = setInterval(() => check() && clearInterval(t), 50);
      }),
  );
  // never before the lockup has formed and held (3.3 s on the screen's clock, which starts after navigation)
  expect(out.at).toBeGreaterThanOrEqual(3280);
  if (out.how === "land") expect(out.pe).toBe("none");
  // the cap or the reveal: gone by 4.7 s on its own clock (a busy two-worker runner can delay the start of that clock)
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 4000 });
  await ctx.close();
});

test("the reveal opens an aperture on the live map from Monterey Canyon, and lands the mark exactly on the header's mark", async ({ browser }) => {
  for (const vp of [{ viewport: { width: 1440, height: 900 } }, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }]) {
    const ctx = await browser.newContext(vp);
    await ctx.addInitScript(force);
    // hold the screen once it has landed: the gate's own removal is ignored (from the start: a loaded runner
    // can take longer than the 120 ms cross-fade to get from the landing to the measurement)
    await ctx.addInitScript(() => {
      const rm = DOMTokenList.prototype.remove;
      DOMTokenList.prototype.remove = function (...c: string[]) {
        if (c.some((x) => x.startsWith("nx-splash"))) return;
        return rm.apply(this, c);
      };
    });
    const page = await ctx.newPage();
    await page.goto(`${OK}/`, { waitUntil: "commit" });
    // the flight starts once the handoff has begun and the lockup has settled
    await page.waitForFunction(() => !!document.documentElement?.classList.contains("nx-splash-land") && (document.querySelector(".nx-splash-stage")?.getAnimations().length ?? 0) > 0, null, { timeout: 15000, polling: 16 });
    const m = await page.evaluate(async () => {
      for (const a of document.getAnimations()) {
        const end = a.effect?.getComputedTiming().endTime;
        if (typeof end === "number" && Number.isFinite(end)) {
          a.pause();
          a.currentTime = end - 1;
        }
      }
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const hero = document.querySelector(".nx-splash-hero svg")!.getBoundingClientRect();
      const brand = [...document.querySelectorAll('[data-testid="brand-home"] svg')].find((s) => s.getBoundingClientRect().width > 0)!.getBoundingClientRect();
      // the aperture: the ground (with the chart on it) is masked open from the canyon's place on the map
      const ground = document.querySelector<HTMLElement>(".nx-splash:not(.nx-splash-top)")!;
      const mask = getComputedStyle(ground).maskImage || getComputedStyle(ground).webkitMaskImage;
      const map = (window as unknown as { __cwMap: { project(l: [number, number]): { x: number; y: number }; getContainer(): HTMLElement } }).__cwMap;
      const box = map.getContainer().getBoundingClientRect();
      const c = map.project([-121.879, 36.79]);
      const ax = parseFloat(ground.style.getPropertyValue("--nx-ax"));
      const ay = parseFloat(ground.style.getPropertyValue("--nx-ay"));
      const chart = { mask, off: Math.hypot(ax - (box.left + c.x), ay - (box.top + c.y)) };
      // the lockup's layer sits above the header, so the mark is never hidden under the bar
      const z = Number(getComputedStyle(document.querySelector(".nx-splash-top")!).zIndex);
      return { hero: [hero.x, hero.y, hero.height], brand: [brand.x, brand.y, brand.height], chart, z };
    });
    // the mark lands on the header's mark (within 2 px)
    expect(Math.abs(m.hero[0] - m.brand[0])).toBeLessThan(2);
    expect(Math.abs(m.hero[1] - m.brand[1])).toBeLessThan(2);
    expect(Math.abs(m.hero[2] - m.brand[2])).toBeLessThan(2);
    // the map opens from the canyon's true place (contours stay outside the aperture, never on the map), and
    // the mark lands above the header
    expect(m.chart.mask).toContain("radial-gradient");
    expect(m.chart.off).toBeLessThan(2);
    expect(m.z).toBeGreaterThan(45);
    await ctx.close();
  }
});

test("when the cap ends it (the app's script late), the screen is over and the header's brand is back", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(force);
  // whether the screen was ever on, recorded in the page (a loaded runner can look only after it is over)
  await ctx.addInitScript(() => {
    const w = window as unknown as { __wasOn?: boolean };
    // (the document, not its root element: an init script runs before <html> exists)
    new MutationObserver(() => {
      if (document.documentElement?.classList.contains("nx-splash-on")) w.__wasOn = true;
    }).observe(document, { attributes: true, attributeFilter: ["class"], subtree: true, childList: true });
  });
  const page = await ctx.newPage();
  // the app's script arrives after the cap: no handoff, the CSS fade ends the screen
  // (scripts only: the stylesheets live under the same path, and the screen's timing is its CSS)
  await page.route("**/_next/static/chunks/**/*.js", async (r) => {
    await new Promise((d) => setTimeout(d, 4500));
    await r.continue();
  });
  await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 8000 });
  // the brand comes back with the cap even before any script (CSS), and the screen's classes go once it runs
  await expect(page.getByTestId("brand-home").locator("svg").first()).toHaveCSS("opacity", "1", { timeout: 8000 });
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/, { timeout: 10000 });
  expect(await page.evaluate(() => (window as unknown as { __wasOn?: boolean }).__wasOn)).toBe(true);
  await ctx.close();
});

test("the header stays usable during the opening", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeVisible();
  // the header sits above the screen: the click reaches it (and skips the opening)
  const top = await page.getByTestId("more-menu").evaluate((b) => {
    const r = b.getBoundingClientRect();
    return b.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
  });
  expect(top).toBe(true);
  await page.getByTestId("more-menu").click();
  await expect(page.getByRole("navigation", { name: "More pages" })).toBeVisible();
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
    // assert from the first paint: on a slow runner the load event can come after the 2.2 s screen has left
    await page.goto(`${OK}/`, { waitUntil: "commit" });
    const splash = page.getByTestId("splash");
    await expect(splash).toBeVisible();
    // no visible skip hint (v9): the screen's status says it, for assistive technology
    await expect(splash.getByRole("status")).toContainText("Press any key to skip");
    const before = await page.evaluate(() => performance.now());
    if (how === "tap") await page.touchscreen.tap(200, 400);
    else if (how === "click") await page.mouse.click(700, 450);
    else await page.keyboard.press("Space");
    // leaving at once, gone in its 200 ms exit
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

test("reduced motion: a still lockup that fades within about a second", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  await ctx.addInitScript(force);
  const page = await ctx.newPage();
  // assert from the first paint: on a slow runner the load event can come after the 2.2 s screen has left
    await page.goto(`${OK}/`, { waitUntil: "commit" });
  await expect(page.getByTestId("splash")).toBeVisible();
  // the composed final frame at once (the wordmark is there from the first paint; the screen is short-lived)
  await expect(page.locator(".nx-splash-word")).toBeVisible({ timeout: 500 });
  // nothing inside it draws or moves (its own fade is the only animation)
  const moving = await page.evaluate(() => document.getAnimations().filter((a) => ((a.effect as KeyframeEffect | null)?.target as Element | null)?.parentElement?.closest(".nx-splash")).length);
  expect(moving).toBe(0);
  // held 600-800 ms on the screen's clock, then a 200 ms fade (the CSS cap does the same without a script)
  await expect(page.getByTestId("splash")).toBeHidden({ timeout: 3000 });
  await ctx.close();
});

test("automated browsers do not see it unless they ask", async ({ page }) => {
  await page.goto(`${OK}/about`);
  await expect(page.locator("html")).not.toHaveClass(/nx-splash-on/);
  await expect(page.getByTestId("splash")).toBeHidden();
});
