import { expect, test, type Page } from "@playwright/test";
import { openLayers } from "./helpers";
import { OK } from "./ports";

/**
 * Motion v9 (docs/coastwatch/v9/motion.md §5): rasters fade in 220 ms on a layer switch and cut on a day
 * step of the same layer; camera journeys are capped at 1.8 s; reduced motion cuts both; one WebGL map;
 * the opening never runs on a reading page or a second visit.
 */
const NOW = "2026-10-08T20:00:00Z";
type CwMap = {
  loaded(): boolean;
  getLayer(id: string): unknown;
  getPaintProperty(id: string, p: string): unknown;
  once(e: string, f: () => void): void;
  isMoving(): boolean;
};
type W = { __cwMap?: CwMap; __moved?: number | null };

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}${url}`);
  await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
}

const fade = (page: Page, layer: string) =>
  page.evaluate((id) => {
    const m = (window as unknown as W).__cwMap!;
    return m.getLayer(id) ? m.getPaintProperty(id, "raster-fade-duration") : "missing";
  }, layer);

/** The selected day's neighbour on the timebar (a day step of the same layer). */
async function stepDay(page: Page) {
  const radios = page.getByTestId("timebar").locator('[role="radio"]');
  const n = await radios.count();
  const on = await radios.evaluateAll((els) => els.findIndex((e) => e.getAttribute("aria-checked") === "true"));
  await radios.nth(on + 1 < n ? on + 1 : on - 1).click();
}

/** A pick through the search: the camera move's own duration (`movestart` to `moveend`), ms. */
async function pickTime(page: Page, query: string) {
  const input = page.getByTestId("place-search-input");
  await input.fill(query);
  const opt = page.getByTestId("place-search-list").getByRole("option").first();
  await expect(opt).toBeVisible();
  await page.evaluate(() => {
    const w = window as unknown as W & { __t0?: number };
    w.__moved = null;
    w.__cwMap!.once("movestart", () => {
      w.__t0 = performance.now();
      w.__cwMap!.once("moveend", () => (w.__moved = performance.now() - w.__t0!));
    });
  });
  await opt.click();
  await page.waitForFunction(() => (window as unknown as W).__moved != null, null, { timeout: 5000 });
  return page.evaluate(() => (window as unknown as W).__moved!);
}

test.describe("motion v9: rasters and camera", () => {
  test("a layer switch fades the new raster in 220 ms; a day step of the same layer cuts", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1");
    await expect.poll(() => fade(page, "forecast-raster")).toBe(220);
    // forecast -> satellite: the newly switched layer fades
    await openLayers(page);
    await page.getByTestId("layer-satellite").click();
    await expect.poll(() => fade(page, "satellite-raster")).toBe(220);
    // back to the forecast, then a day step of it: the new day cuts in
    await page.getByTestId("layer-forecast").click();
    await expect.poll(() => fade(page, "forecast-raster")).not.toBe("missing");
    await page.keyboard.press("Escape");
    const before = page.url();
    await stepDay(page);
    await expect.poll(() => page.url()).not.toBe(before);
    await expect.poll(() => fade(page, "forecast-raster")).toBe(0);
  });

  test("one WebGL map, and a far harbour pick arrives within the 1.8 s cap", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1");
    expect(await page.locator("canvas.maplibregl-canvas").count()).toBe(1);
    const ms = await pickTime(page, "Crescent City");
    // one frame of slack over the 1.8 s cap
    expect(ms).toBeLessThanOrEqual(1800 + 50);
    expect(await page.locator("canvas.maplibregl-canvas").count()).toBe(1);
  });

  test("reduced motion: rasters cut and the camera jumps", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    const page = await ctx.newPage();
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1");
    await expect.poll(() => fade(page, "forecast-raster")).toBe(0);
    await openLayers(page);
    await page.getByTestId("layer-satellite").click();
    await expect.poll(() => fade(page, "satellite-raster")).toBe(0);
    await page.keyboard.press("Escape");
    const ms = await pickTime(page, "Crescent City");
    expect(ms).toBeLessThan(50);
    await ctx.close();
  });
});

test.describe("motion v9: the opening's scope", () => {
  test("a second visit, My Coast and Research never show the opening", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await ctx.addInitScript(() => localStorage.setItem("nx-splash-force-v1", "1"));
    const page = await ctx.newPage();
    for (const path of ["/coast", "/research"]) {
      await page.goto(`${OK}${path}`);
      expect(await page.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
      expect(await page.locator('[data-testid="splash"]').count()).toBe(0);
    }
    // the device has now been seen: Explore does not open with it either
    await page.goto(`${OK}/`);
    expect(await page.evaluate(() => document.documentElement.classList.contains("nx-splash-on"))).toBe(false);
    await ctx.close();
  });

  test("reduced motion: a still lockup with nothing animating inside it, gone within about a second", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    await ctx.addInitScript(() => localStorage.setItem("nx-splash-force-v1", "1"));
    const page = await ctx.newPage();
    await page.goto(`${OK}/`, { waitUntil: "commit" });
    await page.waitForFunction(() => document.documentElement.classList.contains("nx-splash-on"), null, { polling: 10 });
    const inside = await page.evaluate(() => {
      const s = document.querySelector(".nx-splash")!;
      return document
        .getAnimations()
        .filter((a) => {
          const t = (a.effect as KeyframeEffect | null)?.target;
          return t && t !== s && s.contains(t as Node);
        })
        .map((a) => (a as CSSAnimation).animationName ?? "waapi");
    });
    expect(inside).toEqual([]);
    // gone for the reader within about a second on the screen's own clock: its CSS fade (the only animation)
    // ends at 1000 ms even before the app's script runs; read from the animation's own timing, since a loaded
    // runner's frames can trail it by several hundred ms
    const end = await page.evaluate(() => {
      const s = document.querySelector<HTMLElement>(".nx-splash")!;
      const cap = s.getAnimations().find((a) => (a as CSSAnimation).animationName === "nx-splash-gone");
      return cap ? Number(cap.effect?.getComputedTiming().endTime) : null;
    });
    expect(end).not.toBeNull();
    expect(end!).toBeLessThanOrEqual(1000);
    await expect(page.getByTestId("splash")).toBeHidden({ timeout: 3000 });
    await ctx.close();
  });
});

test.describe("motion v9: the field pack's check", () => {
  for (const reduced of [false, true]) {
    test(`a saved pack ends on a check${reduced ? " (reduced motion: shown, not drawn)" : ", drawn after the bar gathers"}`, async ({ browser }) => {
      test.setTimeout(150_000);
      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: reduced ? "reduce" : "no-preference" });
      const page = await ctx.newPage();
      await page.goto(`${OK}/coast?place=port-550`);
      await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
      await page.getByTestId("pack-open").click();
      await expect(page.getByTestId("pack-download")).toBeEnabled({ timeout: 60_000 });
      await page.getByTestId("pack-download").click();
      const saved = page.getByTestId("pack-saved");
      await expect(saved).toBeVisible({ timeout: 90_000 });
      await expect(saved).toHaveClass(/nx-saved-play/);
      const tick = await saved.locator(".nx-saved-tick").evaluate((el) => {
        const cs = getComputedStyle(el);
        return { name: cs.animationName, dash: cs.strokeDasharray, offset: cs.strokeDashoffset };
      });
      if (reduced) {
        expect(tick.name).toBe("none");
        // a whole dash over the path's length of 1: the check is fully there
        expect(parseFloat(tick.offset) || 0).toBe(0);
      } else {
        expect(tick.name).toContain("nx-draw");
      }
      await expect(saved.getByRole("status")).toContainText("Saved and verified");
      await ctx.close();
    });
  }
});
