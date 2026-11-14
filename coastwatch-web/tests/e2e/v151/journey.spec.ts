import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium, expect, test, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, newContext, noSideways, NOW, planTrip, returning, SIZES, tripKeys } from "../v15/helpers";

/**
 * Neritic v15.1 cross-feature journey (DIRECTION §7, §8): exploration and trips complement each other.
 *
 *   plan and save a Moss Landing trip → open the statewide ocean map from Today → the camera goes from the port to the
 *   whole California coast (Oregon border to Mexico) → zoom back to the port → return to the saved trip with its
 *   planning context intact → the same trip reopens offline after a browser restart.
 *
 * Plus the same transition under reduced motion (no flight, the final frame at once) and on a low-power device (no
 * WebGL: the static tier, where the statewide view is a baked picture and MapLibre never loads).
 *
 * Parts that need agent A's v15.1 entry point and statewide camera are `test.fixme` until A's branch is merged into
 * feat/neritic-v15.1; the parts the V15 app already supports run now.
 */

// ---------------------------------------------------------------- geography

/** Moss Landing harbour (port-592). */
const MOSS_LL = { lat: 36.805, lon: -121.787 };
/** California's coast: the Oregon border (42.0 N) to the Mexican border (32.53 N). */
const CA = { north: 42.0, south: 32.53 };

type Bounds = { w: number; s: number; e: number; n: number };
const bounds = (page: Page) =>
  page.evaluate(() => {
    const m = (window as unknown as { __cwMap?: { getBounds(): { getWest(): number; getSouth(): number; getEast(): number; getNorth(): number } } }).__cwMap;
    if (!m) return null;
    const b = m.getBounds();
    return { w: b.getWest(), s: b.getSouth(), e: b.getEast(), n: b.getNorth() };
  }) as Promise<Bounds | null>;

const statewide = (b: Bounds | null) => !!b && b.n >= CA.north - 0.15 && b.s <= CA.south + 0.15;
const atPort = (b: Bounds | null) => !!b && b.s < MOSS_LL.lat && b.n > MOSS_LL.lat && b.w < MOSS_LL.lon && b.e > MOSS_LL.lon && b.n - b.s < 2.5;

const mapReady = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
const scene = (page: Page) => page.evaluate(() => (window as unknown as { __nxChart?: { scene: string | null } }).__nxChart?.scene ?? null);

// ---------------------------------------------------------------- the v15.1 entry points (agent A)

/** "Explore California Waters" on Today: a visible entry, not the menu. */
const exploreEntry = (page: Page) => page.getByTestId("nx-explore-entry").or(page.getByRole("link", { name: /explore california waters/i })).first();
/** Inside the statewide map: back to the chosen port. */
const backToPort = (page: Page) => page.getByTestId("nx-explore-port").or(page.getByRole("button", { name: /back to moss landing|moss landing/i })).first();
/** Inside the statewide map: back to the trip the reader came from. */
const backToTrip = (page: Page) => page.getByTestId("nx-explore-trip").or(page.getByRole("link", { name: /trip/i })).first();

/** The device has no WebGL at all: the chart host chooses the static tier (as tests/e2e/nx-shell.spec.ts emulates it). */
async function lowPower(ctx: BrowserContext) {
  await ctx.addInitScript(() => {
    const get = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...rest: unknown[]) {
      return /webgl/.test(type) ? null : (get as (...a: unknown[]) => unknown).call(this, type, ...rest);
    } as typeof get;
    delete (window as unknown as { WebGL2RenderingContext?: unknown }).WebGL2RenderingContext;
  });
}

/** What the saved trip holds that the journey must not disturb. */
async function tripState(page: Page, id: string) {
  return page.evaluate((id) => {
    const keys = Object.keys(localStorage).filter((k) => k.startsWith("nx-trips-v1:")).sort();
    return { keys, trip: localStorage.getItem(`nx-trips-v1:trip:${id}`), snaps: keys.filter((k) => k.includes("snap")).map((k) => localStorage.getItem(k)) };
  }, id);
}

async function expectWorkspace(page: Page, id: string) {
  await expect(page).toHaveURL(new RegExp(`/coast/trips\\?id=${id}`));
  await expect(page.getByTestId("trip-workspace")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("trip-workspace")).toHaveAttribute("data-state", "planned");
  await expect(page.getByTestId("trip-money-figure")).toContainText("$2,602.24");
}

// ---------------------------------------------------------------- the journey

test.describe("v15.1 journey: a saved trip, the whole coast, and back", () => {
  test("V15 base: plan and save, open Explore at the port, return to the trip with its planning context intact", async ({ browser }) => {
    test.setTimeout(120_000);
    const { ctx, page, errors } = await newContext(browser);
    const id = await planTrip(page);
    const before = await tripState(page, id);
    expect(before.trip, "the trip is stored").toBeTruthy();

    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    await page.goto(`${OK}/explore?place=${MOSS}`);
    await mapReady(page);
    await expect(page.getByTestId("nx-explore")).toBeVisible();
    expect(await scene(page)).toBe(`explore:${MOSS}`);
    expect(atPort(await bounds(page)), "Explore opens framed on the port").toBe(true);

    await page.goto(`${OK}/coast/trips?id=${id}`);
    await expectWorkspace(page, id);
    expect(await tripState(page, id), "exploring changed nothing the trip holds").toEqual(before);
    expect(errors).toEqual([]);
    await ctx.close();
  });

  test.fixme("v15.1: Today → Explore California Waters → statewide → zoom to the port → back to the saved trip", async ({ browser }) => {
    test.setTimeout(150_000);
    const { ctx, page, errors } = await newContext(browser);
    const id = await planTrip(page);
    const before = await tripState(page, id);

    // Today at the port, with a visible entry to the statewide map
    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    expect(atPort(await bounds(page))).toBe(true);
    const entry = exploreEntry(page);
    await expect(entry).toBeVisible();
    await entry.click();

    // the camera animates from the port to the whole coast (it passes through intermediate frames)
    await expect(page).toHaveURL(/\/explore/);
    await expect(page.getByTestId("nx-explore")).toBeVisible();
    await expect.poll(async () => statewide(await bounds(page)), { timeout: 15_000, message: "statewide: Oregon border to Mexico" }).toBe(true);
    await expect(page.getByTestId("nx-legend").or(page.getByTestId("nx-legend-sticky")).first()).toBeVisible();
    await noSideways(page);

    // the chosen port is kept: zoom back to it
    await backToPort(page).click();
    await expect.poll(async () => atPort(await bounds(page)), { timeout: 15_000 }).toBe(true);

    // and back to the saved trip, its planning context intact
    await page.goto(`${OK}/coast/trips?id=${id}`);
    await expectWorkspace(page, id);
    expect(await tripState(page, id)).toEqual(before);
    expect(errors).toEqual([]);
    await ctx.close();
  });

  test.fixme("v15.1: the statewide map offers the way back to the trip the reader came from", async ({ browser }) => {
    test.setTimeout(120_000);
    const { ctx, page } = await newContext(browser);
    const id = await planTrip(page);
    // from the trip's workspace into the statewide map, then straight back
    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    await exploreEntry(page).click();
    await expect.poll(async () => statewide(await bounds(page)), { timeout: 15_000 }).toBe(true);
    const back = backToTrip(page);
    await expect(back).toBeVisible();
    await back.click();
    await expectWorkspace(page, id);
    await ctx.close();
  });

  test("offline: the trip saved before exploring reopens after a browser restart with no network", async () => {
    test.setTimeout(150_000);
    const channel = process.env.CI ? undefined : "chrome";
    const dir = mkdtempSync(join(tmpdir(), "nx-v151-journey-"));
    try {
      let id = "";
      {
        const ctx = await chromium.launchPersistentContext(dir, { channel, viewport: SIZES.desk, serviceWorkers: "allow" });
        const page = ctx.pages()[0] ?? (await ctx.newPage());
        await page.clock.setFixedTime(new Date(NOW));
        await page.goto(`${OK}/about`);
        await page.evaluate((kv) => Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, v)), returning);
        id = await planTrip(page);
        await expect.poll(() => page.evaluate(async () => !!(await (await caches.open("neritic-trips-v1")).match("/coast/trips"))), { timeout: 30_000 }).toBe(true);
        await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller), { timeout: 15_000 }).toBe(true);
        // explore the coast, then leave the browser
        await page.goto(`${OK}/explore?place=${MOSS}`);
        await mapReady(page);
        await ctx.close();
      }
      const ctx = await chromium.launchPersistentContext(dir, { channel, viewport: SIZES.desk, offline: true, serviceWorkers: "allow" });
      const page = ctx.pages()[0] ?? (await ctx.newPage());
      await page.clock.setFixedTime(new Date(NOW));
      await page.goto(`${OK}/coast/trips?id=${id}`);
      await expectWorkspace(page, id);
      await expect(page.getByTestId("nx-offline-notice")).toBeVisible();
      await expect(page.getByTestId("nx-offline-notice")).not.toContainText(/live|current/i);
      expect(await tripKeys(page)).toContain(`nx-trips-v1:trip:${id}`);
      await ctx.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------- reduced motion

test.describe("v15.1 journey under reduced motion", () => {
  test("first visit: no cinematic opening, the final frame at once", async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: "reduce", viewport: SIZES.desk, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/`);
    expect(await page.evaluate(() => document.documentElement.getAttribute("data-nx-opening"))).toBeNull();
    await expect(page.getByTestId("nx-still")).toBeHidden();
    await mapReady(page);
    await ctx.close();
  });

  test.fixme("Today → statewide: the camera jumps, no flight", async ({ browser }) => {
    const { ctx, page } = await newContext(browser);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    await exploreEntry(page).click();
    await expect(page.getByTestId("nx-explore")).toBeVisible();
    // the statewide frame is there almost at once (a flight takes over a second)
    await expect.poll(async () => statewide(await bounds(page)), { timeout: 1_500, intervals: [100] }).toBe(true);
    const moving = await page.evaluate(() => (window as unknown as { __cwMap?: { isMoving(): boolean } }).__cwMap?.isMoving() ?? false);
    expect(moving, "no camera animation under reduced motion").toBe(false);
    await backToPort(page).click();
    await expect.poll(async () => atPort(await bounds(page)), { timeout: 1_500, intervals: [100] }).toBe(true);
    await ctx.close();
  });
});

// ---------------------------------------------------------------- low power

test.describe("v15.1 journey on a low-power device (static tier)", () => {
  test("Today at the port draws the static chart and never loads MapLibre", async ({ browser }) => {
    const { ctx, page, errors } = await newContext(browser);
    await lowPower(ctx);
    const scripts: string[] = [];
    page.on("request", (r) => scripts.push(r.url()));
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", "static");
    await expect(page.getByTestId("nx-poster").locator("img").first()).toBeVisible();
    await expect(page.locator(".maplibregl-canvas")).toHaveCount(0);
    expect(scripts.filter((u) => /maplibre/.test(u) && !/\.webp/.test(u))).toEqual([]);
    expect(errors).toEqual([]);
    await ctx.close();
  });

  test.fixme("Today → statewide on the static tier: the baked statewide fallback renders, with its honest note", async ({ browser }) => {
    const { ctx, page, errors } = await newContext(browser);
    await lowPower(ctx);
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", "static");
    await exploreEntry(page).click();
    await expect(page.getByTestId("nx-explore")).toBeVisible();
    await expect(page.locator(".maplibregl-canvas")).toHaveCount(0);
    // the fallback is a real picture of the statewide view, not an empty frame
    const img = page.getByTestId("nx-poster").locator("img").or(page.getByTestId("nx-static-field")).first();
    await expect(img).toBeVisible();
    const box = (await img.boundingBox())!;
    expect(box.width * box.height).toBeGreaterThan(200 * 200);
    await expect(page.getByTestId("nx-static-note")).toBeVisible();
    expect(errors).toEqual([]);
    await ctx.close();
  });
});
