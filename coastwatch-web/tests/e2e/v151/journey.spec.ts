import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium, expect, test, type Browser, type BrowserContext, type Page } from "@playwright/test";
import { OK } from "../ports";
import { MOSS, newContext, noSideways, NOW, planTrip, returning, seed, serious, SIZES, tripKeys } from "../v15/helpers";

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
 * The entries are Today's own (nx-today-explore), the navigation's (nx-nav-explore / nx-tab-explore) and the trip
 * workspace's (trip-open-explore); the address contract is src/nx/explore/links.ts.
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

/**
 * A returning reader whose device clock starts at the fixture instant and runs on. Only Date is shifted:
 * page.clock (tests/e2e/v15/helpers.ts newContext) also fakes performance.now and requestAnimationFrame, which freezes
 * Motion, so a panel can pass toBeVisible while it sits at opacity 0. The journey checks what a reader would see.
 */
async function runningContext(browser: Browser, size: { width: number; height: number } = SIZES.desk, o: { reducedMotion?: "reduce" } = {}) {
  const ctx = await browser.newContext({ viewport: size, ...(size.width <= 640 ? { isMobile: true, hasTouch: true } : {}), reducedMotion: o.reducedMotion, storageState: { cookies: [], origins: [] } });
  await seed(ctx, returning);
  await ctx.addInitScript((now) => {
    const Real = Date;
    const off = now - Real.now();
    class Shifted extends Real {
      constructor(...a: []) {
        if (a.length) super(...(a as unknown as [number]));
        else super(Real.now() + off);
      }
      static now() {
        return Real.now() + off;
      }
    }
    (window as unknown as { Date: DateConstructor }).Date = Shifted as unknown as DateConstructor;
  }, Date.parse(NOW));
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return { ctx, page, errors };
}

/** The element is on screen for real: in the viewport and fully opaque once its entrance has run. */
async function onScreen(page: Page, testid: string) {
  const el = page.getByTestId(testid).first();
  await expect(el).toBeVisible();
  await expect.poll(() => el.evaluate((e) => getComputedStyle(e).opacity), { timeout: 5_000, message: `${testid} fades in` }).toBe("1");
}

/**
 * The whole coast is on screen and not under the page's own furniture (the bar, Explore's panel, the time dock):
 * six points on the shore from the Oregon border (42.0 N) to the Mexican border (32.53 N). The topmost element at each
 * point must belong to the chart.
 */
async function coastUncovered(page: Page) {
  return page.evaluate(() => {
    const m = (window as unknown as { __cwMap?: { project(ll: [number, number]): { x: number; y: number } } }).__cwMap;
    if (!m) return ["no map"];
    const canvas = document.querySelector(".maplibregl-canvas")!;
    const box = canvas.getBoundingClientRect();
    const map = canvas.closest("[data-testid=nx-map]") ?? canvas.parentElement!;
    const bad: string[] = [];
    const coast = [
      ["Oregon border", [-124.21, 42.0]],
      ["Cape Mendocino", [-124.41, 40.44]],
      ["Golden Gate", [-122.48, 37.81]],
      ["Point Conception", [-120.47, 34.45]],
      ["Los Angeles harbor", [-118.27, 33.72]],
      ["Mexican border", [-117.12, 32.53]],
    ] as const;
    for (const [name, ll] of coast) {
      const p = m.project(ll as unknown as [number, number]);
      const x = box.left + p.x;
      const y = box.top + p.y;
      if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) {
        bad.push(`${name} off screen (${Math.round(x)}, ${Math.round(y)})`);
        continue;
      }
      // what a reader sees there: the chart itself (its canvas or a mark drawn on it), or a piece of the page over it
      const top = document.elementFromPoint(x, y);
      if (top && !map.contains(top)) {
        const owner = top.closest("[data-testid]");
        bad.push(`${name} under ${owner?.getAttribute("data-testid") ?? top.tagName.toLowerCase() + "." + top.className} (${Math.round(x)}, ${Math.round(y)})`);
      }
    }
    return bad;
  });
}

// ---------------------------------------------------------------- the v15.1 entry points (agent A, src/nx/explore/**)

/** Explore California Waters on Today itself (src/nx/today/ExploreEntry.tsx). */
const exploreEntry = (page: Page) => page.getByTestId("nx-today-explore");
/** The same from the navigation: the bar's pill on a desk, the tab bar on a phone (src/nx/shell/Shell.tsx). */
const navExplore = (page: Page) => page.getByTestId("nx-nav-explore").or(page.getByTestId("nx-tab-explore")).filter({ visible: true }).first();
/** Inside the statewide map: back to the chosen port (Explore.tsx "Back to <port>"). */
const backToPort = (page: Page) => page.getByTestId("nx-back-port");
/** Inside Explore: back to where the reader came from (links.ts backLink: Today, Research or one saved trip). */
const backFrom = (page: Page) => page.getByTestId("nx-back-from");

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
    await expect(page.getByTestId("nx-explore")).toBeVisible({ timeout: 20_000 });
    expect(await scene(page)).toBe(`explore:${MOSS}`);
    expect(atPort(await bounds(page)), "Explore opens framed on the port").toBe(true);

    await page.goto(`${OK}/coast/trips?id=${id}`);
    await expectWorkspace(page, id);
    expect(await tripState(page, id), "exploring changed nothing the trip holds").toEqual(before);
    expect(errors).toEqual([]);
    await ctx.close();
  });

  test("v15.1: Today → Explore California Waters → statewide → zoom to the port → back to the saved trip", async ({ browser }) => {
    test.setTimeout(150_000);
    const { ctx, page, errors } = await runningContext(browser);
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
    // the flight passes through frames that are neither the port nor the state
    const seen: boolean[] = [];
    await expect
      .poll(async () => {
        const b = await bounds(page);
        seen.push(!atPort(b) && !statewide(b));
        return statewide(b);
      }, { timeout: 15_000, intervals: [100], message: "statewide: Oregon border to Mexico" })
      .toBe(true);
    // on the full (GPU) tier the camera flies (≈1.5 s, intermediate frames); a software-rendered chart (the reduced tier,
    // as on CI) takes a 300 ms glide that a slow machine may draw in one or two frames, and the static tier cuts by design
    const tier = await page.getByTestId("nx-sheet").getAttribute("data-tier");
    if (tier === "full") expect(seen.some(Boolean), "the camera flies from the port to the coast (intermediate frames)").toBe(true);
    await expect(page).toHaveURL(/\/explore\?view=coast&place=port-592/);
    await expect(page.getByTestId("nx-explore-heading")).toHaveText("Explore California Waters");
    await onScreen(page, "nx-explore-panel");
    await expect(page.getByTestId("nx-legend").or(page.getByTestId("nx-legend-sticky")).first()).toBeVisible();
    await noSideways(page);

    // the chosen port is kept: zoom back to it
    await expect(backToPort(page)).toHaveText(/Back to Moss Landing/);
    await backToPort(page).click();
    await expect.poll(async () => atPort(await bounds(page)), { timeout: 15_000 }).toBe(true);

    // and back to the saved trip, its planning context intact
    await page.goto(`${OK}/coast/trips?id=${id}`);
    await expectWorkspace(page, id);
    await expect(page.locator("[data-panel]").first()).toBeVisible();
    for (const k of await page.locator("[data-panel]").evaluateAll((els) => els.map((e) => e.getAttribute("data-panel")!)))
      await expect.poll(() => page.locator(`[data-panel="${k}"]`).evaluate((e) => getComputedStyle(e).opacity), { message: `trip panel ${k} fades in` }).toBe("1");
    expect(await tripState(page, id)).toEqual(before);
    expect(errors).toEqual([]);
    await ctx.close();
  });

  test("v15.1: the trip workspace opens the coast, and Explore brings the reader back to that trip", async ({ browser }) => {
    test.setTimeout(120_000);
    const { ctx, page } = await runningContext(browser);
    const id = await planTrip(page);
    const before = await tripState(page, id);
    const open = page.getByTestId("trip-open-explore");
    await open.scrollIntoViewIfNeeded();
    await expect(open).toHaveAttribute("href", `/explore?view=coast&place=${MOSS}&from=trip&trip=${id}`);
    await open.click();
    await mapReady(page);
    await expect.poll(async () => statewide(await bounds(page)), { timeout: 15_000 }).toBe(true);
    const back = backFrom(page);
    await expect(back).toHaveText(/Back to your trip/);
    await back.click();
    await expectWorkspace(page, id);
    expect(await tripState(page, id)).toEqual(before);
    await ctx.close();
  });

  test("v15.1: Explore in the navigation opens the whole coast with the port kept", async ({ browser }) => {
    const { ctx, page } = await runningContext(browser);
    await page.goto(`${OK}/research?place=${MOSS}`);
    await navExplore(page).click();
    await expect(page).toHaveURL(new RegExp(`/explore\\?view=coast&place=${MOSS}`));
    await mapReady(page);
    await expect.poll(async () => statewide(await bounds(page)), { timeout: 15_000 }).toBe(true);
    await expect(backToPort(page)).toHaveText("Back to Moss Landing");
    await ctx.close();
  });

  for (const size of [SIZES.desk, { width: 1280, height: 800 }, { width: 1100, height: 800 }, SIZES.phone, SIZES.small]) {
    test(`v15.1: the statewide view shows Oregon to Mexico uncovered at ${size.width} px`, async ({ browser }) => {
      const { ctx, page } = await runningContext(browser, size);
      await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
      await mapReady(page);
      await expect.poll(async () => statewide(await bounds(page)), { timeout: 15_000 }).toBe(true);
      await page.waitForFunction(() => !(window as unknown as { __cwMap: { isMoving(): boolean } }).__cwMap.isMoving());
      await page.waitForTimeout(600);
      expect(await coastUncovered(page), "the coast from Oregon to Mexico is visible, not under the panel, bar or time dock").toEqual([]);
      await noSideways(page);
      await ctx.close();
    });
  }

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

// ---------------------------------------------------------------- accessibility (axe, WCAG 2.1 A/AA, serious or critical)

test.describe("v15.1 journey: accessibility of the new states", () => {
  for (const size of [SIZES.desk, SIZES.phone]) {
    test(`statewide Explore, C-HARM with currents and stations, at ${size.width} px`, async ({ browser }) => {
      const { ctx, page } = await runningContext(browser, size);
      await page.goto(`${OK}/explore?view=coast&place=${MOSS}&layer=charm&cur=1&st=1`);
      await mapReady(page);
      await onScreen(page, "nx-explore-panel");
      expect(await serious(page)).toEqual([]);
      await ctx.close();
    });
  }

  test("My Coast with a panel moved away from its rest (Reset layout showing)", async ({ browser }) => {
    const { ctx, page } = await runningContext(browser);
    await page.goto(`${OK}/coast?place=${MOSS}`);
    const grip = page.locator("[data-panel-grip]").first();
    await expect(grip).toBeVisible({ timeout: 20_000 });
    const b = (await grip.boundingBox())!;
    await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
    await page.mouse.down();
    await page.mouse.move(b.x + b.width / 2 - 300, b.y + b.height / 2 + 120, { steps: 10 });
    await page.mouse.up();
    await expect(page.getByTestId("panel-reset").first()).toBeVisible();
    expect(await serious(page)).toEqual([]);
    await ctx.close();
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

  test("Today → statewide: the camera jumps, no flight", async ({ browser }) => {
    const { ctx, page } = await newContext(browser);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    await exploreEntry(page).click();
    await expect(page.getByTestId("nx-explore")).toBeVisible({ timeout: 20_000 });
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

  test("Today → statewide on the static tier: the baked statewide fallback renders, with its honest note", async ({ browser }) => {
    const { ctx, page, errors } = await newContext(browser);
    await lowPower(ctx);
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-tier", "static");
    await exploreEntry(page).click();
    await expect(page.getByTestId("nx-explore")).toBeVisible({ timeout: 20_000 });
    await expect(page.locator(".maplibregl-canvas")).toHaveCount(0);
    // the fallback is a real picture of the statewide view, not an empty frame
    const img = page.getByTestId("nx-static-field").or(page.getByTestId("nx-poster").locator("img")).first();
    await expect(img).toBeVisible();
    const box = (await img.boundingBox())!;
    expect(box.width * box.height).toBeGreaterThan(200 * 200);
    await expect(page.getByTestId("nx-static-note")).toBeVisible();
    expect(errors).toEqual([]);
    await ctx.close();
  });
});
