import { expect, test, type Page } from "@playwright/test";
import { HFRGAP, OK } from "./ports";
import { openDisplay } from "./helpers";

/**
 * v10 currents (docs/coastwatch/design-research/geophysical-visualization.md): observed HF-radar currents
 * as moving particles in the map's own WebGL context, inside observed cells only; a still frame under
 * reduced motion; a pause that persists; the hour that observed the region in view; the selection's true
 * footprint, carried in the URL; playback that holds the shown hour until the next one's data is in.
 */
const NOW = "2026-10-08T20:00:00Z";
type Flow = {
  running(): boolean;
  slow(): boolean;
  measured(): boolean;
  fps(): number;
  count(): number;
  stats: { frames: number; segments: number; footprintCells: number };
  points(): [number, number][];
  mask(): { lat_first: number; lat_step: number; lon_first: number; lon_step: number; width: number; height: number; observed: number[] } | null;
  fieldId(): string;
};
type CwMap = {
  loaded(): boolean;
  getLayer(id: string): { type: string } | undefined;
  getLayersOrder(): string[];
  getSource(id: string): { serialize(): { data: GeoJSON.Feature } } | undefined;
  getContainer(): HTMLElement;
  project(p: [number, number]): { x: number; y: number };
};
type W = { __cwMap?: CwMap; __cwFlow?: Flow; __mark?: Flow };

async function open(page: Page, url: string, base = OK) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${base}${url}`);
  await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
}
const flowReady = (page: Page) => page.waitForFunction(() => ((window as unknown as W).__cwFlow?.stats.segments ?? 0) > 0);
const frames = (page: Page) => page.evaluate(() => (window as unknown as W).__cwFlow!.stats.frames);

test.describe("v10 currents: an honest flow field", () => {
  // grid requests are watched and held by the page itself (a service worker would answer them unseen)
  test.use({ serviceWorkers: "block" });
  test("drawn as a custom layer in the map's one WebGL context, under the labels, animating", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay");
    await flowReady(page);
    const info = await page.evaluate(() => {
      const m = (window as unknown as W).__cwMap!;
      const order = m.getLayersOrder();
      return { type: m.getLayer("nx-flow")?.type, flow: order.indexOf("nx-flow"), coastline: order.indexOf("coastline"), labels: order.filter((id) => /^(place-|water-name|official|ports-label|v4-hl)/.test(id)).map((id) => order.indexOf(id)) };
    });
    expect(info.type).toBe("custom");
    // directly under the overlay group: below the coastline, the labels, the notices and the selection
    expect(info.flow).toBe(info.coastline - 1);
    for (const i of info.labels) expect(i).toBeGreaterThan(info.flow);
    expect(await page.locator("canvas").count()).toBe(1);
    // animating; or, on a device that draws it too slowly (a software-WebGL runner: about 2 fps), stopped on the
    // still frame after its first frames, which says so beside a Play choice
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.measured()), { timeout: 30_000 }).toBe(true);
    if (await page.evaluate(() => (window as unknown as W).__cwFlow!.slow())) {
      expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(false);
      await expect(page.getByTestId("flow-slow").first()).toContainText("draws the moving streaks too slowly");
      await expect(page.getByTestId("flow-pause").first()).toContainText("Play motion");
      test.info().annotations.push({ type: "slow-webgl", description: "motion checked as the slow-device still frame" });
    } else {
      expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(true);
      const f0 = await frames(page);
      await page.waitForTimeout(500);
      expect((await frames(page)) - f0).toBeGreaterThan(5);
    }
    // the legend says the motion is a scaled symbol; the static arrows stay one tap away
    await expect(page.getByTestId("flow-scale-note").first()).toContainText("Screen speed is a scaled symbol");
    await openDisplay(page);
    await page.getByTestId("flow-view-arrows").first().click();
    await expect.poll(() => page.evaluate(() => !!(window as unknown as W).__cwMap!.getLayer("currents-arrows-1"))).toBe(true);
    expect(await page.evaluate(() => !!(window as unknown as W).__cwMap!.getLayer("nx-flow"))).toBe(false);
    await expect(page.getByTestId("legend-currents").first()).toContainText("No arrow means no observation, not calm water");
    await page.getByTestId("flow-view-flow").first().click();
    await flowReady(page);
  });

  test("particles stay inside observed cells, at every sample (the layer's own positions against the cell mask)", async ({ page }) => {
    await open(page, "/?layer=currents&at=36.8,-122.05,10.5");
    await flowReady(page);
    let checked = 0;
    for (let i = 0; i < 12; i++) {
      const bad = await page.evaluate(() => {
        const f = (window as unknown as W).__cwFlow!;
        const m = f.mask()!;
        const pts = f.points();
        const out = pts.filter(([lon, lat]) => {
          const r = Math.floor((lat - m.lat_first) / m.lat_step + 0.5);
          const c = Math.floor((lon - m.lon_first) / m.lon_step + 0.5);
          return r < 0 || c < 0 || r >= m.height || c >= m.width || !m.observed[r * m.width + c];
        });
        return { n: pts.length, out: out.length, holes: m.observed.filter((x) => !x).length };
      });
      expect(bad.holes).toBeGreaterThan(0);
      expect(bad.out).toBe(0);
      checked += bad.n;
      await page.waitForTimeout(120);
    }
    expect(checked).toBeGreaterThan(1000);
  });

  test("reduced motion: one still frame, no animation loop, nothing to pause", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await open(page, "/?layer=currents&region=monterey_bay");
    await flowReady(page);
    await expect(page.getByTestId("flow-still").first()).toBeVisible();
    await expect(page.getByTestId("flow-pause")).toHaveCount(0);
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(false);
    await page.waitForTimeout(300);
    const f0 = await frames(page);
    await page.waitForTimeout(1000);
    // no loop: the map draws only when something else asks it to (nothing does here)
    expect((await frames(page)) - f0).toBeLessThan(3);
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.stats.segments)).toBeGreaterThan(100);
  });

  test("the pause control is reachable by keyboard, stops the loop and is kept on this device", async ({ page }) => {
    // the person's own pause, not the slow-device one: a fixed rate is never measured (a software-WebGL runner
    // would otherwise be paused for slowness first)
    await page.addInitScript(() => ((window as unknown as { __cwFlowFps?: number }).__cwFlowFps = 60));
    await open(page, "/?layer=currents&region=monterey_bay");
    await flowReady(page);
    const pause = page.getByTestId("flow-pause").first();
    // v10: it sits in view beside the currents legend: from the legend's own control, one Tab reaches it; Enter presses it
    await page.getByTestId("legend-currents").first().getByTestId("legend-more").focus();
    await page.keyboard.press("Tab");
    await expect(pause).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(pause).toHaveAttribute("aria-pressed", "true");
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(false);
    const f0 = await frames(page);
    await page.waitForTimeout(600);
    expect((await frames(page)) - f0).toBeLessThan(3);
    // paused, the particles are still drawn (a frozen frame, not an empty sea)
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.stats.segments)).toBeGreaterThan(0);
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
    await flowReady(page);
    await expect(page.getByTestId("flow-pause").first()).toHaveAttribute("aria-pressed", "true");
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(false);
    await page.getByTestId("flow-pause").first().press("Space");
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.running())).toBe(true);
  });

  test("Monterey Bay opens on the newest hour with radar coverage there, and the dock says why", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay", HFRGAP);
    await flowReady(page);
    await expect(page.getByTestId("time-20261008T11Z").first()).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("currents-hour-note").first()).toContainText("has no coverage in Monterey Bay");
    await expect(page.getByTestId("currents-hour-note").first()).toContainText("the newest hour with coverage here");
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.fieldId())).toBe("hfr2km_currents_20261008T11Z");
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.stats.footprintCells)).toBeGreaterThan(500);
    // the empty hour stays one tap away, chosen explicitly: an empty sea, said to be no observation
    await page.getByTestId("time-20261008T12Z").first().click();
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.fieldId())).toBe("hfr2km_currents_20261008T12Z");
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.stats.footprintCells)).toBe(0);
    await expect(page.getByTestId("currents-hour-note")).toHaveCount(0);
  });

  test("a new hour swaps the layer's data (no remount); the shown hour and its label hold until the next one has arrived", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay");
    await flowReady(page);
    await page.evaluate(() => {
      const w = window as unknown as W;
      w.__mark = w.__cwFlow;
    });
    const stamp = page.getByTestId("stamp-currents").first();
    const before = await stamp.textContent();
    // the 9Z grids arrive slowly
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    await page.route(/20261008T09Z-[^/]+\/[uv]\.u16\.gz/, async (route) => {
      await gate;
      await route.continue();
    });
    await page.getByTestId("time-20261008T09Z").first().click();
    // the frame shown, and its time, stay; the step asked for says it is loading (after 400 ms)
    await expect(page.getByTestId("time-20261008T09Z").first()).toHaveAttribute("data-loading", "true");
    await expect(page.getByTestId("time-20261008T12Z").first()).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("time-20261008T09Z").first().getByTestId("time-loading")).toBeVisible();
    expect(await stamp.textContent()).toBe(before);
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow!.fieldId())).toBe("hfr2km_currents_20261008T12Z");
    release();
    await expect(page.getByTestId("time-20261008T09Z").first()).toHaveAttribute("aria-checked", "true");
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.fieldId())).toBe("hfr2km_currents_20261008T09Z");
    await expect(stamp).not.toHaveText(before ?? "");
    // the same layer, its buffers swapped
    expect(await page.evaluate(() => (window as unknown as W).__cwFlow === (window as unknown as W).__mark)).toBe(true);
  });

  test("playback asks for the next real hour ahead of showing it", async ({ page }) => {
    await open(page, "/?layer=currents&region=monterey_bay");
    await flowReady(page);
    // start on 07Z: playback's next frames are 08Z, then 09Z (neither fetched yet)
    await page.getByTestId("time-20261008T07Z").first().click();
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwFlow!.fieldId())).toBe("hfr2km_currents_20261008T07Z");
    const asked: string[] = [];
    page.on("request", (r) => {
      const m = /(20261008T\d\dZ)-[^/]+\/u\.u16\.gz/.exec(r.url());
      if (m) asked.push(m[1]);
    });
    await page.getByTestId("time-play").first().click();
    // before the first step (1.2 s) the next hour is already being fetched
    await expect.poll(() => asked.includes("20261008T08Z"), { timeout: 1000 }).toBe(true);
    await expect(page.getByTestId("time-20261008T08Z").first()).toHaveAttribute("aria-checked", "true", { timeout: 3000 });
    // and the one after it, before it is shown
    await expect.poll(() => asked.includes("20261008T09Z"), { timeout: 1000 }).toBe(true);
    expect(await page.getByTestId("time-20261008T09Z").first().getAttribute("aria-checked")).toBe("false");
    await page.getByTestId("time-play").first().click();
  });
});

test.describe("v10: the selection's true footprint", () => {
  test("a radar point outlines its 2 km cell after the camera settles; the URL carries it through reload and a trip to My Coast", async ({ page }) => {
    // three map loads with the currents (a software-WebGL runner takes ~12 s for each)
    test.setTimeout(120_000);
    await open(page, "/?layer=currents&sel=point:36.80,-122.10");
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwMap!.getContainer().dataset.footprint), { timeout: 20_000 }).toBe("drawn");
    await expect(page).toHaveURL(/fp=hfr2km(%3A|:)\d+(%3A|:)\d+/);
    const ring = await page.evaluate(() => (window as unknown as W).__cwMap!.getSource("v4-footprint")!.serialize().data.geometry as GeoJSON.LineString);
    const lons = ring.coordinates.map((c) => c[0]);
    const lats = ring.coordinates.map((c) => c[1]);
    expect(ring.coordinates).toHaveLength(5);
    // the fixture's radar lattice: 0.0208° by 0.0180°, around the selected point
    expect(Math.max(...lons) - Math.min(...lons)).toBeCloseTo(0.02082, 4);
    expect(Math.max(...lats) - Math.min(...lats)).toBeCloseTo(0.01798, 4);
    expect(Math.min(...lons)).toBeLessThanOrEqual(-122.1);
    expect(Math.max(...lons)).toBeGreaterThanOrEqual(-122.1);
    expect(Math.min(...lats)).toBeLessThanOrEqual(36.8);
    expect(Math.max(...lats)).toBeGreaterThanOrEqual(36.8);
    const fp = new URL(page.url()).searchParams.get("fp");
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
    await expect.poll(() => new URL(page.url()).searchParams.get("fp")).toBe(fp);
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwMap!.getContainer().dataset.footprint), { timeout: 20_000 }).toBe("drawn");
    // to My Coast and back: the selection and its footprint are still the page's address
    await page.goto(`${OK}/coast`);
    await page.goBack();
    await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
    await expect.poll(() => new URL(page.url()).searchParams.get("fp")).toBe(fp);
    await expect(page.getByTestId("inspect-panel")).toBeVisible();
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwMap!.getContainer().dataset.footprint), { timeout: 20_000 }).toBe("drawn");
  });

  test("a click on the forecast outlines the 3 km model cell it reads, drawn in once", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&at=36.75,-122.1,10.5");
    const at = await page.evaluate(() => {
      const m = (window as unknown as W).__cwMap!;
      const r = m.getContainer().getBoundingClientRect();
      const p = m.project([-122.1, 36.75]);
      return { x: r.left + p.x, y: r.top + p.y };
    });
    await page.mouse.click(at.x, at.y);
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "cell");
    // drawing, then drawn (300 ms)
    await expect.poll(() => page.evaluate(() => (window as unknown as W).__cwMap!.getContainer().dataset.footprint), { timeout: 20_000 }).toBe("drawn");
    await expect(page).toHaveURL(/fp=charm(%3A|:)\d+(%3A|:)\d+/);
    const ring = await page.evaluate(() => (window as unknown as W).__cwMap!.getSource("v4-footprint")!.serialize().data.geometry as GeoJSON.LineString);
    const lons = ring.coordinates.map((c) => c[0]);
    expect(Math.max(...lons) - Math.min(...lons)).toBeCloseTo(0.03, 3);
  });
});

test.describe("v10 fix 2: the rail holds still when an hour turns old", () => {
  test.use({ serviceWorkers: "block" });
  test("the status row reserves the stale line: the scrubber is at the same height with two lines or three", async ({ page }) => {
    // two map loads with the currents (a software-WebGL runner takes ~12 s for each)
    test.setTimeout(120_000);
    const ys: number[] = [];
    // a day after the radar hours they are current (two lines); two days after, older than usual (a third line)
    for (const at of ["2026-10-09T20:00:00Z", "2026-10-10T20:00:00Z"]) {
      await page.clock.setFixedTime(new Date(at));
      await page.goto(`${OK}/?layer=currents&region=monterey_bay`);
      await page.waitForFunction(() => !!(window as unknown as W).__cwMap?.loaded());
      await flowReady(page);
      await expect(page.getByTestId("stale-note")).toHaveCount(at.startsWith("2026-10-10") ? 1 : 0);
      if (at.startsWith("2026-10-10")) {
        await page.getByTestId("time-play").first().click();
        await page.waitForTimeout(1500);
      }
      ys.push((await page.getByTestId("time-play").first().boundingBox())!.y);
    }
    expect(Math.abs(ys[0] - ys[1])).toBeLessThan(1);
  });
});
