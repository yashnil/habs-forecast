import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Journey 3 (docs/coastwatch/v12/design-system/journeys.md): a researcher opens Explore, switches layers, picks
 * frames, taps the water, reads a value that matches the published grid, and follows "See the record nearby"
 * to Research with the station and the time window in the URL. The journey is recorded on video into
 * docs/coastwatch/v12/implementation/explore/. With NX_MEDIA=1 the review stills and videos are written too.
 */
// review media go to the docs only on request (NX_MEDIA=1); an ordinary run keeps them with its own results
const OUT = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/explore") : path.resolve(__dirname, "../../../test-results/journey-media/explore");
const DATA = path.resolve(__dirname, "../../../public/data/fixture/v1");
const manifest = JSON.parse(readFileSync(path.join(DATA, "manifest.json"), "utf8")) as { layers: Array<{ layer_id: string; grid: { url: string; width: number; height: number; lat_first: number; lat_step: number; lon_first: number; lon_step: number; scale_factor: number; add_offset: number; nodata: number; chunks?: { url_template: string } | null } | null }> };

type Hook = { product: string; frame: string; hour: string | null; at: { lat: number; lon: number } | null; field?: string; fieldDraws?: () => number; flow?: { running: () => boolean; segments: () => number; fieldId: () => string; count: () => number } };
const hook = (page: Page) => page.evaluate(() => {
  const h = (window as unknown as { __nxExplore?: Hook }).__nxExplore;
  return h ? { product: h.product, frame: h.frame, hour: h.hour, at: h.at, field: h.field ?? null, draws: h.fieldDraws?.() ?? 0, flowSegments: h.flow?.segments() ?? 0, flowRunning: h.flow?.running() ?? false, flowField: h.flow?.fieldId() ?? null } : null;
});
const mapReady = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });

/** The published chlorophyll value of the cell under a point, straight from the fixture's grid file. */
function publishedChl(layerId: string, lat: number, lon: number): number | null {
  const g = manifest.layers.find((l) => l.layer_id === layerId)!.grid!;
  const file = (g.chunks?.url_template ?? g.url).replace("{row}", "0").replace("{col}", "0");
  const buf = gunzipSync(readFileSync(path.join(DATA, file)));
  const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5), col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  const code = buf.readUInt16LE((row * g.width + col) * 2);
  return code === g.nodata ? null : 10 ** (code * g.scale_factor + g.add_offset);
}

async function tapWater(page: Page, lat: number, lon: number) {
  const p = await page.evaluate(([la, lo]) => (window as unknown as { __cwMap: { project: (x: [number, number]) => { x: number; y: number } } }).__cwMap.project([lo, la]), [lat, lon]);
  const box = (await page.getByTestId("nx-map").boundingBox())!;
  await page.mouse.click(box.x + p.x, box.y + p.y);
}

async function journey(page: Page) {
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  const legend = page.getByTestId("nx-legend");
  await expect(legend).toHaveAttribute("data-product", "chl");
  await expect(legend).toContainText("Algae, not toxin");
  await expect(legend).toContainText("Sentinel-3 OLCI");
  await expect.poll(async () => (await hook(page))?.field, { timeout: 15_000 }).toBe("olci:latest");
  await expect.poll(async () => (await hook(page))?.draws ?? 0).toBeGreaterThan(0);
  // measured currents stream over the observed layer
  await expect.poll(async () => (await hook(page))?.flowSegments ?? 0, { timeout: 10_000 }).toBeGreaterThan(0);
  await page.waitForTimeout(1200);

  // switch layers: each legend is its title, with its kind and mandatory words
  await page.getByTestId("nx-layer-sst").click();
  await expect(legend).toHaveAttribute("data-product", "sst");
  await expect(legend).toContainText("Analysis");
  await page.waitForTimeout(900);
  await page.getByTestId("nx-layer-charm").click();
  await expect(legend).toHaveAttribute("data-product", "charm");
  await expect(legend).toContainText("Past run: every day it covers is before today");
  await expect(legend).toContainText("not toxin in seafood");
  // T25: no streaks over a model
  await expect(page.getByTestId("nx-currents-key")).toHaveCount(0);
  await expect(page.getByTestId("nx-play")).toHaveCount(0);
  await page.waitForTimeout(900);
  await page.getByTestId("nx-layer-outlook").click();
  await expect(legend).toContainText("factor of 2");
  await page.waitForTimeout(900);
  await page.getByTestId("nx-layer-chl").click();
  await expect(legend).toHaveAttribute("data-product", "chl");

  // pick frames: a day of OLCI is a cut to that day's own grid
  await page.getByTestId("nx-frame-olci:2026-10-05").click();
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", "olci:2026-10-05");
  await expect(page.getByTestId("nx-frame-stamp")).toContainText("Seen Oct 5");
  await expect.poll(async () => (await hook(page))?.field).toBe("olci:2026-10-05");
  await page.waitForTimeout(900);
  await page.getByTestId("nx-frame-olci:latest").click();
  await expect.poll(async () => (await hook(page))?.field).toBe("olci:latest");
  // and an hour of the currents
  const hourId = "hfr2km_currents_20261008T02Z";
  await page.getByTestId(`nx-hour-${hourId}`).click();
  await expect.poll(async () => (await hook(page))?.flowField).toBe(hourId);
  await expect(page.getByTestId("nx-frame-stamp")).toContainText("Currents: Oct 7, 7:00 PM PT");
  await page.waitForTimeout(1200);

  // tap the water in Monterey Bay
  await tapWater(page, 36.87, -122.02);
  const readout = page.getByTestId("nx-readout");
  await expect(readout).toContainText("Chlorophyll here");
  const at = (await hook(page))!.at!;
  const expected = publishedChl("olci300_chl_latest", at.lat, at.lon);
  expect(expected).not.toBeNull();
  const shown = await readout.locator("p").filter({ hasText: "mg/m³" }).first().innerText();
  const v = Number(/([\d.]+)\s*mg/.exec(shown)![1]);
  expect(v).toBeCloseTo(Number(expected!.toPrecision(2)), 6);
  await expect(readout).toContainText("Measured · Sentinel-3 OLCI");
  await expect(readout).toContainText("This pixel was seen");
  await expect(readout).toContainText("Nearest measured current");
  await expect(page.getByTestId("nx-transect")).toBeVisible();
  await expect(page).toHaveURL(/at=36\.87/);
  // the transect and the chart read together
  await page.getByTestId("nx-transect").scrollIntoViewIfNeeded();
  const t = (await page.getByTestId("nx-transect").locator("svg").boundingBox())!;
  await page.mouse.move(t.x + t.width * 0.3, t.y + t.height / 2);
  await expect(page.getByTestId("nx-reading-on-chart").locator("circle").first()).toHaveAttribute("visibility", "visible");
  await page.waitForTimeout(1200);

  // follow to Research with the station and the window
  const link = page.getByTestId("nx-record-link");
  await expect(link).toContainText("Santa Cruz Wharf");
  await link.click();
  await expect(page).toHaveURL(/\/research\?station=HABs-SantaCruzWharf&from=\d{4}-\d{2}-\d{2}&to=\d{4}-\d{2}-\d{2}&var=chl_extracted&place=port-592$/);
  await page.waitForTimeout(900);
}

test.describe("Journey 3: Explore to the record", () => {
  test("open, switch layers, pick frames, tap the water, follow to Research (recorded)", async ({ browser }) => {
    test.setTimeout(90_000);
    mkdirSync(OUT, { recursive: true });
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] }, recordVideo: { dir: test.info().outputDir, size: { width: 1440, height: 900 } } });
    await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    const page = await ctx.newPage();
    await journey(page);
    const video = page.video();
    await ctx.close();
    if (video) copyFileSync(await video.path(), path.join(OUT, "journey-explore-1440.webm"));
  });

  test("the URL keeps the selection: layer, frame and point survive a reload", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    await page.goto(`${OK}/explore?place=port-592&layer=charm&frame=charm:1&at=36.8000,-122.0500`);
    await mapReady(page);
    await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", "charm:1");
    await expect(page.getByTestId("nx-readout")).toContainText("C-HARM here");
    await expect(page.getByTestId("nx-readout")).toContainText("not toxin in seafood");
  });

  test("a lost WebGL context: the data layers come back", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    await page.goto(`${OK}/explore?place=port-592`);
    await mapReady(page);
    await expect.poll(async () => (await hook(page))?.flowSegments ?? 0, { timeout: 10_000 }).toBeGreaterThan(0);
    const layers = () => page.evaluate(() => (window as unknown as { __cwMap: { getLayersOrder: () => string[] } }).__cwMap.getLayersOrder().filter((id) => id.startsWith("nx-")));
    expect(await layers()).toEqual(["nx-field", "nx-flow"]);
    await page.evaluate(() => {
      const gl = (window as unknown as { __cwMap: { painter: { context: { gl: WebGL2RenderingContext } } } }).__cwMap.painter.context.gl;
      const ext = gl.getExtension("WEBGL_lose_context")!;
      (window as unknown as { __lose: WEBGL_lose_context }).__lose = ext;
      ext.loseContext();
    });
    await page.waitForTimeout(500);
    await page.evaluate(() => (window as unknown as { __lose: WEBGL_lose_context }).__lose.restoreContext());
    await expect.poll(layers, { timeout: 10_000 }).toEqual(["nx-field", "nx-flow"]);
    const before = (await hook(page))!.draws;
    await page.evaluate(() => (window as unknown as { __cwMap: { triggerRepaint: () => void } }).__cwMap.triggerRepaint());
    await expect.poll(async () => (await hook(page))!.draws, { timeout: 10_000 }).toBeGreaterThan(before);
    await expect.poll(async () => (await hook(page))?.flowSegments ?? 0, { timeout: 10_000 }).toBeGreaterThan(0);
  });

  test("reduced motion: the same streaks, still", async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: "reduce", storageState: { cookies: [], origins: [] } });
    await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    const page = await ctx.newPage();
    await page.goto(`${OK}/explore?place=port-592`);
    await mapReady(page);
    await expect.poll(async () => (await hook(page))?.flowSegments ?? 0, { timeout: 10_000 }).toBeGreaterThan(0);
    expect((await hook(page))!.flowRunning).toBe(false);
    await ctx.close();
  });
});

// ---------------------------------------------------------------- review media (NX_MEDIA=1)
async function still(browser: Browser, name: string, size: { width: number; height: number }, mode: string, url: string, then?: (p: Page) => Promise<void>) {
  const ctx = await browser.newContext({ viewport: size, storageState: { cookies: [], origins: [] } });
  await ctx.addInitScript((m) => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-chart-mode-v1", m);
  }, mode);
  const page = await ctx.newPage();
  await page.goto(`${OK}${url}`);
  await mapReady(page);
  await page.waitForTimeout(2600);
  if (then) await then(page);
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
  await ctx.close();
}

test.describe("review media", () => {
  test.skip(!process.env.NX_MEDIA, "media only on request (NX_MEDIA=1)");
  test.describe.configure({ mode: "serial" });
  const moss = "/explore?place=port-592";
  const read = async (p: Page) => {
    await tapWater(p, 36.87, -122.02);
    await p.waitForTimeout(1800);
  };
  test("stills at the product's widths, in three modes", async ({ browser }) => {
    test.setTimeout(240_000);
    mkdirSync(OUT, { recursive: true });
    for (const [w, h] of [
      [1440, 900],
      [390, 844],
      [320, 640],
      [844, 390],
    ])
      await still(browser, `monterey-day-${w}x${h}`, { width: w, height: h }, "day", moss);
    await still(browser, "monterey-reading-day-1440x900", { width: 1440, height: 900 }, "day", moss, read);
    await still(browser, "monterey-reading-dusk-1440x900", { width: 1440, height: 900 }, "dusk", moss, read);
    await still(browser, "monterey-reading-night-1440x900", { width: 1440, height: 900 }, "night", moss, read);
    await still(browser, "monterey-dusk-390x844", { width: 390, height: 844 }, "dusk", moss);
    await still(browser, "monterey-night-390x844", { width: 390, height: 844 }, "night", moss);
    for (const layer of ["sst", "charm", "outlook"]) await still(browser, `monterey-${layer}-day-1440x900`, { width: 1440, height: 900 }, "day", `${moss}&layer=${layer}`);
    await still(browser, "monterey-charm-dusk-1440x900", { width: 1440, height: 900 }, "dusk", `${moss}&layer=charm`);
    await still(browser, "data-extent-day-1440x900", { width: 1440, height: 900 }, "day", "/explore");
    await still(browser, "statewide-day-1440x900", { width: 1440, height: 900 }, "day", "/explore", async (p) => {
      await p.evaluate(() => (window as unknown as { __cwMap: { jumpTo: (o: object) => void } }).__cwMap.jumpTo({ center: [-120.6, 37.1], zoom: 5.3 }));
      await p.waitForTimeout(1500);
    });
    await still(browser, "statewide-day-390x844", { width: 390, height: 844 }, "day", "/explore", async (p) => {
      await p.evaluate(() => (window as unknown as { __cwMap: { jumpTo: (o: object) => void } }).__cwMap.jumpTo({ center: [-120.3, 36.8], zoom: 5.2 }));
      await p.waitForTimeout(1500);
    });
    await still(browser, "harbour-zoom-day-1440x900", { width: 1440, height: 900 }, "day", moss, async (p) => {
      await p.evaluate(() => (window as unknown as { __cwMap: { jumpTo: (o: object) => void } }).__cwMap.jumpTo({ center: [-121.83, 36.81], zoom: 12 }));
      await p.waitForTimeout(1800);
    });
  });

  test("video: streamlines, then an hour changing and a frame cut", async ({ browser }) => {
    test.setTimeout(60_000);
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] }, recordVideo: { dir: test.info().outputDir, size: { width: 1440, height: 900 } } });
    await ctx.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
    const page = await ctx.newPage();
    await page.goto(`${OK}${moss}`);
    await mapReady(page);
    await page.evaluate(() => (window as unknown as { __cwMap: { jumpTo: (o: object) => void } }).__cwMap.jumpTo({ center: [-122.0, 36.82], zoom: 10.2 }));
    await page.waitForTimeout(4000);
    await page.getByTestId("nx-hour-hfr2km_currents_20261007T20Z").click();
    await page.waitForTimeout(3000);
    await page.getByTestId("nx-frame-olci:2026-10-05").click();
    await page.waitForTimeout(2500);
    await page.getByTestId("nx-frame-olci:latest").click();
    await page.waitForTimeout(2000);
    const video = page.video();
    await ctx.close();
    if (video) copyFileSync(await video.path(), path.join(OUT, "streamlines-and-frames-1440.webm"));
  });
});
