import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";
import { newContext, seedTrip, SIZES, MOSS } from "../v15/helpers";

/**
 * v15.1 Explore California Waters (DIRECTION §1, §3): the entry on Today, the flight to the whole coast (Oregon border
 * to Mexico) at desk and phone widths, every live layer with its legend, units and dates, time playback, a reading
 * tied to a real published cell, and the ways back to the port and to the saved trip.
 */
const DATA = path.resolve(__dirname, "../../../public/data/fixture/v1");
type G = { url: string; width: number; lat_first: number; lat_step: number; lon_first: number; lon_step: number; scale_factor: number; add_offset: number; nodata: number; chunks?: { url_template: string } | null };
const manifest = JSON.parse(readFileSync(path.join(DATA, "manifest.json"), "utf8")) as { layers: Array<{ layer_id: string; grid: G | null }> };

/** The published chlorophyll under a point, read from the fixture's own grid file. */
function publishedChl(lat: number, lon: number): number | null {
  const g = manifest.layers.find((l) => l.layer_id === "olci300_chl_latest")!.grid!;
  const buf = gunzipSync(readFileSync(path.join(DATA, (g.chunks?.url_template ?? g.url).replace("{row}", "0").replace("{col}", "0"))));
  const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5), col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  const code = buf.readUInt16LE((row * g.width + col) * 2);
  return code === g.nodata ? null : 10 ** (code * g.scale_factor + g.add_offset);
}

const mapReady = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
const scene = (page: Page) => page.evaluate(() => document.documentElement.dataset.nxScene ?? null);
const bounds = (page: Page) =>
  page.evaluate(() => {
    const b = (window as unknown as { __cwMap: { getBounds: () => { getSouth: () => number; getNorth: () => number } } }).__cwMap.getBounds();
    return { s: b.getSouth(), n: b.getNorth() };
  });
const moving = (page: Page) => page.evaluate(() => (window as unknown as { __cwMap: { isMoving: () => boolean } }).__cwMap.isMoving());
const hook = (page: Page) => page.evaluate(() => (window as unknown as { __nxExplore?: { product: string; frame: string; at: { lat: number; lon: number } | null; cell: { lat: number; lon: number } | null } }).__nxExplore ?? null);

async function tapWater(page: Page, lat: number, lon: number) {
  const p = await page.evaluate(([la, lo]) => (window as unknown as { __cwMap: { project: (x: [number, number]) => { x: number; y: number } } }).__cwMap.project([lo, la]), [lat, lon]);
  const box = (await page.getByTestId("nx-map").boundingBox())!;
  await page.mouse.click(box.x + p.x, box.y + p.y);
}

for (const [name, size] of [
  ["desk 1440", SIZES.desk],
  ["phone 390", SIZES.phone],
] as const) {
  test(`${name}: Today's entry flies to the whole coast, Oregon border to Mexico, and back to the port`, async ({ browser }) => {
    const { ctx, page, errors } = await newContext(browser, size);
    await page.goto(`${OK}/today?place=${MOSS}`);
    await mapReady(page);
    const entry = page.getByTestId("nx-today-explore");
    await expect(entry).toBeVisible();
    await expect(entry).toContainText("Explore California Waters");
    await entry.click();
    await expect(page).toHaveURL(/\/explore\?view=coast&place=port-592/, { timeout: 15_000 });
    await expect.poll(() => scene(page), { timeout: 15_000 }).toBe("explore:state");
    // the camera leaves once Today's ticket has shrunk away, then lands on the whole coast
    await expect.poll(async () => { const b = await bounds(page); return b.s <= 32.53 && b.n >= 42 && !(await moving(page)); }, { timeout: 15_000 }).toBe(true);
    await expect(page.getByTestId("nx-explore-heading")).toHaveText("Explore California Waters");
    // the port is kept: one tap back to its water
    await page.getByTestId("nx-back-port").click();
    await expect.poll(() => scene(page), { timeout: 15_000 }).toBe("explore:port-592");
    await expect(page).not.toHaveURL(/view=coast/);
    await expect(page.getByTestId("nx-whole-coast")).toBeVisible();
    // and Back to Today returns to the port's Today
    await page.getByTestId("nx-back-from").click();
    await expect(page).toHaveURL(/\/today\?place=port-592/);
    expect(errors).toEqual([]);
    await ctx.close();
  });
}

for (const size of [SIZES.phone, SIZES.small]) {
  test(`statewide at ${size.width}: the Oregon border and the Mexican border are uncovered`, async ({ browser }) => {
    const { ctx, page } = await newContext(browser, size);
    await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
    await mapReady(page);
    const at = (lon: number, lat: number) => page.evaluate(([lo, la]) => (window as unknown as { __cwMap: { project: (x: [number, number]) => { x: number; y: number } } }).__cwMap.project([lo, la]), [lon, lat]);
    const sheet = (await page.getByTestId("nx-sheet").boundingBox())!;
    await expect
      .poll(async () => {
        const n = await at(-124.21, 42.0), s = await at(-117.12, 32.53);
        const p = { x: sheet.x + n.x, y: sheet.y + n.y };
        for (const id of ["nx-bar", "nx-port-control", "nx-menu-button"]) {
          const r = await page.getByTestId(id).boundingBox();
          if (r && p.x >= r.x - 4 && p.x <= r.x + r.width + 4 && p.y >= r.y - 4 && p.y <= r.y + r.height + 4) return `under ${id}`;
        }
        const pane = await page.locator(".n12-margin").first().boundingBox();
        if (pane && sheet.y + s.y > pane.y) return "Mexico under the sheet";
        return "clear";
      }, { timeout: 15_000 })
      .toBe("clear");
    await ctx.close();
  });
}

test("Explore is in the navigation, its own section, and opens on the whole coast", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/today?place=${MOSS}`);
  await mapReady(page);
  await page.getByTestId("nx-nav-explore").click();
  await expect(page).toHaveURL(/\/explore\?view=coast&place=port-592/);
  await expect(page.getByTestId("nx-nav-explore")).toHaveAttribute("aria-current", "page");
  await ctx.close();
});

test("every live layer shows its legend, units and dates; what is not live is never offered as live", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  const legend = page.getByTestId("nx-legend");
  // chlorophyll: an observation, mg/m³, its dates, algae not toxin
  await expect(page.getByTestId("nx-layer-chl")).toHaveAttribute("data-class", "observation");
  await expect(page.getByTestId("nx-layer-chl")).toContainText(/Observation · Oct/);
  await expect(legend).toContainText("mg/m³");
  await expect(legend).toContainText(/Oct \d/);
  await expect(legend).toContainText("Algae, not toxin");
  // one short line in view, the rest one tap away in "About this layer"
  await expect(page.getByTestId("nx-explore-lead")).toHaveText("Algae seen from space · not toxin");
  await expect(page.getByTestId("nx-resolution-note")).toBeHidden();
  await page.getByTestId("nx-about").locator("summary").click();
  // statewide, the 300 m grid is finer than the screen: said with the numbers
  await expect(page.getByTestId("nx-resolution-note")).toContainText(/cells of 278 m/);
  await expect(page.getByTestId("nx-about")).toContainText("Sentinel-3 OLCI");
  // C-HARM: a model probability with its run and valid dates
  await page.getByTestId("nx-layer-charm").click();
  await expect(page.getByTestId("nx-layer-charm")).toHaveAttribute("data-class", "model");
  await expect(page.getByTestId("nx-layer-charm")).toContainText(/Model · run of .+, valid/);
  await expect(legend).toHaveAttribute("data-product", "charm");
  await expect(page.getByTestId("nx-explore-lead")).toContainText("not measured toxin");
  await expect(legend).toContainText("P(pDA > 500 ng/L)");
  await expect(legend).toContainText(/valid Oct \d+, run of Oct \d+/);
  await expect(legend).toContainText("not toxin in seafood");
  // currents: measured HF radar, its hour, and the hour's real speeds in kn and cm/s
  await page.getByTestId("nx-layer-chl").click();
  await page.getByTestId("nx-currents-toggle").click();
  await expect(page.getByTestId("nx-currents-key")).toContainText("HF radar");
  await expect(page.getByTestId("nx-currents-key")).toContainText(/Oct \d+, \d+:\d+/);
  await expect(page.getByTestId("nx-currents-speed")).toContainText(/median [\d.]+ kn \(\d+ cm\/s\)/);
  await expect(page.getByTestId("nx-currents-key")).toContainText("No streak is no data, not calm");
  // stations: points on the chart; a pier opens its own dated sample
  await page.getByTestId("nx-stations-toggle").click();
  await expect(page.getByTestId("nx-stations-key")).toBeVisible();
  const pier = page.locator('[data-testid^="nx-station-shore-"]').first();
  await expect(pier).toBeVisible();
  // statewide, neighbouring piers overlap: every point is a button, opened here from the keyboard
  await pier.focus();
  await page.keyboard.press("Enter");
  const card = page.getByTestId("nx-station-readout");
  await expect(card).toContainText("Observation · CalHABMAP shore station");
  await expect(card.getByTestId("nx-station-value").first()).toBeVisible();
  await expect(card).toContainText(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d+/);
  await ctx.close();
});

test("time: play steps through the published days", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}&frame=olci:2026-10-01`);
  await mapReady(page);
  const start = (await hook(page))!.frame;
  await page.getByTestId("nx-play").first().click();
  await expect.poll(async () => (await hook(page))?.frame, { timeout: 8_000 }).not.toBe(start);
  await page.getByTestId("nx-play").first().click();
  await ctx.close();
});

test("a tap on the water reads the real published cell", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  // drill down from the state to the bay, then read a cell
  await page.getByTestId("nx-back-port").click();
  await expect.poll(() => scene(page), { timeout: 15_000 }).toBe("explore:port-592");
  await expect.poll(() => moving(page), { timeout: 10_000 }).toBe(false);
  await tapWater(page, 36.87, -122.02);
  const readout = page.getByTestId("nx-readout");
  await expect(readout).toContainText("Chlorophyll in this 300 m cell");
  const at = (await hook(page))!.at!;
  const expected = publishedChl(at.lat, at.lon);
  expect(expected).not.toBeNull();
  const shown = await readout.locator("p").filter({ hasText: "mg/m³" }).first().innerText();
  expect(Number(/([\d.]+)\s*mg/.exec(shown)![1])).toBeCloseTo(Number(expected!.toPrecision(2)), 6);
  await ctx.close();
});

test("from a saved trip: Back to your trip returns to that trip", async ({ browser }) => {
  const seeded = await seedTrip();
  const { ctx, page } = await newContext(browser, SIZES.desk, seeded.storage);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}&from=trip&trip=${seeded.trip.id}`);
  await mapReady(page);
  const back = page.getByTestId("nx-back-from");
  await expect(back).toHaveText("Back to your trip");
  await back.click();
  await expect(page).toHaveURL(new RegExp(`/coast/trips\\?id=${seeded.trip.id}`));
  await expect(page.getByTestId("trip-workspace")).toBeVisible();
  await ctx.close();
});
