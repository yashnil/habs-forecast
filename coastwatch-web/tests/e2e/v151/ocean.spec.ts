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
    await expect(page.getByRole("heading", { level: 1, name: "Explore California Waters: Chlorophyll" })).toBeVisible();
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

for (const size of [SIZES.desk, { width: 1280, height: 800 }, { width: 1100, height: 800 }, SIZES.phone, SIZES.small]) {
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
        for (const sel of ["[data-testid=nx-bar]", "[data-testid=nx-port-control]", "[data-testid=nx-menu-button]", ".nx-nav", ".nx-bar-end"]) {
          const id = sel;
          const r = await page.locator(sel).first().boundingBox();
          if (r && p.x >= r.x - 4 && p.x <= r.x + r.width + 4 && p.y >= r.y - 4 && p.y <= r.y + r.height + 4) return `under ${id}`;
        }
        // a phone's sheet spans the width; at a desk the time dock runs along the chart's foot
        const pane = await page.locator(size.width <= 640 ? ".n12-margin" : "[data-testid=nx-time-dock]").first().boundingBox();
        const m = { x: sheet.x + s.x, y: sheet.y + s.y };
        if (pane && m.y > pane.y && m.x >= pane.x && m.x <= pane.x + pane.width) return "Mexico under the sheet or dock";
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
  // the run's issue date is inferred in the data, so it is dated by its valid days only (round 2, C)
  await expect(page.getByTestId("nx-layer-charm")).toContainText(/Model · valid .+ \(expired\)/);
  await expect(page.getByTestId("nx-layer-charm")).not.toContainText("run of");
  await expect(legend).toHaveAttribute("data-product", "charm");
  await expect(page.getByTestId("nx-explore-lead")).toContainText("not measured toxin");
  await expect(legend).toContainText("P(pDA > 500 ng/L)");
  await expect(legend).toContainText(/valid Oct \d+, last run available/);
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
  // land inside the grid's box (inland, by Castroville) is outside the layer's area, never "cloud" (review j10)
  // (the first reading's ticket is closed first, so it cannot stand over the next point)
  await readout.getByRole("button", { name: "Close the reading" }).click();
  await tapWater(page, 36.72, -121.76);
  await expect(readout).toContainText("Outside this layer's area: land");
  await expect(readout).not.toContainText("cloud, fog or no pass");
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

test("round 2 (A): the plain chart is the first option, its elements switch independently, and the choice is kept", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  const first = page.getByTestId("nx-data-layers").getByRole("radio").first();
  await expect(first).toHaveAttribute("data-testid", "nx-layer-chart");
  await first.click();
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-base", "chart");
  await expect(page.getByTestId("nx-legend")).toHaveAttribute("data-product", "chart");
  await expect(page.getByTestId("nx-explore-lead")).toContainText("no data layer");
  await expect(page.getByTestId("nx-opacity")).toHaveCount(0);
  await expect(page).toHaveURL(/layer=chart/);
  // the chart's own elements, in Explore's one set of controls (Today's chips step aside)
  await expect(page.getByTestId("nx-layers")).toBeHidden();
  const depth = page.getByTestId("nx-depth-toggle"), reg = page.getByTestId("nx-regulation-toggle");
  await expect(depth).toHaveAttribute("aria-checked", "true");
  await reg.click();
  await expect(reg).toHaveAttribute("aria-checked", "true");
  await depth.click();
  await expect(depth).toHaveAttribute("aria-checked", "false");
  // overlays over the plain chart
  await page.getByTestId("nx-stations-toggle").click();
  await expect(page.locator('[data-testid^="nx-station-shore-"]').first()).toBeVisible();
  // kept on this device: a bare address opens the plain chart with the same elements
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-base", "chart");
  await expect(page.getByTestId("nx-depth-toggle")).toHaveAttribute("aria-checked", "false");
  await expect(page.getByTestId("nx-regulation-toggle")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("nx-stations-toggle")).toHaveAttribute("aria-checked", "true");
  // a data layer, its opacity kept too
  await page.getByTestId("nx-layer-chl").click();
  const slider = page.getByTestId("nx-opacity").locator("input");
  await slider.fill("60");
  await expect(page.getByTestId("nx-opacity")).toContainText("60 %");
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-base", "chl");
  await expect(page.getByTestId("nx-opacity")).toContainText("60 %");
  await ctx.close();
});

test("round 2 (E): the time rail in three families, each layer with its own latest point; an expired run says so", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?view=coast&place=${MOSS}`);
  await mapReady(page);
  await page.getByTestId("nx-rail-toggle").click();
  const rail = page.getByTestId("nx-timerail");
  for (const [id, label] of [["obs", "Observations"], ["est", "Analysis / Estimates"], ["fc", "Forecasts"]]) await expect(page.getByTestId(`nx-rail-family-${id}`)).toContainText(label);
  await expect(rail).not.toContainText(/\bNow\b/);
  await expect(page.getByTestId("nx-rail-latest-olci")).toContainText(/latest Oct \d+/);
  await expect(page.getByTestId("nx-rail-latest-charm")).toContainText(/valid to Oct \d+/);
  await expect(page.getByTestId("nx-rail-expired-charm")).toContainText(/Last run available: valid .+ · expired · .*not a current forecast/);
  // choosing a forecast frame from its own row makes C-HARM the layer on show
  await page.locator('[data-testid^="nx-frame-charm:"]').first().click();
  await expect(page.getByTestId("nx-legend")).toHaveAttribute("data-product", "charm");
  await ctx.close();
});

test("round 2 (B): chlorophyll's views: Observed (one day), Recent composite (pixel ages)", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?place=${MOSS}`);
  await mapReady(page);
  const views = page.getByTestId("nx-chl-views");
  await expect(page.getByTestId("nx-chl-composite")).toHaveAttribute("aria-checked", "true");
  await page.getByTestId("nx-chl-observed").click();
  await expect(page.getByTestId("nx-chl-observed")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", /^olci:\d{4}-\d{2}-\d{2}$/);
  await page.getByTestId("nx-chl-composite").click();
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", "olci:latest");
  await expect(views).toContainText("Estimate");
  await ctx.close();
});


const hook2 = (page: Page) => page.evaluate(() => (window as unknown as { __nxExplore?: { ocean?: string; oceanStep?: string } }).__nxExplore ?? {});

test("round 2 (B): the gap-filled estimate is drawn as an estimate, dated and said stale, on the Analysis / Estimates track", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?rail=1&place=${MOSS}`);
  await mapReady(page);
  const est = page.getByTestId("nx-chl-estimate");
  await expect(est).toBeEnabled({ timeout: 10_000 });
  await est.click();
  await expect(est).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("nx-legend")).toHaveAttribute("data-product", "chl-estimate");
  await expect(page.getByTestId("nx-model-label")).toContainText("Gap-filled estimate (NOAA DINEOF) · Nov 2 · reconstructed where satellites saw nothing, not observed");
  await expect(page.getByTestId("nx-model-label")).toContainText("stale");
  await expect(page.getByTestId("nx-explore-lead")).toContainText("not observed");
  await expect.poll(async () => (await hook2(page)).ocean, { timeout: 15_000 }).toBe("chlest");
  await expect(page.getByTestId("nx-rail-family-est").getByTestId("nx-rail-chl-estimate")).toContainText("stale");
  // a reading names it an estimate, never a measurement
  await tapWater(page, 36.8, -122.0);
  await expect(page.getByTestId("nx-readout")).toContainText(/Chlorophyll estimate here|No estimate value here/);
  await expect(page.getByTestId("nx-readout")).not.toContainText("Measured");
  await ctx.close();
});

test("round 2 (D): WCOFS model temperature and predicted currents, steps on the right tracks, an unreadable step said so", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?rail=1&place=${MOSS}`);
  await mapReady(page);
  const sst = page.getByTestId("nx-layer-sst-model");
  await expect(sst).toBeEnabled({ timeout: 10_000 });
  await expect(sst).toContainText("Model · WCOFS");
  await sst.click();
  await expect(page.getByTestId("nx-explore-lead")).toContainText("not a measurement");
  // it opens on the served step nearest to now, never on one that cannot be read
  await expect.poll(async () => (await hook2(page)).oceanStep, { timeout: 15_000 }).toMatch(/^(n024|f024)$/);
  await expect(page.getByTestId("nx-model-label")).not.toContainText("could not be read");
  // nowcast steps on Analysis / Estimates, forecast steps on Forecasts
  await expect(page.getByTestId("nx-rail-family-est").getByTestId("nx-step-n024")).toBeVisible();
  await expect(page.getByTestId("nx-rail-family-fc").getByTestId("nx-step-f024")).toBeVisible();
  // a step the fixture serves draws; its label is the model's own, a forecast not a measurement
  await page.getByTestId("nx-step-f024").click();
  await expect.poll(async () => (await hook2(page)).oceanStep, { timeout: 15_000 }).toBe("f024");
  await expect(page.getByTestId("nx-model-label")).toContainText(/NOAA WCOFS model guidance \(developmental\) · cycle Nov 14 03z · valid .* · a model forecast, not a measurement; not a bloom or toxin prediction/);
  await page.getByTestId("nx-step-n024").click();
  await expect(page.getByTestId("nx-model-label")).toContainText("a model nowcast");
  // a step the fixture does not serve: said, nothing drawn in its place
  await page.getByTestId("nx-step-f048").click();
  await expect(page.getByTestId("nx-model-label")).toContainText("could not be read");
  await expect(page.getByTestId("nx-step-f048")).toHaveAttribute("data-failed", "1");
  // predicted currents: a sub-switch beside the measured radar, their own key
  await page.getByTestId("nx-layer-chl").click();
  await page.getByTestId("nx-currents-toggle").click();
  await page.getByTestId("nx-currents-predicted").click();
  await page.getByTestId("nx-step-f024").click();
  await expect(page.getByTestId("nx-currents-predicted-key")).toContainText("Predicted currents · NOAA WCOFS model guidance (developmental)");
  await expect(page.getByTestId("nx-currents-key")).toHaveCount(0);
  await page.getByTestId("nx-currents-measured").click();
  await expect(page.getByTestId("nx-currents-predicted-key")).toHaveCount(0);
  await ctx.close();
});


test("round 2 (E): every row sits on its true times on one shared axis; WCOFS plays only over served steps", async ({ browser }) => {
  const { ctx, page } = await newContext(browser, SIZES.desk);
  await page.goto(`${OK}/explore?rail=1&place=${MOSS}`);
  await mapReady(page);
  await expect(page.getByTestId("nx-step-n024")).toBeVisible({ timeout: 15_000 });
  const box = async (id: string) => (await page.getByTestId(id).boundingBox())!;
  const rail = await box("nx-timerail");
  const n = await box("nx-step-n024"), f = await box("nx-step-f024"), last = await box("nx-step-f072");
  // narrow 3-hour ticks at their valid times: the nowcast before the forecast, the last forecast inside the axis
  expect(n.width).toBeLessThan(12);
  expect(n.x).toBeLessThan(f.x);
  expect(last.x + last.width).toBeLessThanOrEqual(rail.x + rail.width + 1);
  // the estimate's single day sits at its date, left of the model's steps (Nov 2 before Nov 13)
  const est = await box("nx-rail-chl-estimate-day");
  expect(est.x + est.width).toBeLessThan(n.x);
  // nothing stretched: the nowcast row's ticks span about a day, not the whole axis
  const first = await box("nx-step-n003");
  expect(n.x - first.x).toBeLessThan(rail.width / 4);
  // playback over the forecast steps: an unserved step is skipped, never shown as "could not be read"
  await page.getByTestId("nx-layer-sst-model").click();
  await expect.poll(async () => (await hook2(page)).oceanStep, { timeout: 15_000 }).toMatch(/^(n024|f024)$/);
  await page.getByTestId("nx-play").first().click();
  await expect.poll(async () => (await hook2(page)).oceanStep, { timeout: 30_000 }).toBe("f024");
  await expect(page.getByTestId("nx-model-label")).not.toContainText("could not be read");
  await ctx.close();
});
