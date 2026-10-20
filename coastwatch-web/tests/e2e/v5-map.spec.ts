import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * v5 map: the coast filled honestly. The C-HARM layer extends the nearest NOAA cell to the shore
 * (labelled context, never a value), says "outside NOAA's model" in the bays and shows chlorophyll
 * there; the derived index (tests/fixture-data/v1/derived) adds the 8-day outlook and the
 * gap-filling estimate, each with its provenance, and nothing breaks without it.
 */
const NOW = "2026-10-13T18:00:00Z";
type Win = { __cwMap: { jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number }; loaded(): boolean; getSource(id: string): unknown; getStyle(): { layers: { id: string }[] }; once(e: string, f: () => void): void; areTilesLoaded(): boolean } };

async function open(page: Page, q: string, base = OK) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${base}/?${q}`);
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
}

async function jump(page: Page, lat: number, lon: number, zoom: number) {
  await page.evaluate(([la, lo, z]) => (window as unknown as Win).__cwMap.jumpTo({ center: [lo, la], zoom: z }), [lat, lon, zoom]);
  await page.waitForFunction(() => (window as unknown as Win).__cwMap.loaded() && (window as unknown as Win).__cwMap.areTilesLoaded(), null, { timeout: 15_000 });
}

async function hoverAt(page: Page, lat: number, lon: number) {
  const p = await page.evaluate(([la, lo]) => (window as unknown as Win).__cwMap.project([lo, la]), [lat, lon]);
  const box = (await page.getByTestId("map").boundingBox())!;
  await page.mouse.move(box.x + p.x, box.y + p.y);
  await page.mouse.move(box.x + p.x + 1, box.y + p.y);
}

test("the forecast tints the water NOAA does not model near shore; the nearest value is text only, with its pier-sample check", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=1");
  await expect(page.getByTestId("legend-outside")).toContainText("Outside NOAA’s model near shore");
  await expect(page.getByTestId("legend-outside")).toContainText("no probability is shown there");
  await expect(page.getByTestId("legend-context-check")).toContainText("1,058 CalHABMAP pier samples");
  await expect(page.getByTestId("legend-context-check")).toContainText("no demonstrated skill at the shore");
  // nearshore cells of the fixture without a pDA value, next to one with a value (Monterey Bay's east shore)
  await jump(page, 36.78, -121.82, 11);
  const readout = page.getByTestId("hover-readout");
  let found = false;
  for (const [lat, lon] of [[36.76, -121.8], [36.73, -121.8], [36.79, -121.77], [36.82, -121.77], [36.7, -121.83]]) {
    await hoverAt(page, lat, lon);
    await expect(readout).toBeVisible();
    if ((await readout.getAttribute("data-tone")) === "context") {
      found = true;
      break;
    }
  }
  expect(found).toBe(true);
  await expect(page.getByTestId("hover-context")).toContainText(/^NOAA's model does not reach this water\. The nearest NOAA cell, \d+\.\d km away, reads \d+%: text only, not a forecast for this water\./);
  await expect(page.getByTestId("hover-context")).toContainText("no demonstrated skill at the shore");
  await expect(page.getByTestId("hover-label")).toHaveText("Outside NOAA's model near shore");
  // never a value of its own
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
});

test("San Francisco Bay: NOAA's model does not reach it; the inspector points to pier samples and the filled chlorophyll", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=1");
  // no chlorophyll or probability is painted under the forecast: one neutral tint
  expect(await page.evaluate(() => (window as unknown as Win).__cwMap.getStyle().layers.map((l) => l.id).filter((id) => id.startsWith("under-")))).toEqual([]);
  await jump(page, 37.75, -122.3, 10.5);
  await hoverAt(page, 37.75, -122.3);
  await expect(page.getByTestId("hover-readout")).toHaveAttribute("data-tone", "outside");
  await expect(page.getByTestId("hover-outside")).toContainText("NOAA's model does not reach this water");
  const box = (await page.getByTestId("map").boundingBox())!;
  const p = await page.evaluate(() => (window as unknown as Win).__cwMap.project([-122.3, 37.75]));
  await page.mouse.click(box.x + p.x, box.y + p.y);
  const out = page.getByTestId("cell-outside");
  await expect(out).toContainText("Outside NOAA’s model near shore");
  await expect(page.getByTestId("cell-value")).toHaveText("No value");
  await expect(page.getByTestId("cell-piers")).toContainText("CalHABMAP pier samples");
  await page.getByTestId("switch-estimate").click();
  // the selection stays open; the folded layer deck names the new layer
  await expect(page).toHaveURL(/layer=olci300/);
  await expect(page.getByTestId("layer-capsule")).toContainText("Satellite chlorophyll");
});

test("the 8-day outlook: its own layer, horizon, skill against baselines, caveats and readout label", async ({ page }) => {
  await open(page, "layer=outlook");
  await expect(page.getByTestId("layer-outlook")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-outlook")).toContainText("Outlook, not observed");
  await expect(page.getByTestId("outlook-horizon")).toBeVisible();
  await expect(page.getByTestId("outlook-issued")).toHaveText("Issued Oct 12");
  await expect(page.getByTestId("outlook-window")).toHaveText("Mean Oct 13 → Oct 20");
  await expect(page.getByTestId("stamp-outlook")).toContainText("Outlook, not observed.");
  const skill = page.getByTestId("outlook-skill");
  await expect(skill).toContainText("lower is better");
  for (const v of ["0.754", "0.764", "0.850", "0.892"]) await expect(skill).toContainText(v);
  await expect(page.getByTestId("derived-caveats")).toContainText("not toxin");
  await expect(page.getByTestId("legend-outlook")).toContainText("Outlook, not observed.");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("outlook"))).toBe(true);
  await jump(page, 36.85, -122.1, 9.5);
  await hoverAt(page, 36.85, -122.1);
  await expect(page.getByTestId("hover-label")).toHaveText("Outlook, not observed");
  await expect(page.getByTestId("hover-value")).toHaveText(/^[\d.]+$/);
  await expect(page.getByTestId("hover-readout")).toContainText(/typically [\d.]+–[\d.]+ mg\/m³/);
});

test("the estimate lies under the observed pixels, stippled and labelled, and can be turned off", async ({ page }) => {
  await open(page, "layer=olci300");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(true);
  const order = await page.evaluate(() => (window as unknown as Win).__cwMap.getStyle().layers.map((l) => l.id));
  expect(order.indexOf("estimate-raster")).toBeLessThan(order.indexOf("satellite-raster"));
  await expect(page.getByTestId("legend-estimate")).toContainText("Estimated, not observed.");
  await page.getByTestId("estimate-provenance").locator("summary").click();
  await expect(page.getByTestId("estimate-validation")).toContainText("0.6–3 km");
  await expect(page.getByTestId("estimate-method")).toBeVisible();
  // SF Bay has no observed pixel in the fixture: the estimate is what is read there, so labelled
  await jump(page, 37.75, -122.3, 10.5);
  await hoverAt(page, 37.75, -122.3);
  await expect(page.getByTestId("hover-label")).toHaveText("Estimated, not observed");
  await page.getByTestId("sat-estimate-toggle").uncheck();
  await expect(page).toHaveURL(/est=0/);
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(false);
});

test("without a derived index nothing derived is offered and the map still works", async ({ page }) => {
  await open(page, "layer=olci300", FAILED);
  await expect(page.getByTestId("layer-satellite")).toBeVisible();
  await expect(page.getByTestId("layer-outlook")).toHaveCount(0);
  await expect(page.getByTestId("legend-estimate")).toHaveCount(0);
  expect(await page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(false);
  // a link to the outlook or to sea temperature falls back to the default layer
  for (const q of ["layer=outlook", "layer=sst&sel=harbor:593"]) {
    await open(page, q, FAILED);
    await expect(page).not.toHaveURL(/layer=(outlook|sst)/);
    await expect(page.getByTestId("layer-sst")).toHaveCount(0);
  }
});

test("on a phone the outlook is one chip away and its horizon reads", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await page.getByTestId("chip-outlook").first().click();
  await expect(page.getByTestId("chip-outlook").first()).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("outlook-horizon").first()).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await ctx.close();
});

test("sea temperature: an observation, a °F legend, °C and °F in the readout, a perceptually even ramp", async ({ page }) => {
  await open(page, "layer=sst");
  await expect(page.getByTestId("layer-sst")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-sst")).toContainText("Observed Oct 12");
  await expect(page.getByTestId("legend-sst")).toContainText("°F");
  await expect(page.getByTestId("stamp-sst")).toContainText("Observation.");
  await expect(page.getByTestId("stamp-sst")).toContainText("NOAA CoastWatch VIIRS SST");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("sst"))).toBe(true);
  await jump(page, 36.85, -122.2, 9.5);
  await hoverAt(page, 36.85, -122.2);
  await expect(page.getByTestId("hover-label")).toHaveText("Observation");
  await expect(page.getByTestId("hover-readout")).toContainText(/[\d.]+\s*°C/);
  await expect(page.getByTestId("hover-readout")).toContainText(/≈ [\d.]+ °F/);
});

test("My Coast's sea-temperature link opens the layer with the harbour selected", async ({ page }) => {
  await open(page, "layer=sst&sel=harbor:593");
  await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
  await expect(page.getByTestId("layer-capsule")).toContainText("Sea temperature");
  await expect(page).toHaveURL(/layer=sst/);
});
