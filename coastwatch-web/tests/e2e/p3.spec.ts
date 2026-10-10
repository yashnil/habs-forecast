import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/** P3 map refinement: values next to ports belong to the forecast only, the map states what
 *  it shows and when, speed-class arrows, and the opt-in combined currents + chlorophyll view. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const OK = "http://localhost:3200";
const BASE = "/data/fixture/v1";
const olci = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "olci300_chl_latest");

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}
const source = (page: Page, id: string) =>
  page.evaluate((sid) => (window as unknown as { __cwMap: { getSource: (s: string) => { tiles?: string[] } | undefined } }).__cwMap.getSource(sid)?.tiles?.[0] ?? null, id);

test("forecast percentages next to ports appear only while the forecast is on the map", async ({ page }) => {
  await open(page, `${OK}/?region=monterey_bay`);
  await expect(page.getByTestId("port-row-593")).toContainText("%");
  await expect(page.getByTestId("region-ports").locator("..")).toContainText("C-HARM forecast");
  await page.getByTestId("group-satellite").click();
  await expect(page.getByTestId("port-row-593")).not.toContainText("%");
  await page.getByTestId("group-currents").click();
  await expect(page.getByTestId("port-row-593")).not.toContainText("%");
});

test("the map itself states what is shown and when", async ({ page }) => {
  await open(page, OK);
  await expect(page.getByTestId("stamp-forecast")).toContainText("C-HARM forecast for");
  await expect(page.getByTestId("stamp-forecast")).toContainText("issued");
  await page.getByTestId("group-satellite").click();
  await expect(page.getByTestId("stamp-satellite")).toContainText("pixels observed");
  await page.getByTestId("group-currents").click();
  await expect(page.getByTestId("stamp-currents")).toContainText("Currents, HF radar · observed");
  await expect(page.getByTestId("stamp-currents")).toContainText("(8 h ago)");
  await expect(page.getByTestId("map-stamp")).not.toContainText("forecast");
});

test("arrows are drawn per speed class, matching the legend", async ({ page }) => {
  await open(page, `${OK}/?layer=currents`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap: { getLayer: (l: string) => unknown } }).__cwMap.getLayer("currents-arrows-1"));
  const img = await page.evaluate(() => JSON.stringify((window as unknown as { __cwMap: { getLayoutProperty: (l: string, p: string) => unknown } }).__cwMap.getLayoutProperty("currents-arrows-1", "icon-image")));
  expect(img).toContain('"step"');
  for (let i = 0; i < 5; i++) expect(img).toContain(`cw-arrow-${i}`);
  const has = await page.evaluate(() => [0, 1, 2, 3, 4].every((i) => (window as unknown as { __cwMap: { hasImage: (n: string) => boolean } }).__cwMap.hasImage(`cw-arrow-${i}`)));
  expect(has).toBe(true);
});

test("combined view is opt-in and says what the two layers are and are not", async ({ page }) => {
  await open(page, `${OK}/?layer=currents`);
  expect(await source(page, "satellite")).toBeNull(); // currents alone by default
  await page.getByTestId("toggle-combined").check();
  await expect(page).toHaveURL(/chl=1/);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("satellite"));
  expect(await source(page, "satellite")).toBe(`${BASE}/${olci.tiles.url_template}`); // the published 300 m tiles, unchanged
  const notes = page.getByTestId("combined-notes");
  await expect(notes).toContainText("not toxin");
  await expect(notes).toContainText("observed surface motion, not a forecast");
  await expect(notes).toContainText("Different times");
  await expect(notes).toContainText("do not show where a bloom will travel");
  await expect(page.getByTestId("combined-chl-legend")).toContainText("Chlorophyll-a");
  await expect(page.getByTestId("currents-legend")).toBeVisible();
  await expect(page.getByTestId("stamp-satellite")).toBeVisible();
  await expect(page.getByTestId("stamp-currents")).toBeVisible();
  await expect(page.getByTestId("stamp-combined-gap")).toContainText("Different times: chlorophyll a median");
  // arrows one level sparser over the imagery
  const f = await page.evaluate(() => JSON.stringify((window as unknown as { __cwMap: { getFilter: (l: string) => unknown } }).__cwMap.getFilter("currents-arrows-1")));
  expect(f).toBe('[">=",["get","level"],2]');
  // the satellite raster stays opaque and nearest-sampled (legend colours = map colours)
  const paint = await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: { getPaintProperty: (l: string, p: string) => unknown } }).__cwMap;
    return [m.getPaintProperty("satellite-raster", "raster-opacity"), m.getPaintProperty("satellite-raster", "raster-resampling")];
  });
  expect(paint).toEqual([1, "nearest"]);
});

test("the inspector in the combined view reads both layers with their own times", async ({ page }) => {
  const port = { lat: 36.85, lon: -122.0 };
  await open(page, `${OK}/?layer=currents&chl=1&inspect=${port.lat},${port.lon}`);
  await expect(page.getByTestId("satellite-near")).toBeVisible();
  await expect(page.getByTestId("currents-near")).toBeVisible();
});

test("a region without radar coverage says so on the map", async ({ page }) => {
  // the North Coast fixture has no valid radar cell (true in the recording and live on 2026-10-09)
  await open(page, `${OK}/?region=north_coast&layer=currents`);
  await expect(page.getByTestId("stamp-currents-gap")).toContainText("No radar observations in North Coast this hour: no data, not calm water");
  await open(page, `${OK}/?region=monterey_bay&layer=currents`);
  await expect(page.getByTestId("stamp-currents-gap")).toHaveCount(0);
});
