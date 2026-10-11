import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** C-HARM at the coastline: data rasters stop at the drawn shoreline (a land cover above them,
 *  values unchanged), cells without a value keep the no-data marking, and the readout says why
 *  a cell is empty. Fixture data; cells chosen from the fixture grids. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}
const layerIds = (page: Page) => page.evaluate(() => (window as unknown as { __cwMap: { getStyle(): { layers: { id: string }[] } } }).__cwMap.getStyle().layers.map((l) => l.id));

async function hover(page: Page, lon: number, lat: number) {
  await page.evaluate(([x, y]) => (window as unknown as { __cwMap: { jumpTo(o: unknown): void } }).__cwMap.jumpTo({ center: [x, y], zoom: 10 }), [lon, lat]);
  await page.waitForTimeout(300);
  const box = (await page.getByTestId("map").boundingBox())!;
  const pt = await page.evaluate(([x, y]) => (window as unknown as { __cwMap: { project(p: number[]): { x: number; y: number } } }).__cwMap.project([x, y]), [lon, lat]);
  // arrive in steps, as a pointer does: a single jump is not a hover
  await page.mouse.move(box.x + pt.x - 6, box.y + pt.y - 6);
  await page.mouse.move(box.x + pt.x, box.y + pt.y, { steps: 3 });
}

test("data rasters sit under the coastal land cover, which sits under the coastline and labels", async ({ page }) => {
  await open(page, `${OK}/?layer=forecast&var=particulate_domoic&lead=1`);
  await expect.poll(async () => (await layerIds(page)).includes("forecast-raster")).toBe(true);
  const ids = await layerIds(page);
  const at = (id: string) => ids.indexOf(id);
  expect(at("coastal-land")).toBeGreaterThan(at("forecast-raster"));
  expect(at("coastal-land")).toBeGreaterThan(at("water-nodata"));
  expect(at("coastline")).toBeGreaterThan(at("coastal-land"));
  expect(at("water-name")).toBeGreaterThan(at("coastal-land"));
  // the forecast itself is unchanged: opaque and nearest-sampled
  const paint = await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: { getPaintProperty(l: string, p: string): unknown } }).__cwMap;
    return [m.getPaintProperty("forecast-raster", "raster-opacity"), m.getPaintProperty("forecast-raster", "raster-resampling")];
  });
  expect(paint).toEqual([1, "nearest"]);
});

test("the land cover assets are served and describe their coastline source", async ({ request }) => {
  for (const f of ["coastal-land.webp", "coastal-land-flat.webp"]) expect((await request.get(`${OK}/basemap/${f}`)).ok()).toBe(true);
  const meta = await (await request.get(`${OK}/basemap/coastal-land.json`)).json();
  expect(meta.coastline_license).toContain("OpenStreetMap");
  expect(meta.corners_lnglat).toEqual((await (await request.get(`${OK}/basemap/relief.json`)).json()).corners_lnglat);
});

test("an empty toxin cell next to the shore says why; a cell C-HARM does not model says so", async ({ page }) => {
  const pn = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_pseudo_nitzschia_lead0").grid;
  // fixture cell with a Pseudo-nitzschia value and no particulate domoic acid value (row 33, col 29)
  const lat = pn.lat_first + pn.lat_step * 33;
  const lon0 = pn.lon_first + pn.lon_step * 29;
  const lon = lon0 > 180 ? lon0 - 360 : lon0;
  await open(page, `${OK}/?layer=forecast&var=particulate_domoic&lead=0`);
  await hover(page, lon, lat);
  await expect(page.getByTestId("hover-none")).toContainText("within about 3–6 km of shore");
  await expect(page.getByTestId("hover-none")).toContainText("Pseudo-nitzschia value here");
  // a cell with no value in any C-HARM layer
  await hover(page, -121.59, 36.07);
  await expect(page.getByTestId("hover-none")).toContainText("water C-HARM does not model");
});
