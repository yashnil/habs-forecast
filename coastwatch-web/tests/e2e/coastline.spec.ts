import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** C-HARM at the coastline: data rasters are clipped to the basemap's water (values unchanged),
 *  cells without a value keep the no-data marking, and the readout says why a cell is empty.
 *  Fixture data; cells chosen from the fixture grids. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(url);
  await expect(page.getByTestId("notices-button")).toBeVisible();
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}

async function hover(page: Page, lon: number, lat: number) {
  await page.evaluate(([x, y]) => (window as unknown as { __cwMap: { jumpTo(o: unknown): void } }).__cwMap.jumpTo({ center: [x, y], zoom: 10 }), [lon, lat]);
  await page.waitForTimeout(300);
  const box = (await page.getByTestId("map").boundingBox())!;
  const pt = await page.evaluate(([x, y]) => (window as unknown as { __cwMap: { project(p: number[]): { x: number; y: number } } }).__cwMap.project([x, y]), [lon, lat]);
  // arrive in steps, as a pointer does: a single jump is not a hover
  await page.mouse.move(box.x + pt.x - 6, box.y + pt.y - 6);
  await page.mouse.move(box.x + pt.x, box.y + pt.y, { steps: 3 });
}

type MapLike = {
  jumpTo(o: unknown): void;
  project(p: number[]): { x: number; y: number };
  loaded(): boolean;
  queryRenderedFeatures(p: number[], o: unknown): unknown[];
  setLayoutProperty(l: string, p: string, v: string): void;
  getLayer(l: string): unknown;
  getCanvas(): HTMLCanvasElement;
  once(e: string, f: () => void): void;
  triggerRepaint(): void;
};

test("forecast cells that straddle the shore are drawn over water only, at the basemap's shoreline", async ({ page }) => {
  // renders up to 40 cells at zoom 12 twice (software WebGL in CI, two workers)
  test.setTimeout(120_000);
  const layer = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_pseudo_nitzschia_lead1");
  const g = layer.grid;
  const codes = new Uint16Array(new Uint8Array(gunzipSync(readFileSync(path.join(FIX, g.url)))).buffer);
  const ok = (r: number, c: number) => r >= 0 && r < g.height && c >= 0 && c < g.width && codes[r * g.width + c] !== g.nodata;
  // cells with a value next to a cell without one: the coastal edge of the model
  const coastal: [number, number][] = [];
  for (let r = 1; r < g.height - 1; r++) for (let c = 1; c < g.width - 1; c++) if (ok(r, c) && (!ok(r + 1, c) || !ok(r - 1, c) || !ok(r, c + 1) || !ok(r, c - 1))) coastal.push([r, c]);
  await open(page, `${OK}/?layer=forecast&var=pseudo_nitzschia&lead=1`);
  await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
    for (const id of ["forecast-cell-edges", "coastline"]) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", "none");
  });
  let straddling = 0, landPts = 0, waterPts = 0;
  for (const [r, c] of coastal.filter((_, i) => i % 3 === 0).slice(0, 40)) {
    if (straddling >= 3) break;
    const lat = g.lat_first + r * g.lat_step, lon = g.lon_first + c * g.lon_step;
    // in-page: sample a 7 x 7 lattice inside the cell; for each point, is it water on the basemap,
    // and does the forecast change its pixel (forecast layer on vs off)?
    const res = await page.evaluate(
      async ([lat, lon, step]) => {
        const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
        m.jumpTo({ center: [lon, lat], zoom: 12 });
        const idle = () => new Promise<void>((res) => { m.once("idle", () => res()); m.triggerRepaint(); });
        await idle();
        const grab = async () => {
          await new Promise<void>((res) => { m.once("render", () => res()); m.triggerRepaint(); });
          const c = m.getCanvas();
          const k = document.createElement("canvas");
          k.width = c.width; k.height = c.height;
          const g2 = k.getContext("2d")!;
          g2.drawImage(c, 0, 0);
          return { d: g2.getImageData(0, 0, k.width, k.height).data, w: k.width, s: c.width / c.clientWidth };
        };
        const pts: { x: number; y: number; water: boolean }[] = [];
        for (let i = 1; i <= 7; i++)
          for (let j = 1; j <= 7; j++) {
            const p = m.project([lon - step / 2 + (step * j) / 8, lat - step / 2 + (step * i) / 8]);
            const w = (dx: number, dy: number) => m.queryRenderedFeatures([p.x + dx, p.y + dy], { layers: ["water"] }).length > 0;
            const around = [w(0, 0), w(2, 0), w(-2, 0), w(0, 2), w(0, -2)];
            // the shoreline pixel itself is anti-aliased: classify only points clear of it
            if (around.every(Boolean) || !around.some(Boolean)) pts.push({ x: p.x, y: p.y, water: around[0] });
          }
        const on = await grab();
        m.setLayoutProperty("forecast-raster", "visibility", "none");
        await idle();
        const off = await grab();
        m.setLayoutProperty("forecast-raster", "visibility", "visible");
        await idle();
        return pts.map((p) => {
          const i = (Math.round(p.y * on.s) * on.w + Math.round(p.x * on.s)) * 4;
          const diff = Math.abs(on.d[i] - off.d[i]) + Math.abs(on.d[i + 1] - off.d[i + 1]) + Math.abs(on.d[i + 2] - off.d[i + 2]);
          return { water: p.water, drawn: diff > 12 };
        });
      },
      [lat, lon, g.lat_step] as const,
    );
    const land = res.filter((p) => !p.water), water = res.filter((p) => p.water);
    if (!land.length || !water.length) continue;
    straddling++;
    landPts += land.length;
    waterPts += water.length;
    // no forecast colour on land; the forecast on the cell's water (its value applies there)
    expect(land.filter((p) => p.drawn), `forecast drawn on land in cell ${r},${c}`).toEqual([]);
    expect(water.every((p) => p.drawn), `forecast missing on water in cell ${r},${c}`).toBe(true);
  }
  expect(straddling, `straddling cells checked (${landPts} land and ${waterPts} water points)`).toBeGreaterThan(0);
});

test("an empty toxin cell next to the shore says why; a cell C-HARM does not model says so", async ({ page }) => {
  const pn = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_pseudo_nitzschia_lead0").grid;
  // fixture cell with a Pseudo-nitzschia value and no particulate domoic acid value (row 33, col 29)
  const lat = pn.lat_first + pn.lat_step * 33;
  const lon0 = pn.lon_first + pn.lon_step * 29;
  const lon = lon0 > 180 ? lon0 - 360 : lon0;
  await open(page, `${OK}/?layer=forecast&var=particulate_domoic&lead=0`);
  await hover(page, lon, lat);
  // v5: the cell is tinted "outside NOAA's model near shore"; the reason stays
  const ro = page.getByTestId("hover-readout");
  await expect(ro).toContainText("within about 3–6 km of shore");
  await expect(ro).toContainText("Pseudo-nitzschia value here");
  await expect(ro).toContainText("NOAA's model does not reach this water");
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
  // a cell with no value in any C-HARM layer: never a value; the published reason, context or outside
  await hover(page, -121.59, 36.07);
  await expect(ro).toContainText(/water C-HARM does not model|NOAA's model does not reach this water/);
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
});
