import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** C-HARM drawn from the value grid: every drawn cell edge sits on the true cell boundary (within
 *  a screen pixel at zoom 12), and the readout on either side of a drawn edge reports the cell
 *  drawn there. Fixture data. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const layer = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_particulate_domoic_lead1");
const g = layer.grid;
const codes = new Uint16Array(new Uint8Array(gunzipSync(readFileSync(path.join(FIX, g.url)))).buffer);
const cls = (code: number) => (code === g.nodata ? -1 : Math.min(9, Math.floor(code * g.scale_factor * 10 + 1e-9)));
type Win = { __cwMap: { jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number }; loaded(): boolean; getSource(id: string): { type: string } | undefined; setLayoutProperty(l: string, p: string, v: string): void; getLayer(l: string): unknown } };

async function open(page: Page) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(`${OK}/?layer=forecast&var=particulate_domoic&lead=1`);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
}

/** Vertically adjacent cells of different colour classes (measured down the middle of the cell). */
function boundaries(n: number) {
  const out: { lat: number; lon: number; north: number; south: number }[] = [];
  for (let r = 1; r < g.height && out.length < n; r++)
    for (let c = 0; c < g.width && out.length < n; c++) {
      const a = codes[(r - 1) * g.width + c], b = codes[r * g.width + c];
      if (cls(a) < 0 || cls(b) < 0 || cls(a) === cls(b)) continue;
      if (out.some((o) => Math.abs(o.lat - (g.lat_first + (r - 0.5) * g.lat_step)) < 1e-9)) continue; // one per row
      const [north, south] = g.lat_step > 0 ? [b, a] : [a, b];
      out.push({ lat: g.lat_first + (r - 0.5) * g.lat_step, lon: g.lon_first + c * g.lon_step, north, south });
    }
  return out;
}

test("the forecast is drawn from the value grid, not the published image", async ({ page }) => {
  await open(page);
  expect(await page.evaluate(() => (window as unknown as Win).__cwMap.getSource("forecast")?.type)).toBe("raster");
});

test("drawn cell edges sit on the true cell boundaries, and the readout agrees on both sides", async ({ page }) => {
  await open(page);
  await page.evaluate(() => {
    const m = (window as unknown as Win).__cwMap;
    if (m.getLayer("forecast-cell-edges")) m.setLayoutProperty("forecast-cell-edges", "visibility", "none");
  });
  const list = boundaries(4);
  expect(list.length).toBeGreaterThan(1);
  const box = (await page.getByTestId("map").boundingBox())!;
  for (const q of list) {
    await page.evaluate(([lon, lat]) => (window as unknown as Win).__cwMap.jumpTo({ center: [lon, lat], zoom: 12 }), [q.lon, q.lat]);
    await page.waitForFunction(() => (window as unknown as Win).__cwMap.loaded());
    await page.waitForTimeout(150);
    const pt = await page.evaluate(([lon, lat]) => (window as unknown as Win).__cwMap.project([lon, lat]), [q.lon, q.lat]);
    const y0 = Math.round(pt.y) - 30;
    const strip = await page.screenshot({ clip: { x: box.x + Math.round(pt.x), y: box.y + y0, width: 1, height: 60 } });
    const rgb = await page.evaluate(async (b64) => {
      const im = new Image();
      im.src = `data:image/png;base64,${b64}`;
      await im.decode();
      const c = document.createElement("canvas");
      c.width = 1;
      c.height = 60;
      const g2 = c.getContext("2d")!;
      g2.drawImage(im, 0, 0);
      const d = g2.getImageData(0, 0, 1, 60).data;
      return Array.from({ length: 60 }, (_, i) => [d[i * 4], d[i * 4 + 1], d[i * 4 + 2]]);
    }, strip.toString("base64"));
    const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const edge = y0 + rgb.findIndex((c) => dist(c, rgb[59]) < dist(c, rgb[0]));
    expect(Math.abs(edge - pt.y)).toBeLessThanOrEqual(1);
    // the readout two pixels either side of the drawn edge reports the cell drawn there
    for (const [dy, code] of [[-3, q.north], [3, q.south]] as const) {
      await page.mouse.move(box.x + pt.x - 6, box.y + edge + dy - 6);
      await page.mouse.move(box.x + pt.x, box.y + edge + dy, { steps: 3 });
      await expect(page.getByTestId("hover-value")).toHaveText(`${Math.round(code * g.scale_factor * 100)}%`);
    }
  }
});
