import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { OK } from "./ports";

/** The forecast stays off land when the basemap's tiles fail or are slow: data tiles are clipped to
 *  the bundled water (lib/waterFallback), and the bundled sea, coastline and no-data dots are
 *  drawn while the basemap is unavailable. Truth comes from a normal session's basemap.
 *  Fixture data; cells chosen from the fixture grids. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const OMT = /tiles\.openfreemap\.org/;
const OMT_TILES = /tiles\.openfreemap\.org\/.*\.pbf/;
// each test renders two sessions at zoom 12 (software WebGL in CI, two workers): allow for it
test.describe.configure({ timeout: 120_000 });

type Grid = { url: string; width: number; height: number; nodata: number; lat_first: number; lon_first: number; lat_step: number; lon_step: number };
type MapLike = {
  jumpTo(o: unknown): void;
  project(p: number[]): { x: number; y: number };
  loaded(): boolean;
  isSourceLoaded(id: string): boolean;
  queryRenderedFeatures(p: number[], o: unknown): { properties: { class?: string } }[];
  setLayoutProperty(l: string, p: string, v: string): void;
  getLayoutProperty(l: string, p: string): string | undefined;
  getLayer(l: string): unknown;
  getSource(id: string): { serialize(): { data?: { features?: { properties: Record<string, unknown>; geometry: { coordinates: number[] } }[] } } } | undefined;
  getCanvas(): HTMLCanvasElement;
  once(e: string, f: () => void): void;
  triggerRepaint(): void;
};
type Pt = { lon: number; lat: number; kind: "land" | "ocean" | "other-water" };

function grid(layerId: string) {
  const g: Grid = manifest.layers.find((l: { layer_id: string }) => l.layer_id === layerId).grid;
  const codes = new Uint16Array(new Uint8Array(gunzipSync(readFileSync(path.join(FIX, g.url)))).buffer);
  const ok = (r: number, c: number) => r >= 0 && r < g.height && c >= 0 && c < g.width && codes[r * g.width + c] !== g.nodata;
  const lon = (c: number) => { const v = g.lon_first + c * g.lon_step; return v > 180 ? v - 360 : v; };
  return { g, ok, lat: (r: number) => g.lat_first + r * g.lat_step, lon };
}

/** Cells with a value next to a cell without one: the coastal edge of the model. */
function coastalCells(layerId: string): [number, number][] {
  const { g, ok } = grid(layerId);
  const out: [number, number][] = [];
  for (let r = 1; r < g.height - 1; r++) for (let c = 1; c < g.width - 1; c++) if (ok(r, c) && (!ok(r + 1, c) || !ok(r - 1, c) || !ok(r, c + 1) || !ok(r, c - 1))) out.push([r, c]);
  return out;
}

async function open(page: Page, url: string, untilLoaded = true) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(url);
  await page.waitForFunction((full) => {
    const m = (window as unknown as { __cwMap?: MapLike }).__cwMap;
    return !!m && (!full || m.loaded());
  }, untilLoaded, { timeout: 30_000 });
  await page.evaluate(() => {
    const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
    for (const id of ["forecast-cell-edges", "coastline", "coastline-fallback"]) if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", "none");
  });
}

/** A 7 x 7 lattice inside the cell, classified on the basemap; points within 2 px of its shore are skipped. */
async function classify(page: Page, lat: number, lon: number, step: number): Promise<Pt[]> {
  return page.evaluate(
    async ([lat, lon, step]) => {
      const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
      m.jumpTo({ center: [lon, lat], zoom: 12 });
      await new Promise<void>((res) => { m.once("idle", () => res()); m.triggerRepaint(); });
      const pts: Pt[] = [];
      for (let i = 1; i <= 7; i++)
        for (let j = 1; j <= 7; j++) {
          const ll = [lon - step / 2 + (step * j) / 8, lat - step / 2 + (step * i) / 8];
          const p = m.project(ll);
          const at = (dx: number, dy: number) => m.queryRenderedFeatures([p.x + dx, p.y + dy], { layers: ["water"] });
          const around = [at(0, 0), at(2, 0), at(-2, 0), at(0, 2), at(0, -2)];
          if (around.every((f) => f.length) || around.every((f) => !f.length))
            pts.push({ lon: ll[0], lat: ll[1], kind: !around[0].length ? "land" : around.every((f) => f.some((x) => x.properties.class === "ocean")) ? "ocean" : "other-water" });
        }
      return pts;
    },
    [lat, lon, step] as const,
  );
}

/** Whether the forecast colours each point: the canvas with the forecast on vs off. Waits for the
 *  forecast's own tiles only (not the basemap's), so it works while the basemap is slow. At harbour
 *  zoom the map also writes a note at an unmodelled centre (the gap's distance to the nearest cell),
 *  asynchronously after the move: with `unmodelledCentre` the comparison waits for that note to be
 *  at the centre first, so it is in both frames and never counts as the forecast's colour. */
async function drawn(page: Page, lat: number, lon: number, pts: Pt[], unmodelledCentre = false): Promise<boolean[]> {
  return page.evaluate(
    async ([lat, lon, pts, unmodelledCentre]) => {
      const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
      const frame = () => new Promise<void>((res) => { m.once("render", () => res()); m.triggerRepaint(); });
      m.jumpTo({ center: [lon, lat], zoom: 12 });
      // the first render after the jump requests the view's tiles; until then the source still
      // reports the previous view's tiles as loaded
      await frame();
      await frame();
      for (let i = 0; i < 600 && !m.isSourceLoaded("forecast"); i++) await frame();
      const noteHere = () => {
        const f = m.getSource("forecast-gap")?.serialize().data?.features?.find((x) => x.properties.kind === "note");
        return !!f && Math.abs(f.geometry.coordinates[0] - lon) < 1e-4 && Math.abs(f.geometry.coordinates[1] - lat) < 1e-4;
      };
      for (let i = 0; unmodelledCentre && i < 600 && !noteHere(); i++) await frame();
      await frame();
      await frame();
      const grab = () => {
        const c = m.getCanvas();
        const k = document.createElement("canvas");
        k.width = c.width;
        k.height = c.height;
        const g2 = k.getContext("2d")!;
        g2.drawImage(c, 0, 0);
        return { d: g2.getImageData(0, 0, k.width, k.height).data, w: k.width, s: c.width / c.clientWidth };
      };
      await frame();
      const on = grab();
      m.setLayoutProperty("forecast-raster", "visibility", "none");
      await frame();
      await frame();
      const off = grab();
      m.setLayoutProperty("forecast-raster", "visibility", "visible");
      await frame();
      return pts.map((p) => {
        const q = m.project([p.lon, p.lat]);
        const i = (Math.round(q.y * on.s) * on.w + Math.round(q.x * on.s)) * 4;
        return Math.abs(on.d[i] - off.d[i]) + Math.abs(on.d[i + 1] - off.d[i + 1]) + Math.abs(on.d[i + 2] - off.d[i + 2]) > 12;
      });
    },
    [lat, lon, pts, unmodelledCentre] as const,
  );
}

/** Up to two straddling cells and their classified points, from a normal session. */
async function truth(browser: Browser, url: string, layerId: string) {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
  await open(page, url);
  const { g, lat, lon } = grid(layerId);
  const cells: { lat: number; lon: number; pts: Pt[] }[] = [];
  for (const [r, c] of coastalCells(layerId).filter((_, i) => i % 3 === 0).slice(0, 40)) {
    const pts = await classify(page, lat(r), lon(c), g.lat_step);
    if (pts.some((p) => p.kind === "land") && pts.some((p) => p.kind === "ocean")) cells.push({ lat: lat(r), lon: lon(c), pts });
    if (cells.length >= 2) break;
  }
  await page.context().close();
  return cells;
}

async function expectOffLand(page: Page, cells: Awaited<ReturnType<typeof truth>>) {
  let land = 0, ocean = 0;
  for (const cell of cells) {
    const d = await drawn(page, cell.lat, cell.lon, cell.pts);
    const at = (k: Pt["kind"]) => cell.pts.map((p, i) => [p, d[i]] as const).filter(([p]) => p.kind === k);
    land += at("land").length;
    ocean += at("ocean").length;
    expect(at("land").filter(([, x]) => x).map(([p]) => [p.lon, p.lat]), `forecast drawn on land near ${cell.lon}, ${cell.lat}`).toEqual([]);
    expect(at("ocean").filter(([, x]) => !x).map(([p]) => [p.lon, p.lat]), `forecast missing on the ocean near ${cell.lon}, ${cell.lat}`).toEqual([]);
  }
  expect(cells.length).toBeGreaterThan(0);
  expect(land).toBeGreaterThan(0);
  expect(ocean).toBeGreaterThan(0);
}

const PN = "charm_pseudo_nitzschia_lead1";
const PN_URL = `${OK}/?layer=forecast&var=pseudo_nitzschia&lead=1`;

test("with the basemap's tile server unreachable, straddling cells are still drawn on the ocean only", async ({ browser }) => {
  const cells = await truth(browser, PN_URL, PN);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.route(OMT, (r) => r.abort("connectionrefused"));
  const page = await ctx.newPage();
  await open(page, PN_URL);
  // the bundled sea and coastline replace the basemap's
  await expect.poll(() => page.evaluate(() => (window as unknown as { __cwMap: MapLike }).__cwMap.getLayoutProperty("water-fallback", "visibility"))).toBe("visible");
  await expectOffLand(page, cells);
  await ctx.close();
});

test("with the basemap's tiles 45 s late, the forecast is drawn before they arrive, never on land", async ({ browser }) => {
  test.setTimeout(180_000);
  const cells = await truth(browser, PN_URL, PN);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  let arrived = false;
  await ctx.route(OMT_TILES, async (r) => {
    await new Promise((s) => setTimeout(s, 45_000));
    arrived = true;
    await r.continue().catch(() => {});
  });
  const page = await ctx.newPage();
  const t0 = Date.now();
  await open(page, PN_URL, false);
  const fallbackSea = () => page.evaluate(() => (window as unknown as { __cwMap: MapLike }).__cwMap.getLayoutProperty("water-fallback", "visibility"));
  await expectOffLand(page, cells);
  // meanwhile the bundled sea, coastline and no-data dots stand in for the basemap's
  expect(await fallbackSea()).toBe("visible");
  expect(arrived, `basemap tiles arrived before the check (${Date.now() - t0} ms)`).toBe(false);
  // and give way to the basemap once its tiles arrive
  await expect.poll(fallbackSea, { timeout: 90_000 }).toBe("none");
  expect(arrived).toBe(true);
  await ctx.close();
});

test("with the basemap unavailable, no-value water and the toxin mask keep their outside-model tint and readout; land has neither", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.route(OMT, (r) => r.abort("connectionrefused"));
  const page = await ctx.newPage();
  await open(page, `${OK}/?layer=forecast&var=particulate_domoic&lead=0`);
  const { g, ok, lat, lon } = grid("charm_particulate_domoic_lead0");
  const pn = grid("charm_pseudo_nitzschia_lead0");
  // cell centres: a toxin-mask cell (Pseudo-nitzschia value, no toxin value), cells C-HARM does not
  // model next to valued cells, and valued cells
  const mask: number[][] = [], gaps: number[][] = [], valued: number[][] = [];
  for (let r = 1; r < g.height - 1; r++)
    for (let c = 1; c < g.width - 1; c++) {
      const p = [lon(c), lat(r)];
      if (!ok(r, c) && pn.ok(r, c)) mask.push(p);
      else if (!ok(r, c) && !pn.ok(r, c) && (ok(r + 1, c) || ok(r - 1, c) || ok(r, c + 1) || ok(r, c - 1))) gaps.push(p);
      else if (ok(r, c) && ok(r + 1, c) && ok(r - 1, c) && ok(r, c + 1) && ok(r, c - 1)) valued.push(p);
    }
  const probe = (pts: number[][]) =>
    page.evaluate(async (pts) => {
      const m = (window as unknown as { __cwMap: MapLike }).__cwMap;
      const out: { sea: boolean; dots: boolean }[] = [];
      for (const [x, y] of pts) {
        m.jumpTo({ center: [x, y], zoom: 11 });
        await new Promise<void>((res) => { m.once("idle", () => res()); m.triggerRepaint(); });
        const q = m.project([x, y]);
        out.push({
          sea: m.queryRenderedFeatures([q.x, q.y], { layers: ["water-fallback"] }).length > 0,
          dots: m.queryRenderedFeatures([q.x, q.y], { layers: ["water-nodata-fallback"] }).length > 0,
        });
      }
      return out;
    }, pts);
  const pick = (pts: number[][], n: number) => pts.filter((_, i) => i % Math.max(1, Math.floor(pts.length / n)) === 0).slice(0, n);
  // toxin mask and unmodelled water: on the bundled sea, tinted "outside NOAA's model near shore"
  // (v5: the tint is their mark, so the no-data dots are not drawn under the forecast)
  let maskOnSea: number[] | undefined;
  for (const [name, set] of [["toxin mask", pick(mask, 6)], ["not modelled", pick(gaps, 8)]] as const) {
    const res = await probe(set as number[][]);
    const sea = (set as number[][]).filter((_, i) => res[i].sea);
    expect(sea.length, `${name}: cells on the sea`).toBeGreaterThan(0);
    expect(res.filter((r) => r.sea && r.dots).length, `${name}: sea cells hatched under the tint`).toBe(0);
    if (name === "toxin mask") maskOnSea = sea[0];
    // v5: unmodelled water is tinted "outside NOAA's model near shore" (one neutral colour, never a
    // probability; its readout says so), so the coast is not left blank
    for (const [x, y] of sea.slice(0, 2)) {
      // a few pixels around the centre
      const around = [[0, 0], [0.0011, 0.0007], [-0.0009, 0.0012], [0.0006, -0.0013]].map(([dx, dy]) => ({ lon: x + dx, lat: y + dy, kind: "ocean" as const }));
      const d = await drawn(page, y, x, around, true);
      expect(d.some(Boolean), `${name}: outside-model tint at ${x}, ${y}`).toBe(true);
    }
  }
  // land has neither the sea nor the dots
  const land = await probe([[-121.65, 36.68], [-122.27, 37.8]]);
  expect(land).toEqual([{ sea: false, dots: false }, { sea: false, dots: false }]);
  // valued offshore cells are coloured
  for (const [x, y] of pick(valued.filter(([x]) => x < -122.3), 2)) expect((await drawn(page, y, x, [{ lon: x, lat: y, kind: "ocean" }]))[0]).toBe(true);
  // the readout still says why a toxin-mask cell on the sea is empty
  await probe([maskOnSea!]);
  const box = (await page.getByTestId("map").boundingBox())!;
  const q = await page.evaluate(([x, y]) => (window as unknown as { __cwMap: MapLike }).__cwMap.project([x, y]), maskOnSea!);
  await page.mouse.move(box.x + q.x - 6, box.y + q.y - 6);
  await page.mouse.move(box.x + q.x, box.y + q.y, { steps: 3 });
  await expect(page.getByTestId("hover-readout")).toContainText("within about 3–6 km of shore");
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
  await ctx.close();
});
