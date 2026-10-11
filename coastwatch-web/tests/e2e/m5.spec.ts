import { readFileSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";
import { openPlaces } from "./helpers";

/** M5 Ocean Map: opening layer, map area, controls, inspector, readouts, tiles and motion. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const charmReport = JSON.parse(readFileSync(path.join(FIX, "verification/charm-points.json"), "utf8"));
type Win = { __cwMap: { [k: string]: (...a: unknown[]) => unknown } };

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}
const mapCall = (page: Page, fn: string, ...args: unknown[]) =>
  page.evaluate(([f, a]) => (window as unknown as Win).__cwMap[f as string](...(a as unknown[])), [fn, args] as const);

/** Share of the map not covered by floating controls (element boxes on a 4 px grid). */
async function unobstructed(page: Page) {
  return page.evaluate(() => {
    const m = document.querySelector("[data-testid=map]")!.getBoundingClientRect();
    const sel = ["control-rail", "place-chip", "nav-card", "layer-dock", "mobile-sheet", "place-button", "detail-panel"].map((t) => `[data-testid=${t}]`).concat(".maplibregl-ctrl-bottom-right .maplibregl-ctrl");
    const boxes = sel.flatMap((s) => [...document.querySelectorAll(s)].map((e) => e.getBoundingClientRect()));
    let cov = 0;
    let tot = 0;
    for (let x = m.left; x < m.right; x += 4)
      for (let y = m.top; y < m.bottom; y += 4) {
        tot++;
        if (boxes.some((b) => x >= b.left && x < b.right && y >= b.top && y < b.bottom)) cov++;
      }
    const free = 1 - cov / tot;
    return { ofMap: free, ofViewport: (free * m.width * m.height) / (innerWidth * innerHeight) };
  });
}

test.describe("opening layer", () => {
  test("current C-HARM run: opens on the forecast, and the link does not pin the layer", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("forecast-panel")).toBeVisible();
    await expect(page.getByTestId("group-forecast")).toHaveAttribute("aria-selected", "true");
    await expect(page).not.toHaveURL(/layer=/);
  });

  test("nothing current (stale run, satellite views 6 days old): seafloor map and an explicit message", async ({ page }) => {
    await open(page, OK, "2026-10-12T20:00:00Z");
    const msg = page.getByTestId("nothing-current");
    await expect(msg).toContainText("No current forecast or recent satellite view");
    await expect(msg).toContainText("issued Thu, Oct 8");
    await expect(msg).toContainText("does not mean conditions are normal");
    expect(await mapCall(page, "getSource", "forecast")).toBeFalsy();
    expect(await mapCall(page, "getSource", "satellite")).toBeFalsy();
    await expect(page.getByTestId("group-forecast")).toHaveAttribute("aria-selected", "false");
    await page.getByTestId("show-latest-forecast").click();
    await expect(page.getByTestId("stale-note")).toBeVisible();
    await expect(page).toHaveURL(/layer=forecast/);
  });

  test("a layer named in the link always wins", async ({ page }) => {
    await open(page, `${OK}/?layer=olci300`, "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("satellite-panel")).toBeVisible();
    await expect(page.getByTestId("nothing-current")).toHaveCount(0);
  });
});

test.describe("map area and layout", () => {
  for (const url of ["/", "/?layer=multi", "/?layer=currents"]) {
    test(`1440 x 900 ${url}: at least 75% of the map is unobstructed`, async ({ page }) => {
      await open(page, `${OK}${url}`);
      await page.waitForTimeout(300);
      expect((await unobstructed(page)).ofMap).toBeGreaterThanOrEqual(0.75);
    });
  }

  test("390 x 844: at least 55% of the screen shows the map", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await open(page, OK);
    await expect(page.getByTestId("mobile-sheet")).toHaveAttribute("data-snap", "peek");
    expect((await unobstructed(page)).ofViewport).toBeGreaterThanOrEqual(0.55);
    await ctx.close();
  });

  for (const vp of [
    { width: 1280, height: 720 },
    { width: 1440, height: 900 },
  ]) {
    test(`${vp.width} x ${vp.height}: rail, place chip, place list, dock and inspector never overlap`, async ({ page }) => {
      await page.setViewportSize(vp);
      await open(page, `${OK}/?region=monterey_bay&port=593`);
      await expect(page.getByTestId("detail-panel")).toBeVisible();
      await openPlaces(page);
      const ids = ["control-rail", "place-chip", "nav-card", "layer-dock", "detail-panel"];
      const boxes = await Promise.all(ids.map((id) => page.getByTestId(id).boundingBox()));
      const overlap = (a: (typeof boxes)[0], b: (typeof boxes)[0]) => !!a && !!b && a.x < b.x + b.width - 0.5 && b.x < a.x + a.width - 0.5 && a.y < b.y + b.height - 0.5 && b.y < a.y + a.height - 0.5;
      for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) expect(overlap(boxes[i], boxes[j]), `${ids[i]} / ${ids[j]}`).toBe(false);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    });
  }

  test("official notices stay one click away in every state", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&port=593`);
    await expect(page.getByTestId("place-chip")).toContainText("Not verified");
    await expect(page.getByTestId("registry-disclosure")).toBeVisible();
    await page.getByTestId("official-summary").click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
  });

  test("Escape closes the place list, then the inspector", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&port=593`);
    await openPlaces(page);
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("nav-card")).toHaveCount(0);
    await expect(page.getByTestId("detail-panel")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("detail-panel")).toHaveCount(0);
  });

  test("the layer rail is a keyboard tab list", async ({ page }) => {
    await open(page, OK);
    await page.getByTestId("group-forecast").focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.getByTestId("group-satellite")).toBeFocused();
    await expect(page.getByTestId("group-satellite")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("satellite-panel")).toBeVisible();
  });
});

test.describe("phone sheet", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  test("snap points from the keyboard; the peek keeps the legend and what is shown", async ({ page }) => {
    await open(page, OK);
    const sheet = page.getByTestId("mobile-sheet");
    await expect(sheet.getByTestId("map-stamp")).toBeInViewport();
    await expect(sheet.getByTestId("probability-legend")).toBeInViewport();
    const handle = page.getByTestId("sheet-handle");
    await handle.focus();
    await page.keyboard.press("ArrowUp");
    await expect(sheet).toHaveAttribute("data-snap", "half");
    await page.keyboard.press("End");
    await expect(sheet).toHaveAttribute("data-snap", "full");
    await page.keyboard.press("Home");
    await expect(sheet).toHaveAttribute("data-snap", "peek");
    await expect(handle).toHaveAttribute("aria-expanded", "false");
  });

  test("reduced motion: the sheet does not animate", async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, reducedMotion: "reduce" });
    const page = await ctx.newPage();
    await open(page, OK);
    const d = await page.getByTestId("mobile-sheet").evaluate((e) => getComputedStyle(e).transitionDuration);
    expect(d).toMatch(/^0s/);
    await ctx.close();
  });
});

test.describe("rasters and motion", () => {
  test("time steps cut, never cross-fade; switching layer fades in", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast&lead=1`);
    await page.getByTestId("lead-2").click();
    await page.waitForFunction(() => (window as unknown as Win).__cwMap.getLayer("forecast-raster"));
    await page.waitForTimeout(800);
    await page.getByTestId("lead-3").click();
    await page.waitForTimeout(50);
    expect(await mapCall(page, "getPaintProperty", "forecast-raster", "raster-fade-duration")).toBe(0);
    await page.getByTestId("group-satellite").click();
    await page.waitForFunction(() => (window as unknown as Win).__cwMap.getLayer("satellite-raster"));
    expect(await mapCall(page, "getPaintProperty", "satellite-raster", "raster-fade-duration")).toBe(220);
  });

  test("reduced motion: layer switches cut too", async ({ browser }) => {
    const ctx = await browser.newContext({ reducedMotion: "reduce" });
    const page = await ctx.newPage();
    await open(page, `${OK}/?layer=forecast`);
    await page.getByTestId("group-satellite").click();
    await page.waitForFunction(() => (window as unknown as Win).__cwMap.getLayer("satellite-raster"));
    expect(await mapCall(page, "getPaintProperty", "satellite-raster", "raster-fade-duration")).toBe(0);
    await ctx.close();
  });

  test("C-HARM cells are outlined faintly from z8.5; the outline layer leaves with the forecast", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast`);
    await page.waitForFunction(() => (window as unknown as Win).__cwMap.getLayer("forecast-cell-edges"));
    const minzoom = await page.evaluate(() => (window as unknown as { __cwMap: { getLayer: (l: string) => { minzoom: number } } }).__cwMap.getLayer("forecast-cell-edges").minzoom);
    expect(minzoom).toBe(8.5);
    await page.getByTestId("group-currents").click();
    await page.waitForFunction(() => !(window as unknown as Win).__cwMap.getLayer("forecast-cell-edges"));
  });

  test("no-data dots show only under a data raster", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast`);
    const vis = () => mapCall(page, "getLayoutProperty", "water-nodata", "visibility");
    await expect.poll(vis).toBe("visible");
    await page.getByTestId("group-currents").click();
    await expect.poll(vis).toBe("none");
  });

  test("satellite tiles: 512 px images, no request for a tile that does not exist", async ({ page }) => {
    const missing: string[] = [];
    let served = 0;
    page.on("response", (r) => {
      if (!/\/(tiles|age-tiles|sensor-tiles)\/\d+\/\d+\/\d+\.png$/.test(r.url())) return;
      if (r.status() === 404) missing.push(r.url());
      else if (r.ok()) served++;
    });
    await open(page, `${OK}/?layer=multi`);
    for (const [lon, lat, z] of [[-122.0, 36.8, 10], [-122.4, 37.6, 11], [-123.5, 36.0, 9]] as const) {
      await mapCall(page, "jumpTo", { center: [lon, lat], zoom: z });
      await page.waitForTimeout(700);
    }
    expect(missing).toEqual([]);
    expect(served).toBeGreaterThan(0);
    const ms = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "multisensor_chl_latest");
    expect(ms.tiles.tile_size).toBe(512);
    expect(ms.tiles.max_native_zoom).toBe(11);
  });
});

test.describe("readouts and inspector", () => {
  test("hover shows the exact published value of the C-HARM cell under the pointer", async ({ page }) => {
    const row = charmReport.rows.find((r: { status: string; layer_id: string; point: string }) => r.status === "pass" && r.layer_id === "charm_particulate_domoic_lead1" && /Monterey/.test(r.point));
    await open(page, `${OK}/?layer=forecast&var=particulate_domoic&lead=1`);
    await mapCall(page, "jumpTo", { center: [row.cell_lon, row.cell_lat], zoom: 9.5 });
    await page.waitForTimeout(300);
    const box = (await page.getByTestId("map").boundingBox())!;
    const pt = (await mapCall(page, "project", [row.cell_lon, row.cell_lat])) as { x: number; y: number };
    await page.mouse.move(box.x + pt.x, box.y + pt.y);
    await expect(page.getByTestId("hover-value")).toHaveText(`${Math.round(row.grid_value * 100)}%`);
    await expect(page.getByTestId("hover-readout")).toContainText("valid");
    await page.mouse.move(box.x + 30, box.y + 400); // over the rail: the readout goes
    await expect(page.getByTestId("hover-readout")).toHaveCount(0);
  });

  test("point inspector: four days at the exact cell, and the strip sets the map day", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast&lead=1&inspect=36.80,-121.95`);
    const strip = page.getByTestId("inspect-strip");
    for (const l of [0, 1, 2, 3]) await expect(strip.getByTestId(`inspect-lead-${l}`)).toBeVisible();
    await expect(strip.getByTestId("inspect-lead-1")).toHaveAttribute("aria-checked", "true");
    await strip.getByTestId("inspect-lead-2").click();
    await expect(page).toHaveURL(/lead=2/);
    await expect(page.getByTestId("lead-2")).toHaveAttribute("aria-checked", "true");
  });

  test("port inspector: notices, then the forecast days, then what was measured nearby", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&port=550`);
    const panel = page.getByTestId("port-panel");
    await expect(panel.getByTestId("port-lead-1")).toBeVisible();
    const measured = panel.getByTestId("measured-nearby");
    await expect(measured.getByTestId("measured-HABs-MontereyWharf")).toContainText("Monterey Wharf");
    await expect(measured.getByTestId("measured-HABs-MontereyWharf")).toContainText("sampled");
    await expect(measured.locator("a[href='/bloom?station=HABs-MontereyWharf']")).toBeVisible();
    await expect(measured).toContainText("not today and not at this point");
    // history is summarised (sparkline) and opens on demand
    await expect(panel.getByTestId("port-history").getByTestId("sparkline")).toBeVisible();
  });

  test("map with place list and inspector open: no serious or critical axe violations", async ({ page }) => {
    await open(page, `${OK}/?region=monterey_bay&port=593`);
    await openPlaces(page);
    await page.waitForLoadState("networkidle");
    const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
    const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
  });
});
