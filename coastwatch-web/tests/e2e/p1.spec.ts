import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/** P1 Ocean Map: layer groups, satellite observations, dates, coverage, alignment, layout. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const satReport = JSON.parse(readFileSync(path.join(FIX, "verification/satellite-points.json"), "utf8"));
const OK = "http://localhost:3200";
const FAILED = "http://localhost:3201";
const BASE = "/data/fixture/v1";

type Layer = { layer_id: string; palette?: { stops: { color: string }[] }; tiles?: { url_template: string; sample_tiles: string[] }; composite?: { oldest_observed_date: string; newest_observed_date: string; window_days: number; age_tiles: { url_template: string } } };
const latest: Layer = manifest.layers.find((l: Layer) => l.layer_id === "olci300_chl_latest");
const fmt = (d: string) => new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" }).format(new Date(`${d}T12:00:00Z`));

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}

const mapSource = (page: Page, id: string) =>
  page.evaluate((sid) => {
    const m = (window as unknown as { __cwMap: { getSource: (s: string) => { tiles?: string[]; url?: string } | undefined } }).__cwMap;
    const s = m.getSource(sid);
    return s ? { tiles: s.tiles ?? null, url: s.url ?? null } : null;
  }, id);

test.describe("layer groups", () => {
  test("HAB forecast first; one group drawn at a time (currents only when chosen)", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("group-forecast")).toHaveAttribute("aria-selected", "true");
    // a dataset without currents keeps that tab disabled: tests/e2e/p2-currents.spec.ts
    expect(await mapSource(page, "currents")).toBeNull();
    // the C-HARM raster is opaque and nearest-sampled, over a hatched no-value sea
    const paint = await page.evaluate(() => {
      const m = (window as unknown as { __cwMap: { getPaintProperty: (l: string, p: string) => unknown; getLayer: (l: string) => unknown } }).__cwMap;
      return { opacity: m.getPaintProperty("forecast-raster", "raster-opacity"), resampling: m.getPaintProperty("forecast-raster", "raster-resampling"), hatch: !!m.getLayer("water-nodata") };
    });
    expect(paint).toEqual({ opacity: 1, resampling: "nearest", hatch: true });
  });

  test("legend colours are exactly the published palette classes", async ({ page }) => {
    await open(page, OK);
    const bar = page.getByTestId("probability-legend").locator("div").first();
    const style = (await bar.getAttribute("style")) ?? "";
    const charm = manifest.layers.find((l: Layer) => l.layer_id === "charm_particulate_domoic_lead1");
    for (const s of charm.palette.stops) expect(style.toLowerCase()).toContain(s.color);
    expect(charm.palette.id).toBe("cw-probability-classes-v1");
  });
});

test.describe("satellite observations", () => {
  test("latest clear view: tiles from the dataset, native 300 m, real observation dates and coverage", async ({ page }) => {
    await open(page, OK);
    await page.getByTestId("group-satellite").click();
    const panel = page.getByTestId("satellite-panel");
    await expect(panel.getByTestId("native-resolution")).toHaveText("native 300 m");
    await expect(panel.getByTestId("sat-dates")).toContainText(`Pixels observed ${fmt(latest.composite!.oldest_observed_date)}–${fmt(latest.composite!.newest_observed_date)}`);
    await expect(panel.getByTestId("sat-dates")).toContainText("% of Monterey Bay ocean observed in the last 7 days");
    await expect(panel).toContainText("not toxin");
    await expect(panel).toContainText("not low chlorophyll");
    await page.waitForFunction(() => !!(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("satellite"));
    const src = await mapSource(page, "satellite");
    expect(src!.tiles![0]).toBe(`${BASE}/${latest.tiles!.url_template}`);
    expect(await mapSource(page, "forecast")).toBeNull(); // one product per legend
    const [z, x, y] = latest.tiles!.sample_tiles[0].split("/");
    const r = await page.request.get(`${OK}${BASE}/${latest.tiles!.url_template.replace("{z}", z).replace("{x}", x).replace("{y}", y)}`);
    expect(r.status()).toBe(200);
    expect(r.headers()["content-type"]).toContain("image/png");
    await expect(page).toHaveURL(/layer=olci300/);
  });

  test("each day shows its coverage, and a fully clouded day says so instead of drawing anything", async ({ page }) => {
    await open(page, `${OK}/?layer=olci300`);
    await expect(page.getByTestId("sat-day-2026-10-06")).toBeVisible();
    await page.getByTestId("sat-day-2026-10-07").click();
    await expect(page.getByTestId("sat-dates")).toContainText("No clear observation on Wed, Oct 7");
    await page.waitForFunction(() => !(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("satellite"));
    await page.getByTestId("sat-day-2026-10-06").click();
    await expect(page.getByTestId("sat-dates")).toContainText("Overpass");
    await expect(page).toHaveURL(/layer=olci300%3A2026-10-06|layer=olci300:2026-10-06/);
  });

  test("observation age can be shown per pixel", async ({ page }) => {
    await open(page, `${OK}/?layer=olci300`);
    await page.getByTestId("toggle-age").check();
    await expect(page.getByTestId("age-legend")).toBeVisible();
    const want = `${BASE}/${latest.composite!.age_tiles.url_template}`;
    await page.waitForFunction((t) => (window as unknown as { __cwMap: { getSource: (s: string) => { tiles?: string[] } | undefined } }).__cwMap.getSource("satellite-age")?.tiles?.[0] === t, want);
    const src = await mapSource(page, "satellite-age");
    expect(src!.tiles![0]).toBe(want);
  });

  test("point readout: the published value and its observation date, at the source cell", async ({ page }) => {
    const row = satReport.rows.find((r: { layer_id: string; lat: number; value: number | null; observed_date?: string }) => r.layer_id === "olci300_chl_latest" && r.value != null && r.lat < 37.0 && r.observed_date);
    await open(page, `${OK}/?layer=olci300&inspect=${row.lat},${row.lon}`);
    const near = page.getByTestId("satellite-near");
    const v = row.value >= 10 ? row.value.toFixed(0) : Number(row.value).toPrecision(2);
    await expect(near.getByTestId("satellite-near-value")).toHaveText(v);
    await expect(near.getByTestId("satellite-near-date")).toHaveText(fmt(row.observed_date));
    // with the multi-sensor layer published, the inspector names the sensor shown there;
    // every OLCI date in the fixture is within 2 days of VIIRS's, so Sentinel-3 is shown
    await expect(near.getByTestId("satellite-near-sensor")).toContainText("Sentinel-3 OLCI 300 m");
    await expect(near).not.toContainText("Nearest clear pixel");
  });
});

test.describe("old or failed satellite data never reads as recent", () => {
  const newest = latest.composite!.newest_observed_date;
  const plus = (d: string, n: number) => new Date(Date.parse(`${d}T20:00:00Z`) + n * 86_400_000).toISOString();
  for (const [days, state] of [
    [1, "current"],
    [5, "stale"],
    [12, "historical"],
  ] as const) {
    test(`${days} days after the newest overpass the layer is ${state}`, async ({ page }) => {
      await open(page, `${OK}/?layer=olci300`, plus(newest, days));
      const badge = page.getByTestId("satellite-panel").getByTestId("freshness");
      await expect(badge).toHaveAttribute("data-state", state);
      await expect(badge).toContainText(`observed ${days} day`);
    });
  }

  test("a product the last update did not refresh says so and keeps its own dates", async ({ page }) => {
    const failed = JSON.parse(readFileSync(path.resolve(__dirname, "../fixture-data-failed/v1/manifest.json"), "utf8"));
    const viirs = failed.layers.find((l: Layer) => l.layer_id === "viirs750_chl_latest");
    await open(page, `${FAILED}/?layer=viirs750`, "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("sat-update-failed")).toContainText("did not refresh this product");
    await expect(page.getByTestId("sat-dates")).toContainText(`Pixels observed ${fmt(viirs.composite.oldest_observed_date)}–${fmt(viirs.composite.newest_observed_date)}`);
    // OLCI did update in that run: no failure note there
    await page.getByTestId("sat-olci300").click();
    await expect(page.getByTestId("sat-update-failed")).toHaveCount(0);
  });
});

test.describe("layout", () => {
  test("no inspector until a port is selected; then official notices come first", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("detail-panel")).toHaveCount(0);
    await page.getByTestId("port-row-593").click();
    const panel = page.getByTestId("detail-panel");
    await expect(panel).toBeVisible();
    await expect(page.getByTestId("official-summary")).toHaveCount(0); // the nav card shrinks to a breadcrumb
    await expect(page.getByTestId("nav-crumb")).toContainText("Monterey Bay");
    const first = await panel.getByTestId("port-official").evaluate(
      (o, f) => !!(o.compareDocumentPosition(f as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
      await panel.getByTestId("port-forecast").elementHandle(),
    );
    expect(first).toBe(true);
    await expect(panel.getByTestId("satellite-near")).toBeVisible();
  });

  for (const vp of [
    { width: 1280, height: 720 },
    { width: 1440, height: 900 },
  ]) {
    test(`at ${vp.width}x${vp.height} the dock, navigation and inspector never overlap`, async ({ page }) => {
      await page.setViewportSize(vp);
      await open(page, `${OK}/?region=monterey_bay&port=593`);
      await expect(page.getByTestId("detail-panel")).toBeVisible();
      const [dock, nav, ins] = await Promise.all(["layer-dock", "nav-card", "detail-panel"].map((id) => page.getByTestId(id).boundingBox()));
      const overlap = (a: typeof dock, b: typeof dock) => !!a && !!b && a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
      expect(overlap(dock, ins)).toBe(false);
      expect(overlap(nav, ins)).toBe(false);
      expect(overlap(nav, dock)).toBe(false);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    });
  }
});
