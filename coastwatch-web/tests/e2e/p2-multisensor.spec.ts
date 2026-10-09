import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/** Multi-sensor satellite view: labelled, one sensor per pixel, members kept, inspectable. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const report = JSON.parse(readFileSync(path.join(FIX, "verification/satellite-points.json"), "utf8"));
const OK = "http://localhost:3200";
const BASE = "/data/fixture/v1";
type Row = { layer_id: string; sensor?: string; lat: number; lon: number; value: number | null };
const ms = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "multisensor_chl_latest");
const rows: Row[] = report.rows.filter((r: Row) => r.layer_id === "multisensor_chl_latest");
const fmtV = (v: number) => (v >= 10 ? v.toFixed(0) : Number(v).toPrecision(2));

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}
const source = (page: Page, id: string) =>
  page.evaluate((sid) => {
    const s = (window as unknown as { __cwMap: { getSource: (s: string) => { tiles?: string[] } | undefined } }).__cwMap.getSource(sid);
    return s?.tiles?.[0] ?? null;
  }, id);

test("multi-sensor view is offered, clearly labelled, and draws its own tiles", async ({ page }) => {
  await open(page, `${OK}/?layer=olci300`);
  await page.getByTestId("sat-multi").click();
  const panel = page.getByTestId("satellite-panel");
  await expect(panel.getByTestId("multi-label")).toContainText("Multi-sensor display");
  await expect(panel.getByTestId("multi-label")).toContainText("nothing is averaged");
  await expect(panel.getByTestId("native-resolution")).toHaveText("native 300 m + 750 m");
  await expect(panel.getByTestId("multi-coverage")).toContainText("observed by either sensor (Sentinel-3 alone 31%)");
  await expect(panel.getByTestId("sat-agreement")).toContainText("No same-day overlap this week");
  await expect(panel.getByTestId("multi-viirs-age")).toContainText("median 5 days old");
  await expect(page).toHaveURL(/layer=multi/);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("satellite"));
  expect(await source(page, "satellite")).toBe(`${BASE}/${ms.tiles.url_template}`);
});

test("which sensor and how old: categorical overlays replace the colours, one at a time", async ({ page }) => {
  await open(page, `${OK}/?layer=multi`);
  await page.getByTestId("toggle-sensor").check();
  await expect(page.getByTestId("sensor-legend")).toContainText("Sentinel-3 OLCI 300 m");
  await expect(page.getByTestId("sensor-legend")).toContainText("VIIRS 750 m");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("satellite-age"));
  expect(await source(page, "satellite-age")).toBe(`${BASE}/${ms.multisensor.sensor_tiles.url_template}`);
  await expect(page).toHaveURL(/sensor=1/);
  await page.getByTestId("toggle-age").check();
  await expect(page.getByTestId("toggle-sensor")).not.toBeChecked();
  await expect(page.getByTestId("age-legend")).toBeVisible();
  await page.waitForFunction((t) => (window as unknown as { __cwMap: { getSource: (s: string) => { tiles?: string[] } | undefined } }).__cwMap.getSource("satellite-age")?.tiles?.[0] === t, `${BASE}/${ms.multisensor.age_tiles.url_template}`);
});

test("the single-sensor layers stay available", async ({ page }) => {
  await open(page, `${OK}/?layer=multi`);
  await page.getByTestId("sat-olci300").click();
  await expect(page.getByTestId("native-resolution")).toHaveText("native 300 m");
  await page.getByTestId("sat-viirs750").click();
  await expect(page.getByTestId("native-resolution")).toHaveText("native 750 m");
});

for (const sensor of ["VIIRS", "Sentinel-3"] as const) {
  test(`inspector names the contributing sensor (${sensor}) with that sensor's own value and date`, async ({ page }) => {
    const r = rows.find((x) => x.sensor === sensor && x.value != null)!;
    await open(page, `${OK}/?layer=multi&inspect=${r.lat},${r.lon}`);
    const near = page.getByTestId("satellite-near");
    await expect(near.getByTestId("satellite-near-sensor")).toContainText(sensor === "VIIRS" ? "VIIRS 750 m" : "Sentinel-3 OLCI 300 m");
    await expect(near.getByTestId("satellite-near-value")).toHaveText(fmtV(r.value!));
    await expect(near.getByTestId("satellite-near-date")).not.toBeEmpty();
  });
}

test("an area neither sensor observed stays unobserved", async ({ page }) => {
  const r = rows.find((x) => x.sensor === "none")!;
  await open(page, `${OK}/?layer=multi&inspect=${r.lat},${r.lon}`);
  await expect(page.getByTestId("satellite-near-none")).toContainText("Neither sensor observed");
});
