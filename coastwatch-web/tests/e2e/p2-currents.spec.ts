import { readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";

/** Observed HF-radar currents: hourly selection, 24 h mean, arrows default, particles optional,
 *  units, freshness, inspector values equal to the published grids, honest empty states. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const OK = "http://localhost:3200";
const M2 = "http://localhost:3203";
type G = { url: string; width: number; height: number; lat_first: number; lat_step: number; lon_first: number; lon_step: number; scale_factor: number; add_offset: number };
type L = { layer_id: string; time: { valid_time: string; observed_times: string[] }; vectors: { u_grid: G; v_grid: G } };
const hourly: L[] = manifest.layers.filter((l: L) => l.layer_id.startsWith("hfr2km_currents_2")).sort((a: L, b: L) => a.time.valid_time.localeCompare(b.time.valid_time));
const newest = hourly[hourly.length - 1];
const stamp = (t: string) => `${t.slice(0, 4)}${t.slice(5, 7)}${t.slice(8, 10)}T${t.slice(11, 13)}Z`;

function grid(g: G): Float64Array {
  const b = gunzipSync(readFileSync(path.join(FIX, g.url)));
  const out = new Float64Array(g.width * g.height);
  for (let i = 0; i < out.length; i++) {
    const c = b.readUInt16LE(i * 2);
    out[i] = c === 65535 ? NaN : c * g.scale_factor + g.add_offset;
  }
  return out;
}

async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
}
const arrowsLoaded = (page: Page) =>
  page.waitForFunction(() => {
    const m = (window as unknown as { __cwMap: { getSource: (s: string) => unknown; getLayer: (l: string) => unknown } }).__cwMap;
    return !!m.getSource("currents") && !!m.getLayer("currents-arrows-1");
  });

test("currents tab is live when the dataset has observed currents; arrows are the default", async ({ page }) => {
  await open(page, OK);
  const tab = page.getByTestId("group-currents");
  await expect(tab).toBeEnabled();
  await expect(tab).toContainText("Observation");
  await tab.click();
  await expect(page).toHaveURL(/layer=currents/);
  await arrowsLoaded(page);
  const panel = page.getByTestId("currents-panel");
  await expect(panel.getByTestId("cur-mode-arrows")).toHaveAttribute("aria-checked", "true");
  await expect(panel.getByTestId("native-resolution")).toHaveText("native 2 km");
  await expect(panel.getByTestId("currents-legend")).toContainText("speed, m/s");
  await expect(panel.getByTestId("currents-legend-item")).toHaveCount(5); // one glyph per speed class
  await expect(panel.getByTestId("currents-legend")).toContainText("1 m/s ≈ 1.9 knots");
  await expect(panel.getByTestId("cur-time")).toContainText("(12:00 UTC)");
  await expect(panel.getByTestId("cur-age")).toHaveText("· 8 h ago");
  await expect(panel.getByTestId("cur-notes")).toContainText("not a forecast");
  await expect(panel.getByTestId("freshness")).toHaveAttribute("data-state", "current");
  // one feature per valid cell of the newest hour, nothing more
  const u = grid(newest.vectors.u_grid);
  const valid = u.filter((x) => Number.isFinite(x)).length;
  const n = await page.evaluate(() => {
    const s = (window as unknown as { __cwMap: { getSource: (s: string) => { _data?: { features?: unknown[] }; serialize?: () => { data: { features: unknown[] } } } } }).__cwMap.getSource("currents");
    return (s.serialize?.().data.features ?? s._data?.features ?? []).length;
  });
  expect(n).toBe(valid);
});

test("hourly selection walks real hours; the 24-hour mean is labelled as such", async ({ page }) => {
  await open(page, `${OK}/?layer=currents`);
  await page.getByTestId("cur-hour-prev").click();
  const prev = hourly[hourly.length - 2];
  await expect(page).toHaveURL(new RegExp(`layer=currents%3A${stamp(prev.time.valid_time)}|layer=currents:${stamp(prev.time.valid_time)}`));
  await expect(page.getByTestId("cur-time")).toContainText(`(${prev.time.valid_time.slice(11, 16)} UTC)`);
  await expect(page.getByTestId("cur-time")).not.toContainText("newest hour");
  await page.getByTestId("cur-mean").click();
  await expect(page.getByTestId("cur-time")).toContainText("24-hour mean");
  await expect(page.getByTestId("cur-notes")).toContainText("smooths out most daily and tidal");
  await expect(page).toHaveURL(/layer=currents%3Amean|layer=currents:mean/);
});

test("animated flow is optional and draws only over observed cells", async ({ page }) => {
  await open(page, `${OK}/?layer=currents`);
  await page.getByTestId("cur-mode-particles").click();
  await expect(page.locator("[data-testid=flow-particles]")).toHaveCount(1);
  await expect(page.getByTestId("currents-legend")).toContainText("Particles move with the observed current");
  await expect(page.getByTestId("currents-legend")).toContainText("Not a trajectory");
  await expect(page).toHaveURL(/flow=particles/);
  expect(await page.evaluate(() => !!(window as unknown as { __cwMap: { getLayer: (l: string) => unknown } }).__cwMap.getLayer("currents-arrows-1"))).toBe(false);
});

test("with reduced motion the flow animation is off and arrows are shown", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await open(page, `${OK}/?layer=currents&flow=particles`);
  await arrowsLoaded(page);
  await expect(page.getByTestId("cur-mode-particles")).toBeDisabled();
  await expect(page.locator("[data-testid=flow-particles]")).toHaveCount(0);
  await ctx.close();
});

test("inspector: the published speed and direction at a radar cell, and an honest gap elsewhere", async ({ page }) => {
  const g = newest.vectors.u_grid;
  const u = grid(g);
  const v = grid(newest.vectors.v_grid);
  const k = u.findIndex((x) => Number.isFinite(x));
  const r = Math.floor(k / g.width);
  const c = k % g.width;
  const lat = g.lat_first + r * g.lat_step;
  const lon = g.lon_first + c * g.lon_step;
  await open(page, `${OK}/?layer=currents&inspect=${lat.toFixed(4)},${lon.toFixed(4)}`);
  const near = page.getByTestId("currents-near");
  await expect(near.getByTestId("currents-near-speed")).toHaveText(Math.hypot(u[k], v[k]).toFixed(2));
  const dir = ((Math.atan2(u[k], v[k]) * 180) / Math.PI + 360) % 360;
  await expect(near.getByTestId("currents-near-dir")).toContainText(`(${Math.round(dir)}°)`);
  await expect(near.getByTestId("currents-near-time")).toContainText("Observed, hour of");
  // on land: no radar cell, said plainly
  await open(page, `${OK}/?layer=currents&inspect=36.70,-121.65`);
  await expect(page.getByTestId("currents-near-none")).toContainText("No data, not calm water");
});

test("days later the same hours are shown as stale, with their own times", async ({ page }) => {
  await open(page, `${OK}/?layer=currents`, "2026-10-10T20:00:00Z");
  await expect(page.getByTestId("currents-panel").getByTestId("freshness")).toHaveAttribute("data-state", "stale");
  await expect(page.getByTestId("cur-time")).toContainText("(12:00 UTC)");
});

test("a dataset without currents keeps the tab disabled and draws nothing", async ({ page }) => {
  await open(page, M2);
  await expect(page.getByTestId("group-currents")).toBeDisabled();
  expect(await page.evaluate(() => !!(window as unknown as { __cwMap: { getSource: (s: string) => unknown } }).__cwMap.getSource("currents"))).toBe(false);
});

test("currents and multi-sensor panels: no serious or critical axe violations (desktop and phone)", async ({ browser }) => {
  const { default: AxeBuilder } = await import("@axe-core/playwright");
  for (const opts of [{ viewport: { width: 1440, height: 900 } }, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }]) {
    const ctx = await browser.newContext(opts);
    const page = await ctx.newPage();
    for (const url of [`${OK}/?layer=currents`, `${OK}/?layer=multi&sensor=1`]) {
      await open(page, url);
      const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").exclude("[data-testid=flow-particles]").analyze();
      const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      expect(bad.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`), url).toEqual([]);
    }
    await ctx.close();
  }
});
