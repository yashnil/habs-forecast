import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** M6 Bloom Intelligence: one station, one timeline. Values are checked against the fixture data. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const obs = JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith("observations-"))!), "utf8"));
const NOW = "2026-10-08T20:00:00Z";
const DAY = 86_400_000;
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const threshold = (v: string) => manifest.layers.find((l: { group_id: string; variable: string; threshold_text?: string }) => l.group_id === "charm" && l.variable === v && l.threshold_text).threshold_text as string;
const SC = obs.stations.find((s: { station_id: string }) => s.station_id === "HABs-SantaCruzWharf");

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
  await expect(page.getByTestId("station-detail")).toBeVisible();
}

function pda(from: number, to = Date.parse(NOW)) {
  const s = SC.series.find((x: { variable: string }) => x.variable === "pDA");
  return SC.sample_times
    .map((t: string, i: number) => ({ t: Date.parse(t), v: s.values[i] as number | null, q: (s.qualifiers ?? {})[String(i)] as string | undefined }))
    .filter((x: { t: number }) => x.t >= from && x.t <= to);
}

test.describe("title block and sampling strip", () => {
  test("the strip has one tick per sampling visit in the last 120 days", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const n = SC.sample_times.filter((t: string) => Date.parse(t) >= Date.parse(NOW) - 120 * DAY).length;
    await expect(page.getByTestId("sampling-strip")).toHaveAttribute("data-visits", String(n));
    await expect(page.getByTestId("freshness")).toContainText("Current");
  });
});

test.describe("primary chart", () => {
  test("reported zeros sit in their own lane and are counted; the highest value in view is labelled with its date", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const rows = pda(Date.parse(NOW) - 365 * DAY);
    const zeros = rows.filter((r: { q?: string }) => r.q === "reported_zero").length;
    const top = rows.filter((r: { v: number | null; q?: string }) => r.v != null && r.v > 0 && r.q !== "reported_zero").sort((a: { v: number }, b: { v: number }) => b.v - a.v)[0];
    const chart = page.getByTestId("primary-chart");
    await expect(chart).toHaveAttribute("data-zeros", String(zeros));
    await expect(page.getByTestId("zero-lane")).toBeVisible();
    expect(await page.locator('[data-testid="chart-measured"] circle[data-q="reported_zero"]').count()).toBe(zeros);
    await expect(page.getByTestId("chart-summary")).toContainText(`${zeros} reported 0`);
    const label = page.getByTestId("highest-in-view");
    const shown = top.v >= 100 ? top.v.toFixed(0) : top.v >= 1 ? top.v.toPrecision(3) : top.v.toPrecision(2);
    await expect(label).toContainText(shown);
    await expect(label).toContainText(new Date(top.t).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "America/Los_Angeles" }));
  });

  test("values below the fixed axis are edge markers with their exact value in the readout, never placed on a gridline", async ({ page }) => {
    // the published record holds five pDA values under 0.0001 ng/mL (e.g. 4.6e-5 on 2016-11-16)
    await open(page, `${OK}/bloom?year=2016`);
    const low = pda(Date.UTC(2016, 0, 1, 8), Date.UTC(2017, 0, 1, 8)).filter((r: { v: number | null; q?: string }) => r.v != null && r.v > 0 && r.v < 1e-4 && r.q !== "reported_zero");
    expect(low.length).toBeGreaterThan(0);
    expect(await page.locator('[data-testid="chart-measured"] [data-q="below-axis"]').count()).toBe(low.length);
    await expect(page.getByTestId("legend-off-axis")).toContainText(`${low.length} value`);
  });

  test("the model band shares the time axis and the crosshair, on its own 0–100 % axis", async ({ page }) => {
    await open(page, `${OK}/bloom?range=3m`);
    await page.getByTestId("model-track").scrollIntoViewIfNeeded();
    const measured = await page.getByTestId("chart-measured").boundingBox();
    const model = await page.getByTestId("model-track").boundingBox();
    expect(Math.abs(measured!.x - model!.x)).toBeLessThan(1);
    expect(Math.abs(measured!.width - model!.width)).toBeLessThan(1);
    await expect(page.getByTestId("model-track")).toContainText("100%");
    // the producer's threshold is quoted verbatim, as on the Ocean Map, for each model quantity
    await expect(page.getByTestId("model-section")).toContainText(threshold("particulate_domoic"));
    await page.getByTestId("model-var-pseudo_nitzschia").click();
    await expect(page.getByTestId("model-section")).toContainText(threshold("pseudo_nitzschia"));
    await page.getByTestId("model-var-particulate_domoic").click();
    await page.mouse.move(model!.x + model!.width * 0.8, model!.y + model!.height / 2);
    await expect(page.getByTestId("chart-tooltip")).toContainText("Model");
  });

  test("the view is in the link: quantity, range and year survive a reload", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await page.getByTestId("quantity-chl_extracted").click();
    await page.getByTestId("range-3y").click();
    await expect(page).toHaveURL(/var=chl_extracted/);
    await expect(page).toHaveURL(/range=3y/);
    await page.reload();
    await expect(page.getByTestId("primary-chart")).toHaveAttribute("data-variable", "chl_extracted");
    await expect(page.getByTestId("range-3y")).toHaveAttribute("aria-checked", "true");
  });
});

test.describe("defaults follow the data", () => {
  test("a station that never measured particulate domoic acid opens on a quantity it measures; the link still wins", async ({ page }) => {
    await open(page, `${OK}/bloom?station=HABs-MorroBayFrontBay`);
    await expect(page.getByTestId("quantity-pDA")).toContainText("Never measured here");
    await expect(page.getByTestId("primary-chart")).toHaveAttribute("data-variable", "pn_seriata");
    await expect(page.getByTestId("primary-chart")).not.toHaveAttribute("data-measured", "0");
    await open(page, `${OK}/bloom?station=HABs-MorroBayFrontBay&var=pDA`);
    await expect(page.getByTestId("primary-chart")).toHaveAttribute("data-variable", "pDA");
    await expect(page.getByTestId("chart-empty")).toContainText("Never measured at this station");
  });
});

test.describe("season heatmap", () => {
  test("choosing a year charts that year; display bins are labelled not risk levels", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const heat = page.getByTestId("season-heatmap");
    await expect(heat).toBeVisible();
    await expect(page.getByTestId("season-section")).toContainText("not risk levels");
    const year = Number(SC.sample_times[0].slice(0, 4)) + 1;
    await page.getByTestId(`heat-year-${year}`).click();
    await expect(page.getByTestId("range-year")).toContainText(String(year));
    await expect(page).toHaveURL(new RegExp(`year=${year}`));
    const n = pda(Date.UTC(year, 0, 1, 8), Date.UTC(year + 1, 0, 1, 8)).length;
    await expect(page.getByTestId("primary-chart")).toHaveAttribute("data-visits", String(n));
    await page.getByTestId("range-year").click();
    await expect(page.getByTestId("range-1y")).toHaveAttribute("aria-checked", "true");
  });

  test("weeks with only reported zeros and weeks not sampled have their own swatches", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    const legend = page.getByTestId("heat-legend");
    await expect(legend).toContainText("reported 0 only");
    await expect(legend).toContainText("not sampled");
    const grid = page.locator('[data-testid="heat-grid"] svg');
    expect(Number(await grid.getAttribute("data-zero-cells"))).toBeGreaterThan(0);
    expect(Number(await grid.getAttribute("data-bin-cells"))).toBeGreaterThan(0);
    // pointing at a cell describes it in words (week, highest value and date, or why there is none)
    await grid.scrollIntoViewIfNeeded();
    const bx = await grid.boundingBox();
    await page.mouse.move(bx!.x + bx!.width * 0.4, bx!.y + bx!.height - 7);
    await expect(page.getByTestId("heat-detail")).toContainText(/Week of .*: (highest|reported 0 only|not sampled)/);
  });
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  test("picker instead of the rail and map; heatmap collapsed with its title visible; no horizontal scroll", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await expect(page.getByTestId("station-map")).toBeHidden();
    await expect(page.getByTestId("station-select")).toBeVisible();
    await expect(page.getByTestId("season-section")).not.toHaveAttribute("open", "");
    await expect(page.getByTestId("season-section")).toContainText("How this season compares");
    await expect(page.getByTestId("model-section")).toBeVisible();
    await expect(page.getByTestId("bloom-official")).toContainText("A missing notice does not mean");
    // a tap on the chart reads the nearest value (no hover on touch screens)
    const svg = page.getByTestId("chart-measured");
    await svg.scrollIntoViewIfNeeded();
    const bx = await svg.boundingBox();
    await page.touchscreen.tap(bx!.x + bx!.width * 0.6, bx!.y + bx!.height * 0.4);
    await expect(page.getByTestId("chart-tooltip")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    const r = await new AxeBuilder({ page }).analyze();
    const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
  });
});
