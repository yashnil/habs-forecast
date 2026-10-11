import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { FAILED, NODATA, OK } from "./ports";
import { openDockDetails } from "./helpers";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const report = JSON.parse(readFileSync(path.join(FIX, "verification/charm-points.json"), "utf8"));


async function open(page: Page, url: string, now = "2026-10-08T20:00:00Z") {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("freshness").first()).not.toHaveText(/Checking/);
}

test.describe("information hierarchy and labelling", () => {
  test("official status comes before the forecast and links to official sources", async ({ page }) => {
    await open(page, OK);
    const summary = page.getByTestId("official-summary");
    const forecast = page.getByTestId("forecast-panel");
    await expect(summary).toBeVisible();
    const before = await summary.evaluate(
      (o, f) => !!(o.compareDocumentPosition(f as Node) & Node.DOCUMENT_POSITION_FOLLOWING),
      await forecast.elementHandle(),
    );
    expect(before).toBe(true);
    await summary.click();
    const official = page.getByTestId("official-drawer");
    await expect(official).toContainText("does not mean an area is open");
    const hrefs = await official.locator("a[href^='https']").evaluateAll((as) => as.map((a) => (a as HTMLAnchorElement).href));
    expect(hrefs).toContain("https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
    expect(hrefs).toContain("https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx");
    await expect(official.locator("a[href^='tel:']")).toHaveCount(2);
  });

  test("fixture data is labelled as test data", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("fixture-banner")).toContainText("Test data");
  });

  test("forecast is labelled as an official forecast with threshold, units and inferred issue date", async ({ page }) => {
    await open(page, OK);
    const panel = page.getByTestId("forecast-panel");
    await expect(panel.getByTestId("product-class")).toHaveText("Agency forecast");
    await expect(panel.getByTestId("freshness").first()).toContainText("issued today");
    // the essential qualifiers are always visible; every published caveat is one click away
    await expect(panel).toContainText(/not a toxin measurement or a closure decision/i);
    await expect(panel).toContainText("does not mean an area is safe");
    await openDockDetails(page);
    await panel.getByText("About this forecast").click();
    const layer = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_particulate_domoic_lead1");
    for (const c of layer.caveats) await expect(panel.getByTestId("forecast-caveats")).toContainText(c);
    await expect(panel.getByTestId("run-line")).toContainText("Issued Thu, Oct 8, 2026 (inferred)");
    await expect(panel.getByTestId("run-line")).toContainText("showing Thu, Oct 8, forecast +1 day");
    await expect(panel.getByTestId("run-line")).toContainText("no forecast exists beyond day 3");
    await expect(panel).toContainText("particulate domoic acid exceeds 500 ng per litre");
    // the colour key sits with the layer it explains, at the layer's native resolution
    await expect(panel.getByTestId("probability-legend")).toContainText("100%");
    await expect(panel.getByTestId("probability-legend")).toContainText("not risk levels");
    await expect(panel.getByTestId("native-resolution")).toHaveText("native 3 km");
  });

  test("lead buttons show real valid dates and select the matching layer", async ({ page }) => {
    await open(page, OK);
    await expect(page.getByTestId("lead-0")).toContainText("Oct 7");
    await expect(page.getByTestId("lead-0")).toContainText("nowcast");
    await expect(page.getByTestId("lead-1")).toContainText("today");
    await expect(page.getByTestId("lead-3")).toContainText("Oct 10");
    await expect(page.getByTestId("lead-4")).toHaveCount(0); // C-HARM issues nowcast + 3 days only
    await page.getByTestId("lead-3").click();
    await expect(page.getByTestId("valid-line")).toContainText("Sat, Oct 10, 2026");
    await expect(page).toHaveURL(/lead=3/);
  });

  test("primary navigation links the three built experiences and nothing unbuilt", async ({ page }) => {
    await open(page, OK);
    const nav = page.getByRole("navigation", { name: "Primary" }).first();
    await expect(page.getByTestId("nav-map")).toHaveAttribute("aria-current", "page");
    await expect(page.getByTestId("nav-bloom")).toHaveAttribute("href", "/bloom");
    await expect(page.getByTestId("nav-fisheries")).toHaveAttribute("href", "/fisheries");
    await expect(nav).not.toContainText(/My Coast|Upcoming/i);
  });
});

test.describe("freshness", () => {
  test("same day: current", async ({ page }) => {
    await open(page, OK, "2026-10-08T20:00:00Z");
    await expect(page.getByTestId("forecast-panel").getByTestId("freshness").first()).toHaveAttribute("data-state", "current");
  });

  // the layer is named in the link: the opening policy (below) does not apply
  test("four days later with no new run: stale, dates unchanged", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast`, "2026-10-12T20:00:00Z");
    const panel = page.getByTestId("forecast-panel");
    await expect(panel.getByTestId("freshness").first()).toHaveAttribute("data-state", "stale");
    await expect(panel.getByTestId("stale-note")).toBeVisible();
    await openDockDetails(page);
    await expect(panel.getByTestId("run-line")).toContainText("Oct 8, 2026");
    await expect(page.getByTestId("lead-1")).toContainText("4 days ago");
  });

  test("two weeks later: historical, explicitly not current", async ({ page }) => {
    await open(page, `${OK}/?layer=forecast`, "2026-10-22T20:00:00Z");
    const panel = page.getByTestId("forecast-panel");
    await expect(panel.getByTestId("freshness").first()).toHaveAttribute("data-state", "historical");
    await expect(panel.getByTestId("historical-note")).toContainText("does not describe current conditions");
  });

  test("a failed update keeps the last good run with its real dates and says so", async ({ page }) => {
    await open(page, `${FAILED}/?layer=forecast`, "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("source-failure-banner")).toContainText("Latest update failed");
    const panel = page.getByTestId("forecast-panel");
    await expect(panel.getByTestId("update-failed")).toBeVisible();
    await openDockDetails(page);
    await expect(panel.getByTestId("run-line")).toContainText("Issued Thu, Oct 8, 2026");
    await expect(page.getByTestId("source-health")).toHaveAttribute("data-state", /stale|historical/);
  });

  test("no data at all: explicit unavailable state, official sources still shown", async ({ page }) => {
    await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
    await page.goto(NODATA);
    await expect(page.getByTestId("data-unavailable")).toContainText("Live data unavailable");
    await expect(page.getByTestId("official-status")).toBeVisible();
    await expect(page.getByTestId("forecast-panel")).toHaveCount(0);
  });
});

test.describe("values and georeferencing", () => {
  test("inspector values equal the source-verified grid values", async ({ page }) => {
    const rows = report.rows.filter((r: { point: string; layer_id: string; status: string }) => r.point === "Monterey Bay (mid-bay)" && r.layer_id.endsWith("lead1"));
    expect(rows).toHaveLength(3);
    await open(page, `${OK}/?var=particulate_domoic&lead=1&inspect=36.80,-121.95`);
    const insp = page.getByTestId("inspector").first();
    await expect(insp).toContainText("valid Thu, Oct 8, 2026");
    for (const r of rows) {
      const variable = r.layer_id.replace(/^charm_/, "").replace(/_lead\d$/, "");
      await expect(insp.getByTestId(`inspect-value-${variable}`)).toHaveText(`${(r.grid_value * 100).toFixed(0)}%`);
    }
    await expect(insp).toContainText("not a closure decision");
  });

  test("nearshore point without a toxin value shows the nearest cell, labelled with its distance", async ({ page }) => {
    await open(page, `${OK}/?lead=1&inspect=36.604,-121.889`);
    const v = page.getByTestId("inspector").first().getByTestId("inspect-value-particulate_domoic");
    await expect(v).toHaveAttribute("data-nearest", "true");
    await expect(v).toHaveText(/\d+% at \d+\.\d km/);
  });

  test("forecast raster is placed at the manifest's Mercator corners and follows the selection", async ({ page }) => {
    await open(page, `${OK}/?var=cellular_domoic&lead=2`);
    await page.waitForFunction(() => {
      const m = (window as unknown as { __cwMap?: { getSource: (id: string) => unknown } }).__cwMap;
      return !!m && !!m.getSource("forecast");
    });
    const src = await page.evaluate(() => {
      const m = (window as unknown as { __cwMap: { getSource: (id: string) => { coordinates: number[][]; url: string } } }).__cwMap;
      const s = m.getSource("forecast");
      return { coordinates: s.coordinates, url: s.url };
    });
    const layer = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "charm_cellular_domoic_lead2");
    expect(src.coordinates).toEqual(layer.image.corners_lnglat);
    expect(src.url).toBe(`/data/fixture/v1/${layer.image.url}`);
  });

  test("selecting satellite chlorophyll replaces the forecast raster (one product per legend)", async ({ page }) => {
    await open(page, OK);
    await page.getByTestId("group-satellite").click();
    await page.getByTestId("sat-imagery").click();
    await page.getByRole("radio", { name: /Chlorophyll · VIIRS/ }).click();
    await page.waitForFunction(() => {
      const m = (window as unknown as { __cwMap?: { getSource: (id: string) => unknown } }).__cwMap;
      return !!m && !!m.getSource("observation") && !m.getSource("forecast");
    });
    const legend = page.getByTestId("chl-legend");
    const viirs = manifest.layers.find((l: { layer_id: string }) => l.layer_id === "gibs_viirs_noaa20_chl");
    await expect(legend.locator("img")).toHaveAttribute("src", viirs.tiles.legend_url);
    await expect(page.getByTestId("observation-panel")).toContainText("does not measure toxins");
    await expect(page.getByTestId("observation-panel")).toContainText("Oct 6");
  });
});

test("sources page lists every source with status and provenance", async ({ page }) => {
  await open(page, `${OK}/sources`);
  for (const id of ["charm", "gibs_chl", "cdfw_ports"]) await expect(page.getByTestId(`source-${id}`)).toBeVisible();
  await expect(page.getByTestId("source-charm")).toContainText("Run issued Thu, Oct 8, 2026 (inferred)");
});
