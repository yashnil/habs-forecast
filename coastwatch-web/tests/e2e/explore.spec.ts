import { expect, test, type Page } from "@playwright/test";
import { openEvidence, openLayers } from "./helpers";
import { OK } from "./ports";

/**
 * Explore (v4): the map-first workspace. One layer at a time with its own legend and provenance; a
 * timebar only for layers with time, stepping through published frames only; one selection, shared
 * by the map, the inspection surface, its charts and the URL; Back returns to the previous view.
 * Fixture run issued 2026-10-08 (leads valid Oct 7 to Oct 10); the clock is Oct 8 in California.
 */
const NOW = "2026-10-08T20:00:00Z";
type Win = { __cwMap: { loaded(): boolean; getSource(id: string): unknown; getZoom(): number; jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number } } };

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}${url}`);
  await expect(page.getByTestId("notices-button")).toBeVisible();
  await page.waitForFunction(() => !!(window as unknown as Partial<Win>).__cwMap?.loaded());
}

test.describe("layers, legend, time", () => {
  test("a forecast link opens on the run's day valid today, with its legend and provenance; no particle flow", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic");
    await expect(page.getByTestId("layer-forecast")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("legend-forecast")).toContainText("Probability that particulate domoic acid exceeds 500 ng per litre");
    await expect(page.getByTestId("legend-forecast")).toContainText("not a toxin measurement");
    await expect(page.getByTestId("stamp-forecast")).toContainText("issued Oct 8");
    // the day valid today (Oct 8 is the run's first forecast day) is selected, and marked today
    const today = page.getByTestId("timebar").locator('[role="radio"][aria-checked="true"]');
    await expect(today).toContainText("Oct 8");
    await expect(today).toContainText("today");
    await expect(new URL(page.url()).searchParams.get("lead")).toBe("1");
  });

  test("the timebar steps only through published frames, by keyboard too, and writes the URL", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1");
    const radios = page.getByTestId("timebar").locator('[role="radio"]');
    await expect(radios).toHaveCount(4);
    await radios.nth(1).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page).toHaveURL(/lead=2/);
    await page.keyboard.press("End");
    await expect(page).toHaveURL(/lead=3/);
    // nowcast and forecast are named on the frames, never a day beyond the run
    await expect(radios.nth(0)).toContainText("nowcast");
    await expect(radios.nth(3)).toContainText("forecast");
  });

  test("the seafloor-only view has no timebar; satellite and currents each bring their own legend", async ({ page }) => {
    await open(page, "/?layer=none");
    await expect(page.getByTestId("timebar")).toHaveCount(0);
    await openLayers(page);
    await page.getByTestId("layer-satellite").click();
    await expect(page.getByTestId("legend-satellite")).toContainText("not toxin");
    await expect(page.getByTestId("legend-satellite")).toContainText("a gap is not clear water");
    await page.getByTestId("satview-age").click();
    await expect(page.getByTestId("legend-age")).toContainText("Days since each pixel was last seen clear");
    await expect(page).toHaveURL(/age=1/);
    await page.getByTestId("layer-currents").click();
    await expect(page.getByTestId("legend-currents")).toContainText("No arrow means no observation, not calm water");
    await expect(page.getByTestId("timebar")).toBeVisible();
  });

  test("v2 links still open their view (region, var, lead, port, inspect)", async ({ page }) => {
    await open(page, "/?region=monterey_bay&layer=forecast&var=pseudo_nitzschia&lead=2&port=550");
    // the dock's layer button names the variable, and opens the layer list
    await expect(page.getByTestId("layer-capsule")).toContainText("Pseudo-nitzschia");
    await page.getByTestId("layer-capsule").click();
    await expect(page.getByTestId("variable-select")).toHaveValue("pseudo_nitzschia");
    await expect(page.getByTestId("timebar").locator('[aria-checked="true"]')).toContainText("Oct 9");
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
    await expect(page.getByTestId("inspect-title")).toHaveText("Monterey");
  });
});

test.describe("selection: map, inspection surface, charts and URL in step", () => {
  test("a station opens its latest samples, a year of samples on fixed axes, and its sources; the URL can be shared", async ({ page }) => {
    await open(page, "/?layer=forecast&sel=station:HABs-SantaCruzWharf");
    const panel = page.getByTestId("inspect-panel");
    await expect(panel).toHaveAttribute("data-kind", "station");
    await expect(page.getByTestId("inspect-title")).toHaveText("Santa Cruz Wharf");
    await expect(page.getByTestId("station-latest")).toContainText("ng/mL");
    // the year of samples is in the panel's "Evidence and sources"
    await openEvidence(page);
    await expect(page.getByTestId("station-series").locator("svg[role=img]")).toBeVisible();
    await expect(page.getByTestId("station-provenance")).toContainText("seawater, not seafood");
    // the selection is drawn on the map, above everything
    const hl = await page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("v4-highlight"));
    expect(hl).toBe(true);
    // keyboard reading of the chart: each sample's own date and value
    await page.getByTestId("series-svg").focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByTestId("series-pDA")).toContainText("sampled");
  });

  test("a model cell says its footprint, value, run, days and whether it is a harbour's own; a nearshore gap is never filled", async ({ page }) => {
    // Monterey's own cell has no C-HARM value (nearshore)
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&sel=cell:36.6045,-121.8908");
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "cell");
    await expect(page.getByTestId("cell-value")).toHaveText("No value");
    await expect(page.getByTestId("cell-gap")).toContainText("not modelled");
    // Monterey's own cell: said to be its own, and said to have no value (never "the model covers it")
    await expect(page.getByTestId("cell-harbor-text")).toContainText("C-HARM publishes no value for it");
    await expect(page.getByTestId("cell-leads").locator("li")).toHaveCount(4);
    // the nearest valued cell is one tap away and is a different, real cell
    await page.getByTestId("cell-nearest").click();
    await expect(page.getByTestId("cell-value")).toHaveText(/^\d+%$/);
    await expect(page.getByTestId("cell-harbor-text")).toContainText("No.");
    // back returns to the empty cell
    await page.getByTestId("inspect-back").click();
    await expect(page.getByTestId("cell-value")).toHaveText("No value");
  });

  test("an area reports what covers it, as counts of published cells, never a harbour estimate", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&sel=area:-122.2,36.5,-121.8,36.95");
    const area = page.getByTestId("inspect-area");
    await expect(area).toContainText("grid cells have a published value");
    await expect(area).toContainText("not an average for any harbour");
    await expect(area).toContainText("Monterey Wharf");
    await expect(area).toContainText("Monterey");
  });

  test("the area tool draws a box with the pointer and selects it", async ({ page }) => {
    await open(page, "/?layer=forecast&var=particulate_domoic&lead=1&region=monterey_bay");
    await page.getByTestId("area-tool").click();
    const ov = (await page.getByTestId("area-overlay").boundingBox())!;
    await page.mouse.move(ov.x + ov.width * 0.4, ov.y + ov.height * 0.3);
    await page.mouse.down();
    await page.mouse.move(ov.x + ov.width * 0.5, ov.y + ov.height * 0.45, { steps: 6 });
    await page.mouse.up();
    await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "area");
    await expect(page).toHaveURL(/sel=area/);
  });

  test("search finds harbours and stations, '/' focuses it, and Escape clears the selection", async ({ page }) => {
    await open(page, "/?layer=none");
    await page.keyboard.press("/");
    await expect(page.getByTestId("place-search-input")).toBeFocused();
    await page.keyboard.type("crescent");
    await page.keyboard.press("Enter");
    await expect(page.getByTestId("inspect-title")).toHaveText("Crescent City");
    await expect(page).toHaveURL(/sel=harbor%3A201|sel=harbor:201/);
    await page.locator("body").click({ position: { x: 5, y: 400 } }).catch(() => null);
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("inspect-panel")).toHaveCount(0);
  });

  test("a harbour's evidence leads to the briefing, and a briefing's 'Show on map' comes back to it", async ({ page }) => {
    await open(page, "/?layer=forecast&sel=harbor:550");
    await expect(page.getByTestId("open-briefing")).toHaveAttribute("href", /\/coast\?place=port-550/);
    await page.goto(`${OK}/coast?place=port-550`);
    await page.getByTestId("brief-map-toggle").click();
    const show = page.getByTestId("locator-explore");
    await show.click();
    await expect(page.getByTestId("back-to-briefing")).toContainText("Back to the Monterey briefing");
    await page.getByTestId("back-to-briefing").click();
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  });
});
