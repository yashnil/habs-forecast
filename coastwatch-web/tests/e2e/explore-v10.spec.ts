import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";
import { openDisplay } from "./helpers";

/**
 * v10 Explore, the living instrument (docs/coastwatch/v10/system.md, direction A): the layer index, the
 * time rail (real frames only, a hard now line, observed and model segments, keyboard and pointer
 * scrubbing) and the selected place's reading panel (each value with its family, time, source and
 * native footprint).
 */
const NOW = "2026-10-08T20:00:00Z"; // 1 PM PDT, Oct 8: the run's first forecast day
type Win = { __cwMap: { loaded(): boolean; getPaintProperty(id: string, p: string): unknown } };

async function open(page: Page, q: string) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?${q}`);
  await page.waitForFunction(() => !!(window as unknown as Partial<Win>).__cwMap?.loaded());
}

test.describe("the time rail", () => {
  test("real frames on one axis: the model's days as radios, the observations as context, a hard now line inside today", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=1");
    const rail = page.getByTestId("timebar");
    // the layer on the map: its four published days, and only those, are the rail's radios
    await expect(rail.locator('[role="radio"]')).toHaveCount(4);
    await expect(page.getByTestId("rail-label-model")).toContainText("C-HARM model · valid");
    await expect(page.getByTestId("rail-label-model")).toContainText("run Oct 8");
    // the satellite's overpasses are drawn in their own (observed) segment, muted, never as radios of this layer
    await expect(page.getByTestId("rail-label-observed")).toContainText("Satellite · observed");
    const ctx = rail.locator('[data-testid^="rail-ctx-sat-"]');
    expect(await ctx.count()).toBeGreaterThan(0);
    await expect(ctx.first()).toHaveAttribute("aria-hidden", "true");
    // now: one hard line, inside today's frame (Oct 8, 1 PM)
    const now = (await page.getByTestId("rail-now").boundingBox())!;
    const today = (await page.getByTestId("time-lead1").boundingBox())!;
    expect(now.x).toBeGreaterThan(today.x + today.width * 0.4);
    expect(now.x).toBeLessThan(today.x + today.width * 0.7);
    await expect(rail).toContainText(/Now: Thu, Oct 8, 1:00 PM PDT/);
    // the nowcast and the forecast days are set apart; the forecast cells are hatched, the nowcast solid
    await expect(page.getByTestId("time-forecast-break")).toHaveCount(1);
  });

  test("keyboard steps through real frames only, and a pointer scrubs to the nearest one (a cut)", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=1");
    await page.getByTestId("time-lead1").focus();
    await page.keyboard.press("ArrowLeft");
    await expect(page).toHaveURL(/lead=0/);
    await expect(page.getByTestId("time-lead0")).toHaveAttribute("aria-checked", "true");
    await page.keyboard.press("End");
    await expect(page).toHaveURL(/lead=3/);
    await page.keyboard.press("Home");
    await expect(page).toHaveURL(/lead=0/);
    // drag from the nowcast across the rail: it lands on the frame under the pointer, never between frames
    const a = (await page.getByTestId("time-lead0").boundingBox())!;
    const c = (await page.getByTestId("time-lead2").boundingBox())!;
    await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
    await page.mouse.down();
    await page.mouse.move(c.x + c.width * 0.6, c.y + c.height / 2, { steps: 6 });
    await page.mouse.up();
    await expect(page).toHaveURL(/lead=2/);
    await expect(page.getByTestId("time-lead2")).toHaveAttribute("aria-checked", "true");
  });

  test("an observed frame on the rail switches the map to that layer and day; the seafloor alone has no frames", async ({ page }) => {
    await open(page, "layer=forecast&var=particulate_domoic&lead=1");
    await page.getByTestId("timebar").locator('[data-testid^="rail-ctx-sat-2"]').first().click();
    await expect(page).toHaveURL(/layer=olci300(:|%3A)2026-10-/);
    await expect(page.getByTestId("layer-satellite")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("timebar").locator('[role="radio"][aria-checked="true"]')).toContainText("overpass");
    await open(page, "layer=none");
    await expect(page.getByTestId("timebar")).toHaveCount(0);
    await expect(page.getByTestId("time-rail")).toBeVisible();
    await expect(page.getByTestId("time-rail").locator('[role="radio"]')).toHaveCount(0);
  });
});

test.describe("the layer index", () => {
  test("every family by an honest name and kind, one radio group, arrow keys switch; the field's opacity is one level in", async ({ page }) => {
    await open(page, "layer=olci300");
    const list = page.getByTestId("layer-list");
    await expect(list).toHaveAttribute("role", "radiogroup");
    await expect(page.getByTestId("layer-forecast")).toContainText("Bloom & toxin probability");
    await expect(page.getByTestId("layer-forecast")).toContainText("Model · NOAA C-HARM");
    await expect(page.getByTestId("layer-satellite")).toContainText("Observed");
    await expect(page.getByTestId("layer-outlook")).toContainText("Outlook, not observed");
    await expect(page.getByTestId("layer-sst")).toContainText("Analysis");
    await expect(page.getByTestId("layer-none")).toContainText("Reference");
    // the display controls are folded until asked for
    await expect(page.getByTestId("layer-opacity")).toBeHidden();
    await openDisplay(page);
    await page.getByTestId("layer-opacity-range").fill("50");
    await expect.poll(() => page.evaluate(() => (window as unknown as Win).__cwMap.getPaintProperty("satellite-raster", "raster-opacity"))).toBe(0.5);
    // the legend keeps the full-opacity colours and says what a gap is
    await expect(page.getByTestId("legend-gap")).toContainText("not low chlorophyll");
    await page.getByTestId("layer-satellite").focus();
    await page.keyboard.press("ArrowDown");
    await expect(page).toHaveURL(/layer=outlook/);
    await expect(page.getByTestId("layer-outlook")).toBeFocused();
  });
});

test.describe("the reading panel", () => {
  test("a point reads every layer there, each with its family, source, native footprint and time; links keep the place", async ({ page }) => {
    await open(page, "layer=olci300&sel=point:36.85,-122.1");
    const panel = page.getByTestId("inspect-panel");
    await expect(panel).toBeVisible();
    await expect(page.getByTestId("inspect-title")).toHaveText(/^A point in /);
    await expect(panel).toContainText(/36\.850°N 122\.100°W · [\d.]+ km [NESW]+ of /);
    const model = page.getByTestId("point-forecast");
    await expect(model).toContainText("NOAA C-HARM model");
    await expect(model.getByTestId("point-when")).toContainText(/native 3 km · valid/);
    const sat = page.getByTestId("point-satellite");
    await expect(sat.getByTestId("point-when")).toContainText(/observed/);
    // the radar is read here on every layer, from its own 2 km cell (never as calm water where it saw nothing)
    await expect(page.getByTestId("point-currents").getByTestId("point-when")).toContainText(/HF radar, native 2 km/);
    // My Coast for this place, and Research with the way back to this exact view (its footprint included)
    await expect(page.getByTestId("brief-this")).toHaveAttribute("href", /\/coast\?place=/);
    const research = await page.getByTestId("research-this").getAttribute("href");
    expect(decodeURIComponent(research!)).toMatch(/from=\/\?layer=olci300.*sel=point:36\.85,-122\.1.*fp=/);
  });
});
