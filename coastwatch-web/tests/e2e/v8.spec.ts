import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * v8 (market pass): what the best ocean and weather apps do that Neritic lacked.
 * - The time bar plays the published frames in turn (Windy, RainViewer, Worldview), stops at any
 *   other input, and marks where the nowcast ends and the forecast frames begin.
 * - At state scale the data rasters are smoothed; from harbour scale every pixel is a real cell.
 * - My Coast opens on a front door: the starting places as cards with what is measured there now.
 * - The briefing draws the day's tide from NOAA's predicted highs and lows.
 */
const NOW = "2026-10-13T18:00:00Z";
type Win = { __cwMap: { loaded(): boolean; getZoom(): number; jumpTo(o: unknown): void; getLayer(id: string): unknown; getPaintProperty(id: string, p: string): unknown; once(e: string, f: () => void): void } };

async function open(page: Page, q: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?${q}`);
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
}

const checked = (page: Page) => page.getByTestId("timebar").getByRole("radio", { checked: true });

test("the time bar plays the forecast frames in turn and stops at any other input", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=0");
  const play = page.getByTestId("time-play");
  await expect(play).toHaveAttribute("aria-pressed", "false");
  // the nowcast and the forecast frames are set apart
  await expect(page.getByTestId("time-forecast-break")).toHaveCount(1);
  const first = await checked(page).getAttribute("data-testid");
  await play.click();
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await expect(play).toHaveAccessibleName("Pause");
  await expect.poll(async () => checked(page).getAttribute("data-testid"), { timeout: 4000 }).not.toBe(first);
  // choosing a frame by hand stops playback where the reader put it
  await page.getByTestId("time-lead0").click();
  await expect(play).toHaveAttribute("aria-pressed", "false");
  await page.waitForTimeout(1500);
  await expect(checked(page)).toHaveAttribute("data-testid", "time-lead0");
  // space plays and pauses from the time bar
  await page.getByTestId("time-lead0").focus();
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await page.keyboard.press("Space");
  await expect(play).toHaveAttribute("aria-pressed", "false");
});

test("data rasters are smoothed at state scale and drawn cell by cell from harbour scale", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=1");
  const mode = () => page.evaluate(() => (window as unknown as Win).__cwMap.getPaintProperty("forecast-raster", "raster-resampling"));
  expect(await page.evaluate(() => (window as unknown as Win).__cwMap.getZoom())).toBeLessThan(8);
  await expect.poll(mode).toBe("linear");
  await page.evaluate(() => (window as unknown as Win).__cwMap.jumpTo({ center: [-121.95, 36.8], zoom: 9.5 }));
  await expect.poll(mode).toBe("nearest");
});

test("My Coast's front door: the six starting places as cards, each with measured readings or saying none", async ({ page }) => {
  await page.goto(`${OK}/coast`);
  await expect(page.getByTestId("coast-title")).toHaveText("Where are you fishing?");
  const cards = page.getByTestId("coast-landing").locator("li");
  await expect(cards).toHaveCount(6);
  const monterey = page.getByTestId("coast-start-port-550");
  await expect(monterey).toContainText("Monterey");
  await expect(monterey.getByTestId("landing-readings")).toContainText("Water");
  // the readings settle: a measured line with its time and buoy, or the plain statement that none is near
  await expect(monterey).toContainText(/Measured|No reading near here/, { timeout: 15_000 });
  await monterey.click();
  await expect(page).toHaveURL(/place=port-550/);
});

test("the briefing draws the day's tide from NOAA's predicted highs and lows", async ({ page }) => {
  // the servers replay NOAA responses recorded for 2026-10-13 06:00 UTC (Oct 12 in California): the page's clock too
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.goto(`${OK}/coast?place=port-550`);
  const tide = page.getByTestId("tide-strip");
  await expect(tide).toBeVisible({ timeout: 15_000 });
  await expect(tide).toContainText("NOAA predictions");
  await expect(tide.getByRole("img")).toHaveAttribute("aria-label", /Predicted tide for the day.*, in feet above MLLW: .*(high|low) \d+\.\d ft at \d{1,2}:\d{2} (AM|PM)/);
});
