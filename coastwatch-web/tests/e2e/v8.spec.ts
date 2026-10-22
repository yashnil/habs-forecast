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

// the page's mock of /api/conditions must not be answered by the service worker's cache
test.describe(() => {
  test.use({ serviceWorkers: "block" });
  test("each card reading names its own instrument and age, and says why when there is none (v8 fix pass)", async ({ page }) => {
    await page.clock.setFixedTime(new Date("2026-10-22T14:00:00Z"));
    // the harbour gauge measured the water at 13:48; the buoy measured waves at 12:00 and reports no wind
    await page.route("**/api/conditions**", (r) =>
      r.fulfill({
        json: {
          lat: 36.6, lon: -121.9,
          marine: { status: "none", reason: "", source: "" }, alerts: { status: "none", reason: "", source: "" }, tides: { status: "none", reason: "", source: "" },
          buoy: { status: "ok", data: { waveFt: { value: 4.3, at: "2026-10-22T12:00:00Z" }, periodS: { value: 10, at: "2026-10-22T12:00:00Z" } }, source: "NDBC", url: "", fetchedAt: "", station: { id: "46236", name: "MONTEREY CANYON OUTER, CA (236)", km: 17.6 } },
          harbor: { status: "ok", data: { waterC: { value: 18.9, at: "2026-10-22T13:48:00Z" }, flags: "0,0,0" }, source: "CO-OPS", url: "", fetchedAt: "", station: { id: "9413450", name: "Monterey", km: 1.0 }, basis: "in-harbour" },
        },
      }),
    );
    await page.goto(`${OK}/coast`);
    const card = page.getByTestId("coast-start-port-550");
    await expect(card.getByTestId("landing-water-age")).toHaveText("12 min ago");
    await expect(card.getByTestId("landing-water")).toContainText("Tide gauge");
    // the older wave reading keeps its own age: never the newest reading's
    await expect(card.getByTestId("landing-waves-age")).toHaveText("2 h ago");
    await expect(card.getByTestId("landing-waves")).toContainText("Buoy");
    await expect(card.getByTestId("landing-wind-none")).toHaveText("Not reported");
    await expect(card.getByTestId("landing-readings")).not.toContainText("–");
    await expect(card.getByTestId("landing-sources")).toHaveText("Measured by Monterey tide gauge (NOAA CO-OPS), 1.0 km · Monterey Canyon Outer buoy (NDBC 46236), 18 km");
  });
});

test("the briefing keeps every change of wind direction the NWS forecasts", async ({ page }) => {
  // the replayed NWS period: "W wind 10 to 15 kt, becoming NW 5 to 10 kt late this evening, veering to N after midnight"
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.goto(`${OK}/coast?place=port-550`);
  const wind = page.getByTestId("glance-wind");
  await expect(wind).toContainText("W 10–15", { timeout: 15_000 });
  // the later stretches keep their own directions: the NW 5–10 is never drawn as W
  await expect(wind).toContainText("then NW 5–10 kt late this evening, then N after midnight");
  // on a phone the changes take the full row under wind and seas
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(wind).not.toContainText("then NW", { useInnerText: true });
  await expect(page.getByTestId("glance-wind-changes")).toHaveText("Wind then NW 5–10 kt late this evening, then N after midnight");
});

test("on a phone the tide's turn labels stay inside the strip and clear of each other and the now line", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.clock.setFixedTime(new Date("2026-10-13T06:00:00Z"));
  await page.goto(`${OK}/coast?place=port-550`);
  const tide = page.getByTestId("tide-strip");
  await expect(tide).toBeVisible({ timeout: 15_000 });
  const boxes = await tide.locator("svg").evaluate((svg) => {
    const s = svg.getBoundingClientRect();
    const labels = [...svg.querySelectorAll('[data-testid="tide-turn-label"]')].map((t) => (t as SVGGraphicsElement).getBoundingClientRect());
    const lines = [...svg.querySelectorAll("line[stroke='var(--nx-ink)']")].map((l) => l.getBoundingClientRect());
    return { s: { l: s.left, r: s.right }, labels: labels.map((b) => ({ l: b.left, r: b.right, t: b.top, b: b.bottom })), lines: lines.map((b) => ({ x: b.left + b.width / 2, t: b.top, b: b.bottom })) };
  });
  expect(boxes.labels.length).toBeGreaterThan(0);
  for (const b of boxes.labels) {
    expect(b.l).toBeGreaterThanOrEqual(boxes.s.l - 0.5);
    expect(b.r).toBeLessThanOrEqual(boxes.s.r + 0.5);
    // no stretch of the now line runs through a label
    for (const ln of boxes.lines) expect(ln.x < b.l || ln.x > b.r || ln.b <= b.t + 1 || ln.t >= b.b - 1).toBe(true);
  }
  for (let i = 0; i < boxes.labels.length; i++)
    for (let j = i + 1; j < boxes.labels.length; j++) {
      const a = boxes.labels[i], c = boxes.labels[j];
      expect(a.r <= c.l || c.r <= a.l || a.b <= c.t || c.b <= a.t).toBe(true);
    }
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
