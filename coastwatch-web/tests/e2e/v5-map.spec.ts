import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * v5 map: the coast filled honestly. The C-HARM layer extends the nearest NOAA cell to the shore
 * (labelled context, never a value), says "outside NOAA's model" in the bays and shows chlorophyll
 * there; the derived index (tests/fixture-derived/v1/derived, overlaid by copy-fixture-data) adds the 8-day outlook and the
 * gap-filling estimate, each with its provenance, and nothing breaks without it.
 */
const NOW = "2026-10-13T18:00:00Z";
type Win = { __cwMap: { jumpTo(o: unknown): void; project(p: number[]): { x: number; y: number }; loaded(): boolean; getSource(id: string): unknown; getStyle(): { layers: { id: string }[] }; once(e: string, f: () => void): void; areTilesLoaded(): boolean } };

async function open(page: Page, q: string, base = OK) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${base}/?${q}`);
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
}

async function jump(page: Page, lat: number, lon: number, zoom: number) {
  await page.evaluate(([la, lo, z]) => (window as unknown as Win).__cwMap.jumpTo({ center: [lo, la], zoom: z }), [lat, lon, zoom]);
  await page.waitForFunction(() => (window as unknown as Win).__cwMap.loaded() && (window as unknown as Win).__cwMap.areTilesLoaded(), null, { timeout: 15_000 });
}

async function hoverAt(page: Page, lat: number, lon: number) {
  const p = await page.evaluate(([la, lo]) => (window as unknown as Win).__cwMap.project([lo, la]), [lat, lon]);
  const box = (await page.getByTestId("map").boundingBox())!;
  await page.mouse.move(box.x + p.x, box.y + p.y);
  await page.mouse.move(box.x + p.x + 1, box.y + p.y);
}

test("the forecast tints the water NOAA does not model near shore; the nearest value is text only, with its pier-sample check", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=1");
  // v6 wording: what NOAA publishes, by variable, never that it "does not model" the water
  await expect(page.getByTestId("legend-outside")).toContainText("NOAA publishes no toxin probability here");
  await expect(page.getByTestId("legend-outside")).toContainText("no probability is shown there");
  await expect(page.getByTestId("legend-context-check")).toContainText("1,058 CalHABMAP pier samples");
  await expect(page.getByTestId("legend-context-check")).toContainText("no demonstrated skill at the shore");
  // nearshore cells of the fixture without a pDA value, next to one with a value (Monterey Bay's east shore)
  await jump(page, 36.78, -121.82, 11);
  const readout = page.getByTestId("hover-readout");
  let found = false;
  for (const [lat, lon] of [[36.76, -121.8], [36.73, -121.8], [36.79, -121.77], [36.82, -121.77], [36.7, -121.83]]) {
    await hoverAt(page, lat, lon);
    await expect(readout).toBeVisible();
    if ((await readout.getAttribute("data-tone")) === "context") {
      found = true;
      break;
    }
  }
  expect(found).toBe(true);
  await expect(page.getByTestId("hover-context")).toContainText(/^NOAA publishes no toxin probability for this water\. The nearest NOAA cell with one, \d+\.\d km away, reads \d+%: text only, not a forecast for this water\./);
  await expect(page.getByTestId("hover-context")).toContainText("no demonstrated skill at the shore");
  await expect(page.getByTestId("hover-label")).toHaveText("NOAA publishes no toxin probability here");
  // never a value of its own
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
});

test("San Francisco Bay: NOAA publishes no toxin probability there; the inspector points to pier samples and the filled chlorophyll", async ({ page }) => {
  await open(page, "layer=forecast&var=particulate_domoic&lead=1");
  // no chlorophyll or probability is painted under the forecast: one neutral tint
  expect(await page.evaluate(() => (window as unknown as Win).__cwMap.getStyle().layers.map((l) => l.id).filter((id) => id.startsWith("under-")))).toEqual([]);
  await jump(page, 37.75, -122.3, 10.5);
  await hoverAt(page, 37.75, -122.3);
  await expect(page.getByTestId("hover-readout")).toHaveAttribute("data-tone", "outside");
  await expect(page.getByTestId("hover-outside")).toContainText("NOAA publishes no toxin probability here or within 6 km");
  const box = (await page.getByTestId("map").boundingBox())!;
  const p = await page.evaluate(() => (window as unknown as Win).__cwMap.project([-122.3, 37.75]));
  await page.mouse.click(box.x + p.x, box.y + p.y);
  const out = page.getByTestId("cell-outside");
  await expect(out).toContainText("NOAA publishes no toxin probability here");
  await expect(page.getByTestId("cell-value")).toHaveText("No value");
  await expect(page.getByTestId("cell-piers")).toContainText("CalHABMAP pier samples");
  await page.getByTestId("switch-estimate").click();
  // the selection stays open; the folded layer deck names the new layer
  await expect(page).toHaveURL(/layer=olci300/);
  await expect(page.getByTestId("layer-capsule")).toContainText("Satellite chlorophyll");
});

test("the 8-day outlook: its own layer, horizon, skill against baselines, caveats and readout label", async ({ page }) => {
  await open(page, "layer=outlook");
  await expect(page.getByTestId("layer-outlook")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-outlook")).toContainText("Outlook, not observed");
  await expect(page.getByTestId("outlook-horizon")).toBeVisible();
  await expect(page.getByTestId("outlook-issued")).toHaveText("Issued Oct 12");
  await expect(page.getByTestId("outlook-window")).toHaveText("Mean Oct 13 → Oct 20");
  await expect(page.getByTestId("stamp-outlook")).toContainText("Outlook, not observed.");
  const skill = page.getByTestId("outlook-skill");
  await expect(skill).toContainText("lower is better");
  for (const v of ["0.754", "0.850", "0.892"]) await expect(skill).toContainText(v);
  await expect(skill).toContainText("This outlook (climatology + AR(1))");
  await expect(page.getByTestId("outlook-is-baseline")).toHaveText("This outlook is the climatology + AR(1) baseline (no learned model).");
  await expect(page.getByTestId("outlook-coverage")).toContainText("about 11–19 km offshore");
  await expect(page.getByTestId("derived-caveats")).toContainText("not toxin");
  await expect(page.getByTestId("legend-outlook")).toContainText("Outlook, not observed.");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("outlook"))).toBe(true);
  await jump(page, 36.85, -122.1, 9.5);
  await hoverAt(page, 36.85, -122.1);
  await expect(page.getByTestId("hover-label")).toHaveText("Outlook, not observed");
  await expect(page.getByTestId("hover-value")).toHaveText(/^[\d.]+$/);
  await expect(page.getByTestId("hover-readout")).toContainText(/typically [\d.]+–[\d.]+ mg\/m³/);
});

test("the estimate lies under the observed pixels, stippled and labelled, and can be turned off", async ({ page }) => {
  await open(page, "layer=olci300");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(true);
  const order = await page.evaluate(() => (window as unknown as Win).__cwMap.getStyle().layers.map((l) => l.id));
  expect(order.indexOf("estimate-raster")).toBeLessThan(order.indexOf("satellite-raster"));
  await expect(page.getByTestId("legend-estimate")).toContainText("Estimated, not observed.");
  // v6: the class grid's shoreline strip has its own legend entry
  await expect(page.getByTestId("legend-strip")).toContainText("Extrapolated toward shore, never observed.");
  // the pipeline's own words, verbatim (the fixture's multiplier is deliberately not a real one)
  await expect(page.getByTestId("legend-strip")).toContainText("x9.9 test multiplier");
  await expect(page.getByTestId("legend-turbid")).toContainText("Bay chlorophyll is approximate (turbid water)");
  await page.getByTestId("estimate-provenance").locator("summary").click();
  await expect(page.getByTestId("estimate-validation")).toContainText("0.6–3 km");
  await expect(page.getByTestId("estimate-method")).toBeVisible();
  await expect(page.getByTestId("estimate-class-validation")).toContainText("0.58");
  // SF Bay has no observed pixel in the fixture: the estimate is what is read there, so labelled
  await jump(page, 37.75, -122.3, 10.5);
  await hoverAt(page, 37.75, -122.3);
  await expect(page.getByTestId("hover-label")).toHaveText("Estimated, not observed");
  // class 3 of the fixture's class grid: a bay, on a borrowed background
  await expect(page.getByTestId("hover-readout")).toContainText("Fixture: bay or estuary, borrowed background; turbid water");
  await expect(page.getByTestId("hover-readout")).not.toContainText("clear pixels");
  await page.getByTestId("sat-estimate-toggle").uncheck();
  await expect(page).toHaveURL(/est=0/);
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(false);
});

test("without a derived index nothing derived is offered and the map still works", async ({ page }) => {
  await open(page, "layer=olci300", FAILED);
  await expect(page.getByTestId("layer-satellite")).toBeVisible();
  await expect(page.getByTestId("layer-outlook")).toHaveCount(0);
  await expect(page.getByTestId("legend-estimate")).toHaveCount(0);
  expect(await page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("estimate"))).toBe(false);
  // a link to the outlook or to sea temperature falls back to the default layer
  for (const q of ["layer=outlook", "layer=sst&sel=harbor:593"]) {
    await open(page, q, FAILED);
    await expect(page).not.toHaveURL(/layer=(outlook|sst)/);
    await expect(page.getByTestId("layer-sst")).toHaveCount(0);
  }
});

test("on a phone the outlook is one chip away and its horizon reads", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await page.getByTestId("chip-outlook").first().click();
  await expect(page.getByTestId("chip-outlook").first()).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("outlook-horizon").first()).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await ctx.close();
});

test("sea temperature: an analysis (MUR foundation temperature), a °F legend, °C and °F in the readout, a perceptually even ramp", async ({ page }) => {
  await open(page, "layer=sst");
  await expect(page.getByTestId("layer-sst")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-sst")).toContainText("Analysis of Oct 12");
  await expect(page.getByTestId("legend-sst")).toContainText("°F");
  await expect(page.getByTestId("sst-caveat")).toContainText("foundation temperature");
  await expect(page.getByTestId("sst-caveat")).toContainText("may mark a front where satellites saw clear sky");
  await expect(page.getByTestId("legend-sst")).not.toContainText("cloud");
  await expect(page.getByTestId("stamp-sst")).toContainText("Analysis, from observations");
  await expect(page.getByTestId("stamp-sst")).toContainText("JPL MUR SST");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap.getSource("sst"))).toBe(true);
  await jump(page, 36.85, -122.2, 9.5);
  await hoverAt(page, 36.85, -122.2);
  await expect(page.getByTestId("hover-label")).toHaveText("Analysis");
  await expect(page.getByTestId("hover-readout")).toContainText(/[\d.]+\s*°C/);
  await expect(page.getByTestId("hover-readout")).toContainText(/≈ [\d.]+ °F/);
});

test("My Coast's sea-temperature link opens the layer with the harbour selected", async ({ page }) => {
  await open(page, "layer=sst&sel=harbor:593");
  await expect(page.getByTestId("inspect-panel")).toHaveAttribute("data-kind", "harbor");
  await expect(page.getByTestId("layer-capsule")).toContainText("Sea temperature");
  await expect(page).toHaveURL(/layer=sst/);
});

// ---------------------------------------------------------------- v6 (track X): bays, the shoreline strip, cross-links

test("the outlook is never drawn or read in San Francisco Bay; the open coast still reads", async ({ page }) => {
  await open(page, "layer=outlook");
  await jump(page, 37.75, -122.3, 10.5);
  await hoverAt(page, 37.75, -122.3);
  await expect(page.getByTestId("hover-bay")).toHaveText("No outlook in bays (turbid water; MODIS chlorophyll unreliable).");
  await expect(page.getByTestId("hover-value")).toHaveCount(0);
  // the fixture writes values over the bay on purpose: the drawn tile is transparent there
  const p = await page.evaluate(() => (window as unknown as Win).__cwMap.project([-122.3, 37.75]));
  const box = (await page.getByTestId("map").boundingBox())!;
  await page.mouse.click(box.x + p.x, box.y + p.y);
  await expect(page.getByTestId("point-outlook")).toContainText("No outlook in bays");
});

test("the shoreline strip reads as extrapolated, in the pipeline's words, never as filled from clear pixels", async ({ page }) => {
  await open(page, "layer=olci300");
  // the fixture's class-2 band: the San Mateo coast, offshore of Half Moon Bay
  await jump(page, 37.45, -122.48, 11);
  await hoverAt(page, 37.45, -122.48);
  await expect(page.getByTestId("hover-readout")).toHaveAttribute("data-tone", "strip");
  await expect(page.getByTestId("hover-label")).toHaveText("Extrapolated, not observed");
  await expect(page.getByTestId("hover-readout")).toContainText("Fixture: extrapolated toward shore from water about 1 km out (x9.9 test multiplier), never observed; not checkable");
  await expect(page.getByTestId("hover-readout")).not.toContainText("clear pixels");
});

test("every inspector ends with Brief this (keeping the trip) and Research this (station, variable, day, way back)", async ({ page }) => {
  const from = encodeURIComponent("/coast?place=port-592&for=commercial&species=dungeness&date=2026-10-14");
  await open(page, `layer=forecast&var=particulate_domoic&lead=1&sel=harbor:592&from=${from}`);
  const brief = page.getByTestId("brief-this");
  await expect(brief).toHaveAttribute("href", "/coast?place=port-592&for=commercial&species=dungeness&date=2026-10-14");
  const href = (await page.getByTestId("research-this").getAttribute("href"))!;
  const u = new URL(href, OK);
  expect(u.searchParams.get("place")).toBe("port-592");
  expect(u.searchParams.get("variable")).toBe("pDA");
  expect(u.searchParams.get("day")).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  expect(u.searchParams.get("from")).toMatch(/sel=harbor(:|%3A)592/);
  await page.getByTestId("research-this").click();
  await expect(page).toHaveURL(/\/research\?/);
  await expect(page.getByTestId("research-arrived")).toContainText(/Arrived from Moss Landing · nearest station .+, \d/);
  await page.getByTestId("research-back").click();
  await expect(page).toHaveURL(/sel=harbor(:|%3A)592/);
  // station, cell, point and area inspectors carry the same two links
  for (const sel of ["station:HABs-SantaCruzWharf", "cell:36.85,-122.1", "point:36.85,-122.1", "area:-122.2,36.7,-121.9,36.9"]) {
    await open(page, `layer=forecast&sel=${sel}`);
    await expect(page.getByTestId("inspect-crosslinks")).toBeVisible();
    await expect(page.getByTestId("brief-this")).toHaveAttribute("href", /^\/coast\?place=/);
    await expect(page.getByTestId("research-this")).toHaveAttribute("href", /^\/research\?station=/);
  }
});

test("search finds species and harvest sites; a species opens My Coast for it", async ({ page }) => {
  await open(page, "layer=forecast&sel=harbor:550");
  const input = page.getByTestId("place-search-input");
  await input.fill("clam beach");
  await page.getByRole("option", { name: /Clam Beach/ }).click();
  await expect(page.getByTestId("inspect-title")).toHaveText("Clam Beach, Humboldt");
  await expect(page.getByTestId("brief-this")).toHaveAttribute("href", /place=site-/);
  await page.getByTestId("inspect-close").click();
  await input.fill("dungeness");
  await expect(page.getByRole("option").first()).toContainText("Dungeness crab");
  await input.press("Enter");
  await expect(page).toHaveURL(/\/coast\?.*species=dungeness/);
});

test("a first-time visitor sees what Neritic is in one line, until dismissed", async ({ browser }) => {
  // a fresh visitor: none of the suite's returning-visitor storage
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  const line = page.getByTestId("explore-purpose");
  await expect(line).toContainText("Neritic: California coastal conditions, models and notices — tap a harbour for its briefing", { timeout: 15_000 });
  await page.getByTestId("explore-purpose-dismiss").click();
  await expect(line).toHaveCount(0);
  await page.reload();
  await page.waitForTimeout(500);
  await expect(line).toHaveCount(0);
  await ctx.close();
});
