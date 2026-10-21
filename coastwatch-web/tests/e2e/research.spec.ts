import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Research (v4): a station's record on fixed axes over a range the station actually covers, the
 * C-HARM history beside it (never compared numerically), provenance and QC, an export that carries
 * its provenance, and the native C-HARM grid with a transect that keeps gaps as gaps.
 */
const NOW = "2026-10-08T20:00:00Z";

test("a station's record: ranges, the model beside it, provenance, QC", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await expect(page.getByTestId("research-title")).toHaveText("Santa Cruz Wharf");
  await expect(page.getByTestId("rvar-pDA")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("series-ref")).toContainText("C-HARM threshold, 500 ng/L");
  await expect(page.getByTestId("model-panel")).toContainText("never compared with them as numbers");
  await page.getByTestId("range-all").click();
  await expect(page.getByTestId("research-stats")).toContainText("never a zero");
  await expect(page.getByTestId("research-provenance")).toContainText("Method");
  // provenance and QC are one tab away (v7): mounted throughout, shown when their tab is chosen
  await page.getByTestId("research-tab-provenance").click();
  await expect(page.getByTestId("research-qc").locator("li").first()).toBeVisible();
});

test("the export is the window's samples with a provenance header", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await page.getByTestId("research-export-menu").click();
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("research-export").click()]);
  const text = readFileSync((await dl.path())!, "utf8");
  expect(text).toContain("# Neritic research export: Santa Cruz Wharf");
  expect(text).toContain("empty value = not measured (never zero)");
  expect(text).toContain("Not a seafood test");
  expect(text.split("\n").find((l) => !l.startsWith("#"))).toBe("sample_time_utc,value,qualifier");
});

test("a My Coast place opens on its nearest station; a transect lists cells and keeps gaps", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?place=port-550`);
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await page.getByTestId("research-tab-grid").click();
  await page.getByTestId("tool-transect").click();
  const map = page.getByTestId("research-grid").locator(".maplibregl-canvas");
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  // the map sits under the tab's controls: in view before it is clicked
  await map.scrollIntoViewIfNeeded();
  const b = (await map.boundingBox())!;
  await page.mouse.click(b.x + b.width * 0.15, b.y + b.height * 0.4);
  await page.mouse.click(b.x + b.width * 0.85, b.y + b.height * 0.6);
  await expect(page.getByTestId("transect")).toContainText("cells crossed");
  await expect(page.getByTestId("transect")).toContainText("no interpolation");
});

// ---------------------------------------------------------------- scientific tools (lineage, QC, grid controls, area, exports, shared cursor)

const manifest = JSON.parse(readFileSync("tests/fixture-data/v1/manifest.json", "utf8")) as { layers: { group_id: string; variable: string; time: { lead_days: number }; qc: { checks: { name: string }[] } }[] };
type Win = { __cwMap?: { loaded(): boolean; getSource(id: string): unknown } };
// the grid map is in the Native grid tab (v7), made only once it is shown: open the tab, bring it near
const gridTab = (page: import("@playwright/test").Page) => page.getByTestId("research-tab-grid").click();
const mapReady = async (page: import("@playwright/test").Page) => {
  await gridTab(page);
  await page.getByTestId("research-map").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
};
const csvOf = async (page: import("@playwright/test").Page, testid: string) => {
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId(testid).click()]);
  return readFileSync((await dl.path())!, "utf8");
};

test("lineage and QC of the grid layer; lead and variable come from the run itself", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  const lin = page.getByTestId("research-lineage");
  await expect(lin).toContainText("wvcharmV3_0day");
  await expect(page.getByTestId("lineage-issued")).toContainText("inferred");
  await expect(page.getByTestId("lineage-upstream")).toContainText("DINEOF");
  await expect(page.getByTestId("lineage-run")).toHaveText("fixture");
  await expect(page.getByTestId("lineage-licence")).toContainText("NOAA CoastWatch");
  await expect(page.getByTestId("lineage-quant")).toContainText("16-bit");
  const checks = manifest.layers.find((l) => l.group_id === "charm" && l.variable === "particulate_domoic" && l.time.lead_days === 0)!.qc.checks;
  await gridTab(page);
  await expect(page.getByTestId("lineage-qc").locator("li")).toHaveCount(checks.length);
  for (const c of checks) await expect(page.getByTestId("lineage-qc")).toContainText(c.name);
  // only the run's own four days, labelled
  await expect(page.getByTestId("grid-controls").getByRole("radio", { name: /nowcast|forecast/ })).toHaveCount(4);
  await page.getByTestId("lead-2").click();
  await expect(page.getByTestId("grid-heading")).toContainText("forecast for Fri, Oct 9");
  await expect(page.getByTestId("lineage-dataset")).toHaveText("wvcharmV3_2day");
  await page.getByTestId("gvar-cellular_domoic").click();
  await expect(page.getByTestId("grid-heading")).toContainText("Cellular DA");
  // cell edges are drawn from the same grid, and can be hidden
  await mapReady(page);
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap!.getSource("forecast-edges"))).toBe(true);
  await page.getByTestId("grid-edges").uncheck();
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap!.getSource("forecast-edges"))).toBe(false);
});

test("a cell: precise value with its rounding, plainly a model cell; exported with provenance", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  const canvas = page.getByTestId("research-grid").locator(".maplibregl-canvas");
  await canvas.scrollIntoViewIfNeeded();
  const b = (await canvas.boundingBox())!;
  await page.mouse.click(b.x + b.width * 0.25, b.y + b.height * 0.6);
  await expect(page.getByTestId("cell-precise")).toContainText(/probability 0\.\d{3} ± 0\.0000076/);
  await expect(page.getByTestId("inspect-cell")).toContainText("Model output for one 3 km cell, not a toxin measurement");
  const csv = await csvOf(page, "export-cell");
  expect(csv).toContain("# Product class: model output");
  expect(csv).toContain("wvcharmV3_0day, wvcharmV3_1day, wvcharmV3_2day, wvcharmV3_3day");
  expect(csv).toContain("# Model probability, not a toxin measurement.");
  expect(csv).toContain("Empty value = no published value");
  expect(csv).toContain("lead_days,kind,valid_date,value\n0,nowcast,2026-10-07,");
});

test("units in words, the PN reference line, and the shared cursor reads both lines without comparing them", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await expect(page.getByTestId("series-axis-title")).toHaveText("pDA (ng/mL, log scale)");
  await expect(page.getByTestId("cursor-measured")).toContainText("Measured at the pier, Wed, Sep 30");
  await expect(page.getByTestId("cursor-model")).toContainText("C-HARM nowcast, same day");
  await expect(page.getByTestId("cursor-model")).toContainText(/\d+%.*median of \d+ cells within 15 km/);
  // another day, read on the model panel: the cursor moves on both charts (60% across: months from
  // the newest sample, whatever the platform's text metrics do to the axis)
  // (in view first: a pointer cannot reach a panel below the fold, and how far down it sits depends
  // on the platform's text metrics)
  await page.getByTestId("model-svg").scrollIntoViewIfNeeded();
  const m = (await page.getByTestId("model-svg").boundingBox())!;
  await page.mouse.move(m.x + m.width * 0.6, m.y + m.height / 2);
  await expect(page.getByTestId("cursor-measured")).not.toContainText("Sep 30");
  // (a vertical line has no width, so it is checked as present, not "visible")
  await expect(page.getByTestId("series-cursor")).toHaveCount(1);
  await expect(page.getByTestId("model-cursor")).toHaveCount(1);
  expect(Math.abs(Number(await page.getByTestId("model-cursor").getAttribute("x1")) - m.width * 0.6)).toBeLessThan(m.width * 0.02);
  const text = await page.getByTestId("research").innerText();
  expect(text).not.toMatch(/correlat|skill score|hit rate|accuracy|R²/i);
  // model history export (in the header's Export menu)
  await page.getByTestId("research-export-menu").click();
  const csv = await csvOf(page, "export-model");
  expect(csv).toContain("date,value,n_cells,partial");
  expect(csv).toContain("# Model probability, not a toxin measurement.");
  await page.getByTestId("rvar-pn_seriata").click();
  await expect(page.getByTestId("series-ref")).toContainText("10,000 cells/L");
  await page.getByTestId("rvar-temp").click();
  await expect(page.getByTestId("series-axis-title")).toHaveText("Temperature (°C)");
  await expect(page.getByTestId("research-chart")).not.toContainText("degree_C");
  await expect(page.getByTestId("research-provenance")).toContainText("°C");
  await expect(page.getByTestId("research-provenance")).not.toContainText("degree_C");
});

test("an area: counted cells, gaps never filled, exported cell by cell", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("tool-area").click();
  const b = (await page.getByTestId("area-overlay").boundingBox())!;
  await page.mouse.move(b.x + b.width * 0.1, b.y + b.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width * 0.4, b.y + b.height * 0.7, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByTestId("area-model")).toContainText(/of \d+ grid cells have a published value/);
  await expect(page.getByTestId("area-model")).toContainText("not an average for any harbour and not a measurement");
  const csv = await csvOf(page, "export-area");
  expect(csv.split("\n").find((l) => !l.startsWith("#"))).toBe("row,col,lat,lon,value");
  expect(csv).toContain("# Licence: NOAA CoastWatch");
  expect(csv).toContain("# Pipeline run: fixture");
});

test("the transect export lists every crossed cell; a gap is an empty value", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("tool-transect").click();
  const b = (await page.getByTestId("research-grid").locator(".maplibregl-canvas").boundingBox())!;
  // offshore, across the coast and onto land: valued cells, then cells without a value
  await page.mouse.click(b.x + b.width * 0.1, b.y + b.height * 0.5);
  await page.mouse.click(b.x + b.width * 0.6, b.y + b.height * 0.3);
  await expect(page.getByTestId("transect")).toContainText("cells crossed");
  const csv = await csvOf(page, "export-transect");
  const rows = csv.trim().split("\n").filter((l) => !l.startsWith("#"));
  expect(rows[0]).toBe("km,row,col,lat,lon,value,in_domain");
  expect(rows.slice(1).some((r) => r.split(",")[5] === "")).toBe(true);
  expect(rows.slice(1).some((r) => /^0\.\d+$/.test(r.split(",")[5]))).toBe(true);
});

// ---------------------------------------------------------------- observation grids (satellite chlorophyll, HF radar)

/** the grid map, in view and settled (the panel beside it changes the page height) */
async function gridMap(page: import("@playwright/test").Page) {
  await page.waitForTimeout(300);
  const canvas = page.getByTestId("research-grid").locator(".maplibregl-canvas");
  await canvas.scrollIntoViewIfNeeded();
  return (await canvas.boundingBox())!;
}

test("satellite: one published pixel with its own date and sensor; only published days are offered", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("gsrc-satellite").click();
  await expect(page.getByTestId("grid-heading")).toContainText("Satellite chlorophyll");
  await expect(page.getByTestId("research-grid")).toContainText("not a toxin");
  // the multi-sensor point: the sensor shown by the rule, each sensor at the point, its date
  const b = await gridMap(page);
  await page.mouse.click(b.x + b.width * 0.3, b.y + b.height * 0.65);
  await expect(page.getByTestId("obs-point")).toContainText("observation, not a forecast");
  await expect(page.getByTestId("obs-point-value")).toContainText("mg/m³ chlorophyll-a");
  await expect(page.getByTestId("obs-point-date")).toContainText(/Observed \w{3}, Oct \d/);
  await expect(page.getByTestId("obs-point-sensor")).toContainText(/Sentinel-3|VIIRS/);
  await expect(page.getByTestId("obs-point-members")).toContainText("VIIRS 750 m");
  // footprint drawn as a pixel outline on the map
  await expect.poll(() => page.evaluate(() => JSON.stringify((window as unknown as { __cwMap: { getSource(id: string): { serialize(): { data: unknown } } } }).__cwMap.getSource("v4-highlight")?.serialize().data ?? null))).toContain('"pixel"');
  // OLCI: days published without a clear pixel cannot be chosen; the pixel on each published day
  await page.getByTestId("sat-product-olci300").click();
  await expect(page.getByTestId("sat-day-2026-10-03")).toBeDisabled();
  await expect(page.getByTestId("sat-day-2026-10-03")).toContainText("no clear pixel");
  await expect(page.getByTestId("obs-day")).toHaveCount(7);
  await expect(page.getByTestId("obs-day-strip")).toContainText("not low chlorophyll");
  const csv = await csvOf(page, "export-obs-point");
  expect(csv).toContain("# Product class: observation (satellite chlorophyll-a), not a forecast");
  expect(csv).toContain("Sentinel-3 300 m");
  expect(csv).toContain("# Pipeline run: fixture");
  expect(csv).toContain("not low chlorophyll");
  expect(csv.split("\n").find((l) => !l.startsWith("#"))).toBe("layer,observed_date,chlorophyll_mg_m3");
  await page.getByTestId("sat-day-2026-10-05").click();
  await expect(page.getByTestId("grid-heading")).toContainText("overpass Mon, Oct 5");
  await expect(page.getByTestId("research-lineage")).toContainText("single day");
});

test("satellite transect: every crossed pixel with its date; gaps stay empty, never joined", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("gsrc-satellite").click();
  await page.getByTestId("sat-product-olci300").click();
  await page.getByTestId("tool-transect").click();
  const b = await gridMap(page);
  await page.mouse.click(b.x + b.width * 0.1, b.y + b.height * 0.5);
  await page.mouse.click(b.x + b.width * 0.6, b.y + b.height * 0.85);
  await expect(page.getByTestId("obs-transect-counts")).toContainText(/\d+ of \d+ pixels crossed have an observation; \d+ have none \(not low chlorophyll\)/);
  await expect(page.getByTestId("obs-transect")).toContainText("never joined across a gap");
  const csv = await csvOf(page, "export-obs-transect");
  const rows = csv.trim().split("\n").filter((l) => !l.startsWith("#"));
  expect(rows[0]).toBe("km,row,col,lat,lon,observed_date,chlorophyll_mg_m3,in_domain");
  const body = rows.slice(1).map((r) => r.split(","));
  // gaps: empty date and value; observations: a date and a value
  expect(body.some((r) => r[5] === "" && r[6] === "" && r[7] === "yes")).toBe(true);
  expect(body.some((r) => /^2026-10-0\d$/.test(r[5]) && Number(r[6]) > 0)).toBe(true);
  expect(csv).toContain("gaps stay empty rows");
});

test("multi-sensor area: sensor and date of every pixel, coverage, and an honest consistency check", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("gsrc-satellite").click();
  await page.getByTestId("tool-area").click();
  await page.waitForTimeout(300);
  await page.getByTestId("area-overlay").scrollIntoViewIfNeeded();
  const b = (await page.getByTestId("area-overlay").boundingBox())!;
  await page.mouse.move(b.x + b.width * 0.1, b.y + b.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.8, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByTestId("obs-area-counts")).toContainText(/of [\d,]+ display pixels \(300 m\) in the box have an observation/);
  await expect(page.getByTestId("obs-area-coverage")).toContainText("C-HARM ocean cells");
  await expect(page.getByTestId("obs-area-dates")).toContainText(/Oct \d: [\d,]+/);
  await expect(page.getByTestId("obs-area-sensors")).toContainText(/Sentinel-3 300 m [\d,]+ pixels; VIIRS 750 m .* pixels \(from [\d,]+ native 750 m VIIRS pixels\)/);
  const cons = page.getByTestId("obs-consistency");
  await expect(cons).toContainText("not a validation");
  // the fixture has no same-day pair in this box, and the pipeline published none: said, not computed
  await expect(page.getByTestId("obs-consistency-none")).toContainText("Agreement cannot be computed in this box");
  await expect(page.getByTestId("obs-consistency-published")).toContainText("no same-day pairs this week");
  const text = await page.getByTestId("research").innerText();
  expect(text).not.toMatch(/correlat|skill score|hit rate|accuracy|R²/i);
  const csv = await csvOf(page, "export-obs-area");
  expect(csv.split("\n").find((l) => !l.startsWith("#"))).toBe("row,col,lat,lon,sensor,observed_date,chlorophyll_mg_m3");
  expect(csv).toContain("nothing is averaged");
  expect(csv).toContain("noaacwN21VIIRSchlaSectorUSDaily");
});

test("radar: published hours only, a cell's every hour, gaps are no data and not calm water", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  await mapReady(page);
  await page.getByTestId("gsrc-radar").click();
  await expect(page.getByTestId("radar-hour").locator("option")).toHaveCount(25);
  await expect(page.getByTestId("grid-heading")).toContainText("HF radar");
  await expect(page.getByTestId("research-grid")).toContainText("not where a bloom will go");
  await expect.poll(() => page.evaluate(() => !!(window as unknown as Win).__cwMap!.getSource("currents"))).toBe(true);
  const b = await gridMap(page);
  await page.mouse.click(b.x + b.width * 0.3, b.y + b.height * 0.65);
  await expect(page.getByTestId("obs-point-value")).toContainText("m/s");
  await expect(page.getByTestId("obs-point-date")).toContainText("Observed Oct 8, 5:00 AM PDT");
  await expect(page.getByTestId("obs-radar-strip")).toContainText(/of 24 hours observed here/);
  await expect(page.getByTestId("obs-radar-strip")).toContainText("not calm water");
  await page.getByTestId("radar-prev").click();
  await expect(page.getByTestId("obs-point-date")).toContainText("Oct 8, 4:00 AM PDT");
  await page.getByTestId("radar-hour").selectOption("mean");
  await expect(page.getByTestId("grid-heading")).toContainText("24-hour mean ending");
  const csv = await csvOf(page, "export-obs-point");
  expect(csv).toContain("# Product class: observation (HF-radar surface currents), not a forecast");
  expect(csv).toContain("ucsdHfrW2");
  expect(csv).toContain("not calm water");
  expect(csv.split("\n").find((l) => !l.startsWith("#"))).toBe("layer,observed_hour_utc,u_ms,v_ms,speed_ms,direction_toward_deg");
  // a box: counted cells, gaps labelled
  await page.getByTestId("tool-area").click();
  await page.waitForTimeout(300);
  await page.getByTestId("area-overlay").scrollIntoViewIfNeeded();
  const o = (await page.getByTestId("area-overlay").boundingBox())!;
  await page.mouse.move(o.x + o.width * 0.1, o.y + o.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(o.x + o.width * 0.5, o.y + o.height * 0.8, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByTestId("obs-area-counts")).toContainText(/\d+ of \d+ radar cells in the box have an observation; \d+ have none \(no radar data or land: not calm water\)/);
});

test("the grid map is made only when its tab is opened, at its full height", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
  const box = page.getByTestId("research-map");
  await expect(box).toHaveAttribute("data-mounted", "false");
  expect(await page.evaluate(() => !!(window as unknown as Win).__cwMap)).toBe(false);
  await page.getByTestId("research-tab-grid").click();
  await box.scrollIntoViewIfNeeded();
  expect(Math.round((await box.boundingBox())!.height)).toBe(380);
  await expect(box).toHaveAttribute("data-mounted", "true");
  await page.waitForFunction(() => !!(window as unknown as Win).__cwMap?.loaded());
  await ctx.close();
});
