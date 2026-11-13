import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Journey 3 (docs/coastwatch/v12/design-system/journeys.md), on the v14 synthesis pass: a researcher arrives from Explore
 * with a station, a window and a variable in the address, reads the plot, moves the window, reads the table of
 * exactly that window's samples and exports exactly the window. Recorded as video on request (NX_MEDIA=1).
 */
const MEDIA = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v13/implementation/research") : path.resolve(__dirname, "../../../test-results/journey-media/research");
test.use({ storageState: { cookies: [], origins: [] } });

const params = (page: Page) => Object.fromEntries(new URL(page.url()).searchParams);
const DAY = 86_400_000;
const at = (d: string) => Date.parse(`${d}T12:00:00Z`);

async function open(page: Page, q: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}/research?${q}`);
  await expect(page.getByTestId("research-record")).toBeVisible();
}

/** The record's tools (table, exports, sources, pickers): a panel over the pass on a wide screen. */
async function tools(page: Page) {
  if ((await page.getByTestId("research-tools-open").getAttribute("aria-expanded")) !== "true") await page.getByTestId("research-tools-open").click();
  await expect(page.getByTestId("research-tools")).toBeVisible();
}

test("arrive from Explore, move the window, read the table, export exactly the window", async ({ browser }) => {
  // recorded as video at 1440 x 900 beside a parallel run: more than the default 45 s on a loaded machine
  test.setTimeout(90_000);
  mkdirSync(MEDIA, { recursive: true });
  const dir = path.join(test.info().outputDir, "video");
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, recordVideo: { dir, size: { width: 1440, height: 900 } }, acceptDownloads: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-MontereyWharf&from=2024-01-01&to=2025-12-31&var=pn_seriata");

  // the Explore selection survives into Research through the address: the link's variable is the one selected
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await expect(page.getByTestId("rail-pn_seriata")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("research-variable")).toContainText("Pseudo-nitzschia, seriata group");
  await expect(page.getByTestId("research-window")).toContainText("Jan 1, 2024");
  expect(params(page)).toMatchObject({ station: "HABs-MontereyWharf", var: "pn_seriata", from: "2024-01-01", to: "2025-12-31" });
  // every variable the station has measured is a tab; the model is its own panel behind its switch, named as a model
  for (const id of ["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]) await expect(page.getByTestId(`rail-${id}`)).toBeVisible();
  await page.getByTestId("research-model-switch").click();
  await expect(page.getByTestId("strip-model")).toContainText("Model");
  await page.getByTestId("research-model-switch").click();

  // the crosshair reads a sample under the pointer
  const plot = page.getByTestId("research-plot");
  const box = (await plot.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.45, box.y + 120, { steps: 12 });
  await expect(plot.locator("[data-crosshair]")).toBeVisible();
  await expect(page.getByTestId("plot-tip").first()).toBeVisible();
  await page.waitForTimeout(400);

  // the brush: the window's end, a year earlier with Shift+Arrow; the address follows
  const end = page.getByRole("slider", { name: "Window end" });
  await end.focus();
  await end.press("Shift+ArrowLeft");
  await expect.poll(() => params(page).to).toBe("2024-12-31");
  // and the window as a whole, a month earlier (its slider; on a 21-year record a one-year frame is too narrow
  // to take between its two edge handles with a pointer: reported to the design system)
  const pan = page.getByRole("slider", { name: /^Window for / });
  await pan.focus();
  await pan.press("ArrowLeft");
  await expect.poll(() => params(page).from).not.toBe("2024-01-01");
  const { from, to } = params(page);
  const lo = at(from);
  const hi = at(to);
  expect(hi - lo).toBeGreaterThanOrEqual(30 * DAY);

  // the table holds exactly the window's samples of the selected variable, exactly as published
  await tools(page);
  await page.getByTestId("disc-samples").getByRole("button").first().click();
  const table = page.getByTestId("plot-table");
  await expect(table).toBeVisible();
  const dates = await table.locator("tbody tr td:first-child").allTextContents();
  expect(dates.length).toBeGreaterThan(0);
  for (const d of dates) {
    const t = Date.parse(`${d.slice(0, 10)}T12:00:00Z`);
    expect(t).toBeGreaterThanOrEqual(lo - DAY);
    expect(t).toBeLessThanOrEqual(hi + DAY);
  }
  await expect(page.getByTestId("row-samples")).toContainText(`${dates.length} seriata group sample`);

  // QC and provenance, as rows that open in place
  await page.getByRole("button", { name: /^Checks/ }).click();
  await expect(page.getByTestId("research-qc").locator("li")).not.toHaveCount(0);
  await page.getByRole("button", { name: /^Source and method/ }).click();
  await expect(page.getByTestId("research-request-url")).toHaveAttribute("href", /erddap\.sccoos\.org\/erddap\/tabledap\/HABs-MontereyWharf\.csv/);

  // export: exactly the window, with source columns, counted by variable
  await page.getByRole("button", { name: /^Export and cite/ }).click();
  await expect(page.getByTestId("research-export-what")).toContainText(`seriata group ${dates.length}`);
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export the window as CSV" }).click()]);
  const csv = readFileSync((await dl.path())!, "utf8");
  const lines = csv.split("\n").filter((l) => l && !l.startsWith("#"));
  const head = lines[0].split(",");
  expect(head).toEqual(expect.arrayContaining(["sample_time_utc", "variable", "value", "qualifier", "station_id", "source", "source_dataset", "source_url"]));
  const rows = lines.slice(1).map((l) => l.split(","));
  expect(rows.length).toBeGreaterThan(0);
  const ti = head.indexOf("sample_time_utc");
  const vi = head.indexOf("variable");
  for (const r of rows) {
    const t = Date.parse(r[ti]);
    expect(t).toBeGreaterThanOrEqual(lo - DAY);
    expect(t).toBeLessThanOrEqual(hi + DAY);
    expect(r[head.indexOf("station_id")]).toBe("HABs-MontereyWharf");
  }
  expect(rows.filter((r) => r[vi] === "pn_seriata")).toHaveLength(dates.length);
  const w = page.getByTestId("research-window-ms");
  const wf = Number(await w.getAttribute("data-from"));
  const wt = Number(await w.getAttribute("data-to"));
  expect(rows.every((r) => Date.parse(r[ti]) >= wf && Date.parse(r[ti]) <= wt)).toBe(true);

  const video = page.video();
  await ctx.close();
  if (video) copyFileSync(await video.path(), path.join(MEDIA, "journey-research-1440.webm"));
});

test("v11 links still open: place, variable, start/end; the page names the nearest station and its distance", async ({ page }) => {
  await open(page, "place=port-592&variable=particulate_domoic&start=2020-01-01&end=2020-12-31");
  await expect(page.getByTestId("research-arrived")).toContainText("Nearest station to Moss Landing");
  await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("research-window")).toContainText("Jan 1–Dec 31, 2020");
  // the first choice rewrites the address in the page's terms, keeping the place
  await page.getByTestId("range-1y").click();
  await expect.poll(() => params(page)).toMatchObject({ place: "port-592", station: "HABs-MontereyWharf", var: "pDA" });
});

test("a quantity the station does not measure is said, not silently swapped", async ({ page }) => {
  await open(page, "station=HABs-TrinidadPier&var=pn_seriata");
  await expect(page.getByTestId("research-arrived")).toContainText("isn’t measured at Trinidad Pier");
});

test("a station picked on the chart, in the picker or in the list opens its record and the chart moves to it", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&from=2024-01-01&to=2026-01-01&var=pDA");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  const scene = () => page.evaluate(() => (window as unknown as { __nxChart?: { scene: string | null } }).__nxChart?.scene ?? null);
  const before = await scene();
  await page.getByTestId("chart-station-HABs-MontereyWharf").click({ force: true });
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await expect.poll(scene).not.toBe(before);
  // the window and the variable travel with the reader to the new station
  expect(params(page)).toMatchObject({ station: "HABs-MontereyWharf", var: "pDA", from: "2024-01-01", to: "2026-01-01" });
  await tools(page);
  await page.getByTestId("research-station-select").selectOption("HABs-StearnsWharf");
  await expect(page.getByTestId("research-title")).toHaveText("Stearns Wharf (Santa Barbara)");
  await page.getByTestId("disc-station").getByRole("button").first().click();
  // the list dates each station by the selected variable, not by its newest visit
  await expect(page.getByTestId("list-latest-HABs-MontereyWharf")).toHaveText("Aug 31, 2022");
  await page.getByTestId("list-station-HABs-ScrippsPier").click();
  await expect(page.getByTestId("research-title")).toHaveText("Scripps Pier (La Jolla)");
  // a station with no model history has no model panel
  await expect(page.getByTestId("strip-model")).toHaveCount(0);
});

test("on a phone the record reads in the sheet under the chart strip, and the strip frames the station with its neighbours", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  // the page's element is the same one from the server's HTML through hydration and the phone query resolving
  const margin = page.getByTestId("nx-margin");
  await margin.evaluate((el) => ((el as HTMLElement & { __kept?: boolean }).__kept = true));
  await page.waitForTimeout(1500);
  expect(await margin.evaluate((el) => (el as HTMLElement & { __kept?: boolean }).__kept === true)).toBe(true);
  await expect(page.getByTestId("nx-research")).toHaveAttribute("data-layout", "stage");
  // the resting sheet leaves at least 40 % of the screen to the chart (design issue 2); the title starts the sheet
  await expect(async () => {
    const sheet = (await margin.boundingBox())!;
    expect(sheet.y).toBeGreaterThanOrEqual(844 * 0.4 - 2);
    const title = (await page.getByTestId("research-title").boundingBox())!;
    expect(title.y).toBeGreaterThanOrEqual(sheet.y);
    expect(title.y + title.height).toBeLessThanOrEqual(844 - 62);
    const station = (await page.getByTestId("chart-station-HABs-SantaCruzWharf").boundingBox())!;
    expect(station.y + station.height).toBeLessThanOrEqual(sheet.y);
  }).toPass({ timeout: 10_000 });
  // in reading order: the tabs, the sample, the plot, the overview of the record, the sample's details, the tools
  const y = async (id: string) => (await page.getByTestId(id).boundingBox())!.y;
  const order = ["rail-pDA", "inspector-position", "research-plot", "research-brush", "inspector-ratio", "research-tools"];
  const ys = [];
  for (const id of order) ys.push(await y(id));
  expect([...ys].sort((a, b) => a - b)).toEqual(ys);
  const plot = (await page.getByTestId("research-plot").boundingBox())!;
  expect(plot.width).toBeGreaterThanOrEqual(340);
  expect(plot.height).toBeGreaterThanOrEqual(240);
  // the compact selector (in the sheet's tools) changes the variable
  await page.getByTestId("research-variable-select").selectOption("temp");
  await expect.poll(() => params(page).var).toBe("temp");
  await page.getByTestId("research-variable-select").selectOption("pDA");
  await expect.poll(() => params(page).var).toBe("pDA");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await ctx.close();
});
