import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Journey 3 (docs/coastwatch/v12/design-system/journeys.md): a researcher arrives from Explore with a station,
 * a window and a quantity in the address, brushes the record, reads the filtered log and exports exactly the
 * window. The journey is recorded as video into docs/coastwatch/v12/implementation/research/.
 */
// the recording goes to the docs only on request (NX_MEDIA=1); an ordinary run keeps it with its own results
const MEDIA = process.env.NX_MEDIA ? path.resolve(__dirname, "../../../../docs/coastwatch/v12/implementation/research") : path.resolve(__dirname, "../../../test-results/journey-media/research");
test.use({ storageState: { cookies: [], origins: [] } });

const params = (page: Page) => Object.fromEntries(new URL(page.url()).searchParams);
const DAY = 86_400_000;
const at = (d: string) => Date.parse(`${d}T12:00:00Z`);

async function open(page: Page, q: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}/research?${q}`);
  await expect(page.getByTestId("research-strips")).toBeVisible();
}

test("arrive from Explore, brush, read the filtered log, export exactly the window", async ({ browser }) => {
  mkdirSync(MEDIA, { recursive: true });
  const dir = path.join(test.info().outputDir, "video");
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, recordVideo: { dir, size: { width: 1440, height: 900 } }, acceptDownloads: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-MontereyWharf&from=2024-01-01&to=2025-12-31&var=pn_seriata");

  // the Explore selection survives into Research through the address
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await expect(page.getByTestId("strip-pn_seriata")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("research-window")).toContainText("Jan 1, 2024");
  expect(params(page)).toMatchObject({ station: "HABs-MontereyWharf", var: "pn_seriata", from: "2024-01-01", to: "2025-12-31" });
  // every strip of the station on one axis, the model strip labelled as model
  for (const id of ["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]) await expect(page.getByTestId(`strip-${id}`)).toBeVisible();
  await expect(page.getByTestId("strip-model")).toContainText("Model");
  await page.waitForTimeout(1800); // the strips ink in

  // the crosshair reads one visit on every strip
  const plot = page.getByTestId("research-strips").locator("[role=group]");
  const box = (await plot.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.45, box.y + 60, { steps: 12 });
  await expect(page.getByTestId("research-crosshair")).toBeVisible();
  await page.waitForTimeout(600);

  // brush: the window's end, a year earlier with Shift+Arrow; the address follows
  const end = page.getByRole("slider", { name: "Window end" });
  await end.focus();
  await end.press("Shift+ArrowLeft");
  await expect.poll(() => params(page).to).toBe("2024-12-31");
  await page.waitForTimeout(900);
  // and dragged by hand: the window's start, later
  const start = page.getByRole("slider", { name: "Window start" });
  const sb = (await start.boundingBox())!;
  await page.mouse.move(sb.x + sb.width / 2, sb.y + sb.height / 2);
  await page.mouse.down();
  await page.mouse.move(sb.x + sb.width / 2 + 40, sb.y + sb.height / 2, { steps: 10 });
  await page.mouse.up();
  await expect.poll(() => params(page).from).not.toBe("2024-01-01");
  const { from, to } = params(page);
  const lo = at(from);
  const hi = at(to);
  expect(hi - lo).toBeGreaterThanOrEqual(30 * DAY);

  // the dated log holds exactly the window's samples of the quantity in focus
  await page.getByRole("button", { name: /^Samples/ }).click();
  const log = page.getByTestId("research-log");
  await expect(log).toBeVisible();
  const dates = await log.locator("tbody th").allTextContents();
  expect(dates.length).toBeGreaterThan(0);
  for (const d of dates) {
    const t = Date.parse(`${d} 12:00 UTC`);
    expect(t).toBeGreaterThanOrEqual(lo - DAY);
    expect(t).toBeLessThanOrEqual(hi + DAY);
  }
  await expect(page.getByTestId("row-samples")).toContainText(`${dates.length} dated`);
  // sortable: by value, a reported 0 never above a measured value
  await page.getByTestId("log-sort-value").click();
  await expect(log.locator("thead th").nth(1)).toHaveAttribute("aria-sort", "descending");
  await page.waitForTimeout(700);

  // QC and provenance, as rows that open in place
  await page.getByRole("button", { name: /^Checks/ }).click();
  await expect(page.getByTestId("research-qc").locator("li")).not.toHaveCount(0);
  await page.getByRole("button", { name: /^Source/ }).click();
  await expect(page.getByTestId("research-request-url")).toHaveAttribute("href", /erddap\.sccoos\.org\/erddap\/tabledap\/HABs-MontereyWharf\.csv/);
  await page.waitForTimeout(600);

  // export: exactly the window, with source columns
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
  // the quantity in focus: the same samples the log lists
  expect(rows.filter((r) => r[vi] === "pn_seriata")).toHaveLength(dates.length);
  // and the window the page holds is the file's
  const w = page.getByTestId("research-window-ms");
  const wf = Number(await w.getAttribute("data-from"));
  const wt = Number(await w.getAttribute("data-to"));
  expect(rows.every((r) => Date.parse(r[ti]) >= wf && Date.parse(r[ti]) <= wt)).toBe(true);
  await page.waitForTimeout(800);

  const video = page.video();
  await ctx.close();
  if (video) copyFileSync(await video.path(), path.join(MEDIA, "journey-research-1440.webm"));
});

test("v11 links still open: place, variable, start/end; the page names the nearest station and its distance", async ({ page }) => {
  await open(page, "place=port-592&variable=particulate_domoic&start=2020-01-01&end=2020-12-31");
  await expect(page.getByTestId("research-arrived")).toContainText("Nearest station to Moss Landing");
  await expect(page.getByTestId("strip-pDA")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("research-window")).toContainText("Jan 1–Dec 31, 2020");
  // the first choice rewrites the address in v12's terms, keeping the place
  await page.getByRole("radio", { name: "1 y" }).click();
  await expect.poll(() => params(page)).toMatchObject({ place: "port-592", station: "HABs-MontereyWharf", var: "pDA" });
});

test("a quantity the station does not measure is said, not silently swapped", async ({ page }) => {
  await open(page, "station=HABs-TrinidadPier&var=pn_seriata");
  await expect(page.getByTestId("research-arrived")).toContainText("isn’t measured at Trinidad Pier");
});

test("a station picked on the chart or in the list opens its record and the chart moves to it", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&from=2024-01-01&to=2026-01-01&var=pDA");
  await expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
  const scene = () => page.evaluate(() => (window as unknown as { __nxChart?: { scene: string | null } }).__nxChart?.scene ?? null);
  const before = await scene();
  await page.getByTestId("chart-station-HABs-MontereyWharf").click({ force: true });
  await expect(page.getByTestId("research-title")).toHaveText("Monterey Wharf");
  await expect.poll(scene).not.toBe(before);
  // the window travels with the reader to the new station
  expect(params(page)).toMatchObject({ station: "HABs-MontereyWharf", from: "2024-01-01", to: "2026-01-01" });
  await page.getByRole("button", { name: /^Station/ }).click();
  await page.getByTestId("list-station-HABs-ScrippsPier").click();
  await expect(page.getByTestId("research-title")).toHaveText("Scripps Pier (La Jolla)");
  // a station with no model history has no model strip and says so in its row
  await expect(page.getByTestId("strip-model")).toHaveCount(0);
});

test("on a phone the record reads in the page under the chart strip, the brush above the strips", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-SantaCruzWharf&var=pDA");
  // v13: Research is a desk page. On a phone the page is the page (no sheet over the chart), under a chart strip
  const margin = page.getByTestId("nx-margin");
  // the page's element is the same one from the server's HTML through hydration and the phone query resolving
  // (it used to be replaced about a second after load, which dropped the brush from under this test)
  await margin.evaluate((el) => ((el as HTMLElement & { __kept?: boolean }).__kept = true));
  await page.waitForTimeout(1500);
  expect(await margin.evaluate((el) => (el as HTMLElement & { __kept?: boolean }).__kept === true)).toBe(true);
  const brush = (await page.getByTestId("research-brush").boundingBox())!;
  const strips = (await page.getByTestId("research-strips").boundingBox())!;
  expect(brush.y).toBeLessThan(strips.y);
  await expect(page.getByTestId("nx-research")).toHaveAttribute("data-layout", "desk");
  await expect(page.getByRole("button", { name: "Show more of this page" })).toBeHidden();
  // the chart is a strip above the page, and the page starts under it
  const pane = (await page.getByTestId("nx-pane").boundingBox())!;
  expect(pane.height).toBeGreaterThan(200);
  expect(pane.height).toBeLessThan(844 * 0.45);
  expect((await page.getByTestId("research-title").boundingBox())!.y).toBeGreaterThanOrEqual(pane.y + pane.height - 1);
  // scrolled, the strip leaves a bar with the chart's control and the record takes the screen
  await page.getByTestId("research-strips").scrollIntoViewIfNeeded();
  await page.getByTestId("nx-work").evaluate((el) => el.scrollBy(0, 400));
  await expect(page.getByTestId("nx-pane-toggle")).toBeInViewport();
  expect((await page.getByTestId("nx-pane").boundingBox())!.y).toBeLessThan(0);
  await ctx.close();
});
