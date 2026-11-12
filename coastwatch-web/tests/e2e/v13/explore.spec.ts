import { expect, test, type Page } from "@playwright/test";
import { M2, OK } from "../ports";

/**
 * Explore's v13 inspection, time and hand-offs (audit-C acceptance tests 9–24), on the fixture. Each of these
 * failed on the v12 baseline.
 */
test.use({ storageState: { cookies: [], origins: [] } });
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-tour-v1", "dismissed");
  });
});

type Cell = { lat: number; lon: number; widthKm: number; heightKm: number; outline: [number, number][] };
type Hook = { product: string; frame: string; field?: string; cell?: Cell | null; flow?: { segments: () => number; streaks: () => { speed: number; px: number }[]; capPx: () => number; heads: () => [number, number][] } };
const hook = <T>(page: Page, f: (h: Hook) => T) => page.evaluate(`(${f.toString()})(window.__nxExplore ?? {})`) as Promise<T>;
const mapReady = (page: Page) => expect(page.getByTestId("nx-sheet")).toHaveAttribute("data-map-ready", "1", { timeout: 20_000 });
const fieldDrawn = (page: Page, frame: string) => expect.poll(() => hook(page, (h) => h.field ?? null), { timeout: 15_000 }).toBe(frame);

async function project(page: Page, lat: number, lon: number) {
  const p = (await page.evaluate(`window.__cwMap.project([${lon}, ${lat}])`)) as { x: number; y: number };
  const box = (await page.getByTestId("nx-map").boundingBox())!;
  return { x: box.x + p.x, y: box.y + p.y };
}
async function tapWater(page: Page, lat: number, lon: number) {
  const p = await project(page, lat, lon);
  await page.mouse.click(p.x, p.y);
}
const readout = (page: Page) => page.getByTestId("nx-readout");

test("C-HARM reads a whole percent with one sign, titled as one cell of the model, dated by its run", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=charm`);
  await mapReady(page);
  await fieldDrawn(page, "charm:3");
  await tapWater(page, 36.6, -122.4);
  await expect(readout(page)).toContainText("C-HARM in this 3 km cell");
  const figure = (await readout(page).locator("p").filter({ hasText: "%" }).first().innerText()).replace(/\s+/g, " ").trim();
  expect(figure).toMatch(/^\d{1,3} ?%$/);
  expect(await readout(page).innerText()).not.toContain("%%");
  await expect(readout(page)).toContainText("Model · C-HARM v3.1, run of Oct 8 · valid Oct 10");
});

test("water outside a model's cells is ruled 'outside the model', keyed in the legend, and its reason is where the gap is", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=charm`);
  await mapReady(page);
  await fieldDrawn(page, "charm:3");
  await expect(page.getByTestId("nx-key-outside")).toContainText("Ruled: outside the model, no value");
  // a point in C-HARM's nearshore gap off Santa Cruz: the layer draws there (it is not the bare chart)
  const p = await project(page, 36.935, -121.95);
  const clip = { x: Math.round(p.x) - 12, y: Math.round(p.y) - 12, width: 24, height: 24 };
  await page.waitForTimeout(800);
  const withLayer = await page.screenshot({ clip });
  await page.evaluate(`(() => { const m = window.__cwMap; m.removeLayer("nx-field"); m.triggerRepaint(); })()`);
  await page.waitForTimeout(800);
  const bare = await page.screenshot({ clip });
  expect(withLayer.equals(bare)).toBe(false);
  await page.reload();
  await mapReady(page);
  await fieldDrawn(page, "charm:3");
  await tapWater(page, 36.935, -121.95);
  await expect(readout(page)).toContainText("Between C-HARM's cells and the shore: the model leaves out the nearshore band and the bays. No value is not a low value.");
  // the outlook: also keyed, and its readout is dated by its window and issue, never its last day
  await page.getByTestId("nx-layer-outlook").click();
  await expect(page.getByTestId("nx-key-outside")).toBeVisible();
  await expect(readout(page)).toContainText("Outlook · Neritic outlook, mean of Oct 13–20 · issued Oct 12");
  await expect(readout(page)).not.toContainText("Outlook · Neritic outlook · Oct 20");
});

test("a 'nearby' record is within 25 km and measures the variable; otherwise the words say what is known", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=charm`);
  await mapReady(page);
  await fieldDrawn(page, "charm:3");
  // 40 km offshore: no pier within 25 km measures particulate domoic acid
  await tapWater(page, 36.6, -122.4);
  await expect(readout(page)).toContainText("No station within 25 km measures this.");
  await expect(page.getByTestId("nx-record-link")).toHaveCount(0);
  await expect(page.getByTestId("nx-record-far")).toContainText(/km away$/);
  // in the bay: a pier that has published it, named with its distance
  await tapWater(page, 36.85, -122.0);
  const near = page.getByTestId("nx-record-link");
  await expect(near).toContainText(/^See the record nearby: .+, \d+(\.\d)? km$/);
  expect(Number(/, ([\d.]+) km$/.exec(await near.innerText())![1])).toBeLessThanOrEqual(25);
  await expect(near).toHaveAttribute("href", /\/research\?station=[^&]+&from=\d{4}-\d{2}-\d{2}&to=\d{4}-\d{2}-\d{2}&var=pDA&place=port-592$/);
  // chlorophyll: the station list does not carry it, so no pier is claimed to measure it
  await page.getByTestId("nx-layer-chl").click();
  await fieldDrawn(page, "olci:latest");
  await tapWater(page, 36.87, -122.02);
  await expect(page.getByTestId("nx-record-unknown")).toContainText("station list does not say which piers measure this");
  await expect(page.getByTestId("nx-record-link")).toHaveCount(0);
  await expect(page.getByTestId("nx-record-look")).toHaveAttribute("href", /var=chl_extracted&place=port-592$/);
});

test("the reading is the cell that was read: its centre in the header, its outline on the chart, the layer kept in sight", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=charm`);
  await mapReady(page);
  await fieldDrawn(page, "charm:3");
  await tapWater(page, 36.705, -122.213);
  await expect(readout(page)).toContainText("C-HARM in this 3 km cell");
  const cell = (await hook(page, (h) => h.cell ?? null))!;
  expect(cell).not.toBeNull();
  // the header's coordinate is the cell's centre, not the tap
  const min = (v: number) => `${Math.floor(Math.abs(v))}°${((Math.abs(v) % 1) * 60).toFixed(1)}′`;
  await expect(readout(page)).toContainText(min(cell.lat));
  await expect(readout(page)).toContainText(min(cell.lon));
  expect(Math.abs(cell.lat - 36.705) + Math.abs(cell.lon + 122.213)).toBeGreaterThan(0);
  // the outline is the cell's own footprint, within a pixel
  const outline = page.getByTestId("nx-cell-outline");
  await expect(outline).toHaveAttribute("visibility", "visible");
  const box = (await outline.boundingBox())!;
  const corners = await Promise.all(cell.outline.slice(0, 4).map(([lon, lat]) => project(page, lat, lon)));
  const xs = corners.map((c) => c.x), ys = corners.map((c) => c.y);
  // the path's box includes half its stroke (1.75 px)
  expect(Math.abs(box.x - Math.min(...xs))).toBeLessThanOrEqual(2);
  expect(Math.abs(box.y - Math.min(...ys))).toBeLessThanOrEqual(2);
  expect(Math.abs(box.width - (Math.max(...xs) - Math.min(...xs)))).toBeLessThanOrEqual(3);
  expect(Math.abs(box.height - (Math.max(...ys) - Math.min(...ys)))).toBeLessThanOrEqual(3);
  // a second tap inside the same cell reads the same cell; one in the next cell reads another
  await tapWater(page, cell.lat + 0.004, cell.lon + 0.004);
  await expect.poll(() => hook(page, (h) => h.cell?.lat ?? null)).toBe(cell.lat);
  await tapWater(page, cell.lat + 0.05, cell.lon);
  await expect.poll(() => hook(page, (h) => h.cell?.lat ?? null)).not.toBe(cell.lat);
  // the layer, its unit and its date stay in the margin's view above the reading
  const ctx = page.getByTestId("nx-context");
  await expect(ctx).toContainText("C-HARM · % · model, past run, valid Oct 10");
  await readout(page).getByTestId("nx-coast-link").scrollIntoViewIfNeeded();
  await expect(ctx).toBeInViewport();
});

test("a reading moves into My Coast with the port, the cell and its date", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  await fieldDrawn(page, "olci:latest");
  await tapWater(page, 36.87, -122.02);
  const link = page.getByTestId("nx-coast-link");
  await expect(link).toHaveAttribute("href", /^\/coast\?place=port-592&at=36\.\d{4},-122\.\d{4}&date=2026-10-\d\d$/);
  const cell = (await hook(page, (h) => h.cell ?? null))!;
  expect(await link.getAttribute("href")).toContain(`at=${cell.lat.toFixed(4)},${cell.lon.toFixed(4)}`);
  await link.click();
  await expect(page).toHaveURL(/\/coast\?place=port-592&at=36\.\d{4},-122\.\d{4}&date=2026-10-\d\d$/);
  await expect(page.getByTestId("nx-app")).toBeVisible();
});

test("playing the days with a reading open never flashes 'Reading…'; the count says which day is on show", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&at=36.8700,-122.0200`);
  await mapReady(page);
  await expect(readout(page)).toContainText("Chlorophyll in this 300 m cell");
  await page.evaluate(() => {
    const w = window as unknown as { __flashes: number; __dims: number };
    w.__flashes = 0;
    w.__dims = 0;
    const el = document.querySelector("[data-testid=nx-readout]")!;
    new MutationObserver(() => {
      if (/Reading /.test(el.textContent ?? "")) w.__flashes++;
      if (el.getAttribute("data-updating") === "1") w.__dims++;
    }).observe(el, { childList: true, subtree: true, characterData: true, attributes: true });
  });
  await page.getByTestId("nx-play").click();
  const seen = new Set<string>();
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(200);
    seen.add((await page.getByTestId("nx-explore").getAttribute("data-frame"))!);
  }
  expect(seen.size).toBeGreaterThanOrEqual(3);
  await expect(page.getByTestId("nx-frame-count")).toContainText(/day \d of 4/);
  await page.getByTestId("nx-play").click();
  expect(await page.evaluate(() => (window as unknown as { __flashes: number }).__flashes)).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { __dims: number }).__dims)).toBeGreaterThan(0);
  // the latest view mixes days: it is not a step of the sequence and has no count
  await page.getByTestId("nx-frame-olci:latest").click();
  await expect(page.getByTestId("nx-frame-count")).toHaveCount(0);
});

test("the time rail runs along the foot of the chart at desktop, and reads in the sheet on a phone", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  const rail = page.getByTestId("nx-timerail");
  await expect(rail).toBeVisible();
  const r = (await rail.boundingBox())!, m = (await page.getByTestId("nx-map").boundingBox())!, margin = (await page.getByTestId("nx-margin").boundingBox())!;
  expect(r.x).toBeGreaterThanOrEqual(margin.x + margin.width - 1);
  expect(r.y).toBeGreaterThanOrEqual(m.y + m.height - 1);
  expect(r.width).toBeGreaterThan(700);
  expect(await page.getByTestId("nx-margin").getByTestId("nx-timerail").count()).toBe(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId("nx-margin").getByTestId("nx-timerail")).toHaveCount(1);
});

test("a radar hour that cannot be read is said so: no streaks are promised", async ({ page }) => {
  await page.route("**/currents/hfr2km/**", (route) => route.abort());
  await page.goto(`${OK}/explore?place=port-592&layer=sst&cur=1`);
  await mapReady(page);
  const toggle = page.getByTestId("nx-currents-toggle");
  await expect(toggle).toContainText("unavailable: this hour could not be read", { timeout: 15_000 });
  await expect(page.getByTestId("nx-currents-key")).toHaveCount(0);
  expect(await hook(page, (h) => h.flow?.segments() ?? 0)).toBe(0);
});

test("streaks are at most about one published cell long, and every head is in the radar's own field", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=sst&cur=1`);
  await mapReady(page);
  await expect.poll(() => hook(page, (h) => h.flow?.segments() ?? 0), { timeout: 15_000 }).toBeGreaterThan(0);
  for (const zoom of [8.5, 11]) {
    await page.evaluate(`window.__cwMap.jumpTo({ center: [-122.0, 36.8], zoom: ${zoom} })`);
    await page.waitForTimeout(1200);
    const r = await hook(page, (h) => ({ cap: h.flow!.capPx(), streaks: h.flow!.streaks() }));
    expect(r.streaks.length).toBeGreaterThan(5);
    // v13 lead decision 7 changed what a streak's length means: it no longer grows with speed past one cell
    // (a trail across several cells read as a path the water would take); speed is carried by weight and ink
    for (const s of r.streaks) expect(s.px).toBeLessThanOrEqual(r.cap * 1.01);
    const cellPx = (0.018 / 360) * 512 * 2 ** zoom;
    expect(r.cap).toBeGreaterThanOrEqual(Math.min(cellPx, 10) - 0.5);
    expect(r.cap).toBeLessThanOrEqual(Math.max(cellPx * 1.25, 10.5));
  }
  await expect(page.getByTestId("nx-currents-key")).toContainText(/The radar measured \d+ % of the coastal ocean this hour\./);
});

test("a layer this data run does not hold stays in the switch, disabled, with the reason", async ({ page }) => {
  // the compat dataset publishes no derived index: no temperature, no outlook
  await page.goto(`${M2}/explore`);
  await mapReady(page);
  for (const id of ["sst", "outlook"]) {
    const b = page.getByTestId(`nx-layer-${id}`);
    await expect(b).toBeDisabled();
    await expect(b).toContainText("not in this data run");
  }
  // what the run does hold is still offered
  expect(await page.locator('[data-testid^="nx-layer-"]:not([disabled])').count()).toBeGreaterThanOrEqual(1);
});

test("the selection survives a reload, and coming Back from Research", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592&layer=charm&frame=charm:1`);
  await mapReady(page);
  await fieldDrawn(page, "charm:1");
  await tapWater(page, 36.85, -122.0);
  await expect(readout(page)).toContainText("C-HARM in this 3 km cell");
  await expect(page).toHaveURL(/layer=charm/);
  await expect(page).toHaveURL(/frame=charm(:|%3A)1/);
  await expect(page).toHaveURL(/at=36\.85/);
  const url = page.url();
  const value = await readout(page).locator("p").filter({ hasText: "%" }).first().innerText();
  await page.reload();
  await mapReady(page);
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", "charm:1");
  await expect(readout(page).locator("p").filter({ hasText: "%" }).first()).toHaveText(value);
  await page.getByTestId("nx-record-link").click();
  await expect(page).toHaveURL(/\/research\?station=/);
  await page.goBack();
  await expect(page).toHaveURL(url);
  await expect(page.getByTestId("nx-explore")).toHaveAttribute("data-frame", "charm:1");
  await expect(page.getByTestId("nx-legend")).toHaveAttribute("data-product", "charm");
  await expect(readout(page).locator("p").filter({ hasText: "%" }).first()).toHaveText(value);
});

for (const size of [{ width: 390, height: 844 }, { width: 320, height: 640 }]) {
  test(`at ${size.width} px a reading peeks over the chart, its cell stays above the sheet, and a second tap reads another cell`, async ({ page }) => {
    await page.setViewportSize(size);
    await page.goto(`${OK}/explore?place=port-592&layer=charm`);
    await mapReady(page);
    await fieldDrawn(page, "charm:3");
    const sheet = page.getByTestId("nx-margin");
    const tapVisible = async (fx: number, fy: number) => {
      // a point of the chart that is in view above the sheet right now
      const m = (await page.getByTestId("nx-map").boundingBox())!;
      const top = (await sheet.boundingBox())!.y;
      await page.mouse.click(m.x + m.width * fx, m.y + (Math.min(top, m.y + m.height) - m.y) * fy);
    };
    await tapVisible(0.35, 0.55);
    await expect(readout(page)).toContainText(/C-HARM in this 3 km cell|C-HARM here/);
    await expect(sheet).toHaveAttribute("data-detent", "peek");
    await page.waitForTimeout(1400);
    const first = (await hook(page, (h) => h.cell ?? null))!;
    expect(first).not.toBeNull();
    // the reading leads the sheet: its figure is on screen at the peek
    await expect(readout(page).locator("header").first()).toBeInViewport();
    // the read cell is on the chart above the sheet, not under it
    const p = await project(page, first.lat, first.lon);
    const top = (await sheet.boundingBox())!.y;
    expect(p.y).toBeLessThan(top - 8);
    expect(p.y).toBeGreaterThan(0);
    expect(p.x).toBeGreaterThan(0);
    expect(p.x).toBeLessThan(size.width);
    // a second tap on water still in view reads another cell
    await tapVisible(0.2, 0.3);
    await expect.poll(() => hook(page, (h) => (h.cell ? `${h.cell.lat},${h.cell.lon}` : null)), { timeout: 10_000 }).not.toBe(`${first.lat},${first.lon}`);
    await page.waitForTimeout(1200);
    const second = (await hook(page, (h) => h.cell ?? null))!;
    const q = await project(page, second.lat, second.lon);
    expect(q.y).toBeLessThan((await sheet.boundingBox())!.y - 8);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  });
}

test("no name on the chart is set over the transect line, the read cell or the phone's locator inset", async ({ page }) => {
  const overlaps = (a: { x: number; y: number; width: number; height: number }, b: { x: number; y: number; width: number; height: number }) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
  const names = () =>
    page.locator(".n12-mark").evaluateAll((els) =>
      els
        .filter((e) => getComputedStyle(e).visibility !== "hidden" && (e.textContent ?? "").trim().length > 0)
        .map((e) => {
          const r = e.getBoundingClientRect();
          return { text: (e.textContent ?? "").trim(), x: r.x, y: r.y, width: r.width, height: r.height };
        }),
    );
  // desktop: a transect along Moss Landing's latitude runs through where the port's name stood
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  await fieldDrawn(page, "olci:latest");
  await tapWater(page, 36.8, -121.95);
  await expect(page.getByTestId("nx-transect")).toBeVisible();
  await page.waitForTimeout(1500);
  const line = (await page.getByTestId("nx-reading-on-chart").locator("path").first().boundingBox())!;
  const band = { x: line.x, y: line.y - 2, width: line.width, height: line.height + 4 };
  const shown = await names();
  expect(shown.length).toBeGreaterThan(1);
  // (the reader's own port symbol may touch the line's end: the port is on that latitude; its letters may not)
  for (const n of shown) expect(overlaps({ ...n, x: n.x + 26, width: Math.max(0, n.width - 26) }, band), `"${n.text}" over the transect`).toBe(false);
  // phone: the locator inset
  await page.setViewportSize({ width: 320, height: 640 });
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  await page.waitForTimeout(1500);
  const inset = (await page.getByTestId("nx-locator").boundingBox())!;
  for (const n of await names()) expect(overlaps(n, inset), `"${n.text}" under the locator`).toBe(false);
});

test("the chart's own buttons: zoom, back to this view, fit the data", async ({ page }) => {
  await page.goto(`${OK}/explore?place=port-592`);
  await mapReady(page);
  const zoom = () => page.evaluate(`window.__cwMap.getZoom()`) as Promise<number>;
  const controls = page.getByTestId("nx-chart-controls");
  await expect(controls).toBeVisible();
  const z0 = await zoom();
  await controls.getByRole("button").first().click();
  await expect.poll(zoom).not.toBeCloseTo(z0, 1);
  await page.getByTestId("nx-fit-data").click();
  await page.waitForTimeout(1500);
  const b = (await page.evaluate(`(() => { const b = window.__cwMap.getBounds(); return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]; })()`)) as number[];
  // the whole of the observed chlorophyll's grid is in view
  expect(b[0]).toBeLessThan(-122.5);
  expect(b[3] - b[1]).toBeGreaterThan(0.7);
});
