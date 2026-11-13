import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Research acceptance (v13 brief, Agent A; journey 4), on the v14 synthesis layout: the chart with the station and its
 * neighbours, the record on a pass along the chart's foot (the plot, its range and variable controls, the overview, the
 * legend and the model switch) and the sample on the pass's stub. Monterey Wharf → pDA → the latest real observation →
 * source and QC → change the range → model beside measured without conflation. The fixture is CalHABMAP as recorded:
 * Monterey Wharf has no pDA after 2022-08-31 and its recent visits record temperature only.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const params = (page: Page) => Object.fromEntries(new URL(page.url()).searchParams);
const MW_1Y = "station=HABs-MontereyWharf&var=pDA&range=1y";

async function open(page: Page, q: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}/research?${q}`);
  await expect(page.getByTestId("research-record")).toBeVisible();
}

/** The record and its tools (the table, exports, sources, the station and comparison pickers): a panel over the pass on a wide screen. */
async function tools(page: Page, on = true) {
  const b = page.getByTestId("research-tools-open");
  if ((await b.getAttribute("aria-expanded")) !== String(on)) await b.click();
  if (on) await expect(page.getByTestId("research-tools")).toBeVisible();
  else await expect(page.getByTestId("research-tools")).toHaveCount(0);
}

async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

/** The sample card on the stub is whole in the first screen, beside the plot, and never needs scrolling (design issue 3). */
async function cardWhole(page: Page, h = 900) {
  await expect(page.getByTestId("inspector-value")).toBeInViewport({ ratio: 1 });
  const card = (await page.getByTestId("research-inspector").boundingBox())!;
  const plot = (await page.getByTestId("research-plot").boundingBox())!;
  expect(card.y).toBeGreaterThanOrEqual(0);
  expect(card.y + card.height).toBeLessThanOrEqual(h);
  expect(card.x).toBeGreaterThanOrEqual(plot.x + plot.width);
  const scroll = await page.getByTestId("research-inspector").evaluate((el) => {
    const stub = el.parentElement!;
    return [stub.scrollHeight, stub.clientHeight];
  });
  expect(scroll[0]).toBeLessThanOrEqual(scroll[1] + 1);
  await expect(page.getByTestId("research-open-record")).toBeInViewport({ ratio: 1 });
  await expect(page.getByTestId("inspector-source-row")).toBeInViewport({ ratio: 1 });
}

test("Monterey Wharf, pDA, one year: the empty range is explained, and one press reaches the 2022-08-31 sample on the card", async ({ page }) => {
  await open(page, MW_1Y);
  // the freshness line is per variable: pDA's last sample, not the station's newest visit
  await expect(page.getByTestId("research-status")).toContainText("pDA last measured Aug 31, 2022 (4.1 years ago)");
  // no empty axis: the reason, what the visits did record, and that it is not a zero
  const empty = page.getByTestId("research-empty");
  await expect(empty).toContainText("Last measured Aug 31, 2022");
  await expect(empty).toContainText("0.052 ng/mL");
  await expect(empty).toContainText("recorded temperature only");
  await expect(empty).toContainText("Not measured is not zero");
  await expect(page.getByTestId("research-plot")).toHaveCount(0);
  // no sample is picked in an empty window
  await expect(page.getByTestId("research-inspector")).toHaveAttribute("data-empty", "");
  // the four actions
  await expect(page.getByTestId("empty-latest")).toContainText("Aug 31, 2022");
  await expect(page.getByTestId("empty-all")).toBeVisible();
  await expect(page.getByTestId("research-empty-neighbour")).toContainText("Santa Cruz Wharf, 41 km away");
  await expect(page.getByTestId("empty-source")).toHaveAttribute("href", "https://erddap.sccoos.org/erddap/tabledap/HABs-MontereyWharf.html");
  // the model is behind its switch; switched on, it is not drawn beside nothing measured, and says why, with its words
  await page.getByTestId("research-model-switch").click();
  await expect(page.getByTestId("research-model-switch")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("research-model-collapsed")).toContainText("is not drawn");
  await expect(page.getByTestId("research-plot-model")).toHaveCount(0);
  await expect(page.getByTestId("research-model-words")).toContainText("not toxin in seafood");
  // what the station measured and when, per variable, in the record's tools
  await tools(page);
  await expect(page.getByTestId("rail-last-pDA")).toContainText("Last Aug 31, 2022");
  await expect(page.getByTestId("rail-last-pDA")).toContainText("years ago");
  await expect(page.getByTestId("rail-count-pDA")).toHaveText("0 in window");
  await expect(page.getByTestId("rail-count-temp")).not.toHaveText("0 in window");
  await expect(page.getByTestId("rail-last-chl_extracted")).toContainText("Last Sep 11, 2024");
  await tools(page, false);

  // jump to the latest sample: the window holds it and the card reads it exactly, in the first screen
  await page.getByTestId("empty-latest").click();
  await expect.poll(() => params(page).day).toBe("2022-08-31");
  await expect(page.getByTestId("research-plot")).toBeVisible();
  await expect(page.getByTestId("research-plot").locator("[data-selection]")).toBeVisible();
  await cardWhole(page);
  await expect(page.getByTestId("inspector-value")).toHaveText("0.05220236 ng/mL");
  await expect(page.getByTestId("inspector-figure")).toContainText("0.052");
  await expect(page.getByTestId("inspector-utc")).toHaveText("2022-08-31 14:45 UTC");
  await expect(page.getByTestId("inspector-local")).toContainText("Aug 31, 2022 · 7:45 AM");
  await expect(page.getByTestId("inspector-ratio")).toHaveText("Below the model’s reference level");
  await expect(page.getByTestId("inspector-others")).toContainText("Water");
  await expect(page.getByTestId("inspector-source-row")).toHaveAttribute("href", /HABs-MontereyWharf\.htmlTable\?time,.*SampleID.*&time=2022-08-31T14%3A45%3A00Z$/);
  // the sample's record opens over the pass: depth, place, the sample id at the source, checks and the citation
  await page.getByTestId("research-open-record").click();
  const rec = page.getByTestId("research-sample-record");
  await expect(rec).toBeVisible();
  await expect(page.getByTestId("inspector-state")).toContainText("Measured, as published");
  await expect(rec).toContainText("5 m");
  await expect(rec).toContainText("Monterey Wharf (HAB_MWII)");
  await expect(page.getByTestId("inspector-sampleid")).toContainText("At the source");
  await expect(page.getByTestId("inspector-qc").locator("li")).not.toHaveCount(0);
  await expect(page.getByTestId("inspector-citation")).toContainText("sample of 2022-08-31 14:45 UTC");
  await page.getByTestId("research-record-close").click();
  await expect(rec).toHaveCount(0);
  // with a measured value in the window the model is drawn, on its own panel
  await expect(page.getByTestId("research-model-collapsed")).toHaveCount(0);
  await expect(page.getByTestId("research-plot-model")).toBeVisible();
  // Previous steps through pDA samples (not the visits between them); Next is at the record's end
  await expect(page.getByTestId("inspector-next")).toHaveAttribute("aria-disabled", "true");
  await page.getByTestId("inspector-prev").click();
  await expect(page.getByTestId("inspector-utc")).not.toHaveText("2022-08-31 14:45 UTC");
  await expect(page.getByTestId("research-inspector-live")).toContainText("Sample of 2022-");
});

test("Monterey Wharf opens on its last pDA sample (Aug 31, 2022), over the whole record", async ({ page }) => {
  await open(page, "station=HABs-MontereyWharf&var=pDA");
  await expect(page.getByTestId("range-all")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("inspector-utc")).toHaveText("2022-08-31 14:45 UTC");
  // the port pill names the station's area
  await expect(page.getByTestId("nx-port-control")).toContainText("Near Monterey");
  await expect(page.getByTestId("inspector-position")).toContainText(/Sample (\d[\d,]*) of \1/);
  await expect(page.getByTestId("research-plot").locator("[data-selection]")).toBeVisible();
});

test("1440 × 900, on load: the pass keeps to the chart's foot (at most 45 % of the height) with the plot, tabs and card whole; so is the empty range with its actions", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  const pass = (await page.getByTestId("research-record").boundingBox())!;
  expect(pass.height).toBeLessThanOrEqual(900 * 0.45 + 1);
  expect(pass.y + pass.height).toBeLessThanOrEqual(900);
  const plot = (await page.getByTestId("research-plot").boundingBox())!;
  expect(plot.width).toBeGreaterThanOrEqual(800);
  expect(plot.height).toBeGreaterThanOrEqual(170);
  expect(plot.y + plot.height).toBeLessThanOrEqual(pass.y + pass.height);
  await expect(page.getByTestId("rail-temp")).toBeInViewport({ ratio: 1 });
  await expect(page.getByTestId("range-3y")).toBeInViewport({ ratio: 1 });
  await cardWhole(page);
  // the station is drawn on the chart above the pass, with its neighbour
  const mark = (await page.getByTestId("chart-station-HABs-SantaCruzWharf").boundingBox())!;
  expect(mark.y + mark.height).toBeLessThan(pass.y);
  await expect(page.getByTestId("chart-station-HABs-MontereyWharf")).toBeVisible();
  // Monterey Wharf has one more variable: its plot still ends inside the pass
  await open(page, "station=HABs-MontereyWharf&var=temp&range=1y");
  const mw = (await page.getByTestId("research-plot").boundingBox())!;
  expect(mw.y + mw.height).toBeLessThanOrEqual(900);
  // F-04: the readout of a sample on the reported-0 lane is whole, on the plot, not cut
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  const p2 = page.getByTestId("research-plot");
  await p2.focus();
  for (let i = 0; i < 60; i++) {
    await page.keyboard.press(i ? "ArrowLeft" : "End");
    if (/reported 0/.test((await page.getByTestId("research-plot-live").textContent()) ?? "")) break;
  }
  await expect(page.getByTestId("research-plot-live")).toContainText("reported 0");
  const tip = p2.getByTestId("plot-tip");
  await expect(tip).toBeInViewport({ ratio: 1 });
  await expect(tip).toContainText("reported 0");
  const tb = (await tip.boundingBox())!;
  const covered = await page.evaluate(([x, y, w, h]) => [[x + 4, y + 4], [x + w - 4, y + h - 4], [x + w / 2, y + h - 3]].every(([px, py]) => !!document.elementFromPoint(px, py)?.closest("[data-testid=plot-tip], [data-plot]")), [tb.x, tb.y, tb.width, tb.height]);
  expect(covered).toBe(true);
  // the reported 0s have their own lane under the axis, labelled, never on the log axis
  await expect(p2.locator("[data-zero-lane]")).toContainText("rep. 0");
  // F-13: the model's reference level is a dashed line named in the axis unit, never a safety limit
  await expect(p2.locator("[data-threshold] line")).toHaveCount(1);
  await expect(page.getByTestId("research-ref-key")).toHaveText("C-HARM model reference, 0.5 ng/mL (not a safety limit)");
  // the empty range: the message and all four actions, without scrolling
  await open(page, MW_1Y);
  await expect(page.getByTestId("research-empty")).toBeInViewport({ ratio: 1 });
  for (const id of ["empty-latest", "empty-all", "empty-neighbour", "empty-source"]) await expect(page.getByTestId(id)).toBeInViewport({ ratio: 1 });
  const box = (await page.getByTestId("research-main").boundingBox())!;
  expect(box.y + box.height).toBeLessThanOrEqual(900);
});

for (const [w, h] of [[1280, 720], [1100, 800]] as const) {
  test(`${w} × ${h}: the sample card fits without scrolling, Same visit included (design issue 3)`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    for (const q of ["station=HABs-SantaCruzWharf&var=pDA&range=3y&day=2026-05-27", "station=HABs-MontereyWharf&var=pDA"]) {
      await open(page, q);
      await expect(page.getByTestId("inspector-others")).toBeVisible();
      await cardWhole(page, h);
      await expect(page.getByTestId("inspector-others").locator("dd").last()).toBeInViewport({ ratio: 1 });
      const pass = (await page.getByTestId("research-record").boundingBox())!;
      expect(pass.height).toBeLessThanOrEqual(h * 0.45 + 1);
    }
    await ctx.close();
  });
}

test("the whole record and the neighbour are one press from the empty range", async ({ page }) => {
  await open(page, MW_1Y);
  await page.getByTestId("empty-all").click();
  await expect(page.getByTestId("research-plot")).toBeVisible();
  await expect(page.getByTestId("range-all")).toHaveAttribute("aria-checked", "true");
  await page.getByTestId("range-1y").click();
  await page.getByTestId("empty-neighbour").click();
  await expect(page.getByTestId("research-title")).toHaveText("Santa Cruz Wharf");
  expect(params(page)).toMatchObject({ station: "HABs-SantaCruzWharf", var: "pDA" });
  await expect(page.getByTestId("research-plot")).toBeVisible();
  await expect(page.getByTestId("research-empty")).toHaveCount(0);
});

test("one plot, the variable tabs as a radiogroup, a comparison on its own panel, the model apart behind its switch", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  // the tabs: arrow keys move the selection and the address
  const rail = page.getByRole("radiogroup", { name: /^Variable measured at Santa Cruz Wharf/ });
  await rail.getByRole("radio", { checked: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => params(page).var).toBe("pn_seriata");
  await expect(page.getByTestId("rail-pn_seriata")).toBeFocused();
  await expect(page.getByTestId("research-variable")).toContainText("Pseudo-nitzschia, seriata group");
  await page.keyboard.press("ArrowLeft");
  await expect.poll(() => params(page).var).toBe("pDA");
  // a comparison variable is a second panel with its own axis, never a second axis on the first
  await tools(page);
  await page.getByTestId("research-compare").selectOption("temp");
  await expect.poll(() => params(page).cmp).toBe("temp");
  await tools(page, false);
  await expect(page.getByTestId("research-plot-cmp")).toBeVisible();
  await expect(page.getByTestId("research-plot").locator("[data-axis=y]")).toHaveCount(1);
  // the model: hidden until asked for; then a separate plot, named a model, with its words
  await expect(page.getByTestId("research-plot-model")).toHaveCount(0);
  await page.getByTestId("research-model-switch").click();
  await expect(page.getByTestId("research-plot-model")).toBeVisible();
  await expect(page.getByTestId("strip-model")).toContainText("Model · C-HARM P(pDA > 500 ng/L)");
  await expect(page.getByTestId("research-model-words")).toContainText("not toxin in seafood");
  await expect(page.getByTestId("research-legend")).toContainText("Reported 0: at or below the lab’s limit");
  // the address restores the whole view
  await page.reload();
  await expect(page.getByTestId("research-plot-cmp")).toBeVisible();
  await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "true");
});

test("keyboard: focus the plot, step to a sample, Enter opens it on the card; Escape clears it; a station opens on its newest sample", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  // a station opens on its newest sample in the window
  await expect(page.getByTestId("inspector-utc")).toContainText("2026-09-30");
  const plot = page.getByTestId("research-plot");
  await plot.focus();
  await page.keyboard.press("End");
  await expect(page.getByTestId("research-plot-live")).toContainText("Sep 30, 2026");
  await page.keyboard.press("ArrowLeft");
  const said = (await page.getByTestId("research-plot-live").textContent())!;
  expect(said).toMatch(/^[A-Z][a-z]{2} \d{1,2}, 2026 · .+ · (measured|at or below an unpublished limit, not absent)/);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("inspector-body")).toHaveAttribute("data-index", /\d+/);
  await expect.poll(() => params(page).day).toMatch(/^2026-09-/);
  await expect(page.getByTestId("inspector-utc")).toContainText("2026-09-");
  await cardWhole(page);
  // Escape clears the sample, and the card says how to pick one
  await plot.focus();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("research-inspector")).toHaveAttribute("data-empty", "");
  await expect(page.getByTestId("research-inspector")).toContainText("Pick a sample");
  expect(params(page).day).toBeUndefined();
});

test("a reported 0 is its own state in the table and on the card", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  await tools(page);
  await page.getByTestId("disc-samples").getByRole("button").first().click();
  const zero = page.getByTestId("plot-table").locator("tbody tr", { hasText: "Reported 0" }).first();
  await zero.click();
  await expect(page.getByTestId("inspector-body")).toHaveAttribute("data-state", "reported_zero");
  await expect(page.getByTestId("research-inspector")).toContainText("Reported 0");
  await expect(page.getByTestId("inspector-value")).toHaveText("0 ng/mL");
  await expect(page.getByTestId("inspector-ratio")).toHaveCount(0);
  await page.getByTestId("research-open-record").click();
  await expect(page.getByTestId("inspector-state")).toContainText("Not quantified, not absent");
});

test("phone, 390 px: the map strip frames the station and its neighbours; a tap on the plot opens the sample; nothing scrolls sideways", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  // design issue 2: the resting sheet leaves at least 40 % of the screen to the chart, and the station and its
  // neighbour are drawn in that strip
  await expect(async () => {
    const sheetTop = (await page.getByTestId("nx-margin").boundingBox())!.y;
    expect(sheetTop).toBeGreaterThanOrEqual(844 * 0.4 - 2);
    for (const id of ["HABs-SantaCruzWharf", "HABs-MontereyWharf"]) {
      const m = (await page.getByTestId(`chart-station-${id}`).boundingBox())!;
      expect(m.y).toBeGreaterThan(40);
      expect(m.y + m.height).toBeLessThan(sheetTop);
    }
  }).toPass({ timeout: 15_000 });
  const plot = page.getByTestId("research-plot");
  // the plot in the middle of the screen, clear of the bars and the destinations
  await plot.evaluate((el) => el.scrollIntoView({ block: "center" }));
  await page.waitForTimeout(600);
  const box = (await plot.boundingBox())!;
  expect(box.width).toBeGreaterThanOrEqual(340);
  expect(box.height).toBeGreaterThanOrEqual(240);
  await page.touchscreen.tap(box.x + box.width * 0.7, box.y + box.height * 0.45);
  await expect(page.getByTestId("inspector-body")).toHaveAttribute("data-index", /\d+/);
  await expect.poll(() => params(page).day).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  // its details are one press away in the sheet
  await page.getByTestId("sample-details").scrollIntoViewIfNeeded();
  await page.getByTestId("sample-details").click();
  await expect(page.getByTestId("inspector-value")).toBeInViewport();
  await expect(page.getByTestId("inspector-value")).toContainText("ng/mL");
  await expect(page.getByTestId("inspector-prev")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  // a link that carries a sample (`day=`) marks it on the plot and the card reads it
  await page.goto(`${OK}/research?station=HABs-MontereyWharf&var=pDA&from=2021-09-30&to=2022-09-30&day=2022-08-31`);
  await expect(page.getByTestId("research-plot").locator("[data-selection]")).toBeAttached();
  await expect(page.getByTestId("inspector-local")).toContainText("Aug 31, 2022");
  await page.getByTestId("sample-details").scrollIntoViewIfNeeded();
  await page.getByTestId("sample-details").click();
  await expect(page.getByTestId("inspector-value")).toBeInViewport();
  await expect(page.getByTestId("inspector-value")).toHaveText("0.05220236 ng/mL");
  // the empty range on a phone keeps its actions inside the page
  await page.goto(`${OK}/research?${MW_1Y}`);
  await expect(page.getByTestId("empty-latest")).toBeAttached();
  for (const id of ["empty-latest", "empty-all", "empty-neighbour", "empty-source"]) {
    await page.getByTestId(id).scrollIntoViewIfNeeded();
    const b = (await page.getByTestId(id).boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.x + b.width).toBeLessThanOrEqual(390);
  }
  await ctx.close();
});

for (const [w, h] of [[1440, 900], [390, 844], [320, 640]] as const) {
  test(`axe at ${w} px: the measured view, the empty range and the sample card have no serious or critical violation`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, ...(w < 500 ? { hasTouch: true, isMobile: true } : {}), storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    for (const q of ["station=HABs-SantaCruzWharf&var=pDA&range=1y&cmp=temp", MW_1Y, "station=HABs-MontereyWharf&var=pDA&from=2021-09-30&to=2022-09-30&day=2022-08-31"]) {
      await open(page, q);
      await page.waitForTimeout(800);
      expect(await serious(page), q).toEqual([]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth), q).toBeLessThanOrEqual(w);
    }
    await ctx.close();
  });
}

for (const [w, h] of [[390, 844], [320, 640]] as const) {
  test(`phone ${w} px: "Latest sample" opens the sample as a tap does, with its mark on the plot; the range control is inside the screen`, async ({ browser }) => {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    // R-4: with the range the reader's own, the range control is inside the screen and names no preset
    await open(page, "station=HABs-SantaCruzWharf&var=pDA&from=2025-01-10&to=2025-09-20");
    for (const r of ["range-90d", "range-all"]) {
      const b = (await page.getByTestId(r).boundingBox())!;
      expect(b.x).toBeGreaterThanOrEqual(0);
      expect(b.x + b.width).toBeLessThanOrEqual(w);
      await expect(page.getByTestId(r)).toHaveAttribute("aria-checked", "false");
    }
    // the empty range's first action
    await open(page, MW_1Y);
    await page.getByTestId("empty-latest").scrollIntoViewIfNeeded();
    await page.getByTestId("empty-latest").tap();
    await expect(page.getByTestId("inspector-utc")).toContainText("2022-08-31");
    const mark = page.getByTestId("research-plot").locator("[data-selection] [data-ring]");
    await mark.scrollIntoViewIfNeeded();
    await expect(mark).toBeInViewport({ ratio: 1 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(w);
    await ctx.close();
  });
}

test("the freshness line keeps an abbreviation's case and follows the variable (Santa Cruz Wharf, 3 y)", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=3y");
  await expect(page.getByTestId("research-status")).toHaveText(/^pDA last measured [A-Z][a-z]{2} \d{1,2}, \d{4} \(\d+ days ago\)$/);
  for (const id of ["temp", "pn_seriata", "chl_extracted"] as const) {
    await page.getByTestId(`rail-${id}`).click();
    await expect(page.getByTestId("research-status")).toHaveText(id === "temp" ? /^Temperature last measured / : id === "pn_seriata" ? /^Seriata group last measured / : /^Chlorophyll last measured /);
  }
  // chlorophyll is said to be algae, not toxin; every caption is water samples, not seafood tests
  await expect(page.getByTestId("research-window-count")).toContainText("algae, not toxin");
  await expect(page.getByTestId("research-variable")).toContainText("water samples, not seafood tests");
});

test("on a slow machine the last of several quick choices wins: variable (compact select and tabs), comparison, station, range", async ({ browser }) => {
  test.setTimeout(120_000);
  // phone, CPU 8x: the page's render lags each change; every choice is made before the previous one has rendered
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  await expect(page.getByTestId("research-plot")).toBeVisible();
  const cdp = await ctx.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 8 });
  const pick = page.getByTestId("research-variable-select");
  const agree = async (v: string, heading: string) => {
    await expect.poll(() => params(page).var, { timeout: 20_000 }).toBe(v);
    await expect(pick).toHaveValue(v, { timeout: 20_000 });
    await expect(page.getByTestId("research-variable")).toContainText(heading, { timeout: 20_000 });
    await expect(page.getByTestId("nx-research")).toHaveAttribute("data-var", v);
    await expect(page.getByTestId(`rail-${v}`)).toHaveAttribute("aria-checked", "true");
    // and it stays so once everything has settled
    await page.waitForTimeout(1500);
    expect(params(page).var).toBe(v);
    await expect(pick).toHaveValue(v);
  };
  for (const [seq, last, heading] of [
    [["temp", "pDA"], "pDA", "Particulate domoic acid"],
    [["temp", "pDA", "temp"], "temp", "Water temperature"],
    [["chl_extracted", "pn_seriata", "temp", "pDA", "chl_extracted"], "chl_extracted", "Chlorophyll-a, extracted"],
    [["pDA", "temp", "pDA"], "pDA", "Particulate domoic acid"],
  ] as const) {
    // no waiting between the choices: each is made while the one before is still being rendered
    for (const v of seq) await pick.selectOption(v, { noWaitAfter: true });
    await agree(last, heading);
  }
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
  await ctx.close();

  // desktop, CPU 8x: the tabs, the comparison, the station picker and the range share the mechanism
  const wide = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
  const p2 = await wide.newPage();
  await open(p2, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  await expect(p2.getByTestId("research-plot")).toBeVisible();
  const c2 = await wide.newCDPSession(p2);
  await c2.send("Emulation.setCPUThrottlingRate", { rate: 8 });
  for (const id of ["temp", "pn_seriata", "pDA", "temp"]) await p2.getByTestId(`rail-${id}`).click({ noWaitAfter: true });
  await expect.poll(() => params(p2).var, { timeout: 20_000 }).toBe("temp");
  await expect(p2.getByTestId("rail-temp")).toHaveAttribute("aria-checked", "true", { timeout: 20_000 });
  await expect(p2.getByTestId("research-variable")).toContainText("Water temperature");
  await tools(p2);
  const cmp = p2.getByTestId("research-compare");
  for (const v of ["pDA", "chl_extracted", "pDA"]) await cmp.selectOption(v, { noWaitAfter: true });
  await expect.poll(() => params(p2).cmp, { timeout: 20_000 }).toBe("pDA");
  await expect(cmp).toHaveValue("pDA", { timeout: 20_000 });
  expect(params(p2).var).toBe("temp"); // the variable chosen before is kept
  const st = p2.getByTestId("research-station-select");
  for (const v of ["HABs-MontereyWharf", "HABs-StearnsWharf", "HABs-MontereyWharf"]) await st.selectOption(v, { noWaitAfter: true });
  await expect.poll(() => params(p2).station, { timeout: 20_000 }).toBe("HABs-MontereyWharf");
  await expect(st).toHaveValue("HABs-MontereyWharf", { timeout: 20_000 });
  await expect(p2.getByTestId("research-title")).toHaveText("Monterey Wharf", { timeout: 20_000 });
  await tools(p2, false);
  for (const r of ["range-90d", "range-3y", "range-1y"]) await p2.getByTestId(r).click({ noWaitAfter: true });
  await expect(p2.getByTestId("range-1y")).toHaveAttribute("aria-checked", "true", { timeout: 20_000 });
  await p2.waitForTimeout(1500);
  expect(params(p2)).toMatchObject({ station: "HABs-MontereyWharf", var: "temp" });
  await expect(p2.getByTestId("range-1y")).toHaveAttribute("aria-checked", "true");
  await c2.send("Emulation.setCPUThrottlingRate", { rate: 1 });
  await wide.close();
});
