import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "../ports";

/**
 * Research acceptance (v13 brief, Agent A; journey 4): Monterey Wharf → pDA → the latest real observation →
 * source and QC → change the range → model beside measured without conflation. The fixture is CalHABMAP as
 * recorded: Monterey Wharf has no pDA after 2022-08-31 and its recent visits record temperature only.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const params = (page: Page) => Object.fromEntries(new URL(page.url()).searchParams);
const MW_1Y = "station=HABs-MontereyWharf&var=pDA&range=1y";

async function open(page: Page, q: string) {
  await page.addInitScript(() => localStorage.setItem("nx-opening-seen-v1", "1"));
  await page.goto(`${OK}/research?${q}`);
  await expect(page.getByTestId("research-record")).toBeVisible();
}

async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

test("Monterey Wharf, pDA, one year: the empty range is explained, and one press reaches the 2022-08-31 sample in the inspector", async ({ page }) => {
  await open(page, MW_1Y);
  // the status and the rail are per variable: pDA's last sample, not the station's newest visit
  await expect(page.getByTestId("research-status")).toContainText("last measured Aug 31, 2022");
  await expect(page.getByTestId("rail-last-pDA")).toContainText("Last Aug 31, 2022");
  await expect(page.getByTestId("rail-last-pDA")).toContainText("years ago");
  await expect(page.getByTestId("rail-count-pDA")).toHaveText("0 in window");
  await expect(page.getByTestId("rail-count-temp")).not.toHaveText("0 in window");
  await expect(page.getByTestId("rail-last-chl_extracted")).toContainText("Last Sep 11, 2024");
  // no empty axis: the reason, what the visits did record, and that it is not a zero
  const empty = page.getByTestId("research-empty");
  await expect(empty).toContainText("Last measured Aug 31, 2022");
  await expect(empty).toContainText("0.052 ng/mL");
  await expect(empty).toContainText("recorded temperature only");
  await expect(empty).toContainText("Not measured is not zero");
  await expect(page.getByTestId("research-plot")).toHaveCount(0);
  // the model is not drawn beside nothing measured, and says why; its mandatory words stay
  await expect(page.getByTestId("research-model-collapsed")).toContainText("is not drawn");
  await expect(page.getByTestId("research-plot-model")).toHaveCount(0);
  await expect(page.getByTestId("research-model-words")).toContainText("not toxin in seafood");
  // the four actions
  await expect(page.getByTestId("empty-latest")).toContainText("Aug 31, 2022");
  await expect(page.getByTestId("empty-all")).toBeVisible();
  await expect(page.getByTestId("research-empty-neighbour")).toContainText("Santa Cruz Wharf, 41 km away");
  await expect(page.getByTestId("empty-source")).toHaveAttribute("href", "https://erddap.sccoos.org/erddap/tabledap/HABs-MontereyWharf.html");

  // jump to the latest sample: the window holds it and the inspector reads it exactly
  await page.getByTestId("empty-latest").click();
  await expect.poll(() => params(page).day).toBe("2022-08-31");
  await expect(page.getByTestId("research-plot")).toBeVisible();
  await expect(page.getByTestId("research-selected")).toBeVisible();
  const insp = page.getByTestId("research-inspector");
  await expect(page.getByTestId("inspector-value")).toHaveText("0.05220236 ng/mL");
  await expect(page.getByTestId("inspector-state")).toContainText("Measured, as published");
  await expect(page.getByTestId("inspector-utc")).toHaveText("2022-08-31 14:45 UTC");
  await expect(page.getByTestId("inspector-local")).toContainText("Aug 31, 2022, 7:45 AM");
  await expect(insp).toContainText("5 m");
  await expect(insp).toContainText("Monterey Wharf (HAB_MWII)");
  await expect(page.getByTestId("inspector-sampleid")).toContainText("At the source");
  await expect(page.getByTestId("inspector-others")).toContainText("Water temperature");
  await expect(page.getByTestId("inspector-qc").locator("li")).not.toHaveCount(0);
  await expect(page.getByTestId("inspector-source-row")).toHaveAttribute("href", /HABs-MontereyWharf\.htmlTable\?time,.*SampleID.*&time=2022-08-31T14%3A45%3A00Z$/);
  await expect(page.getByTestId("inspector-citation")).toContainText("sample of 2022-08-31 14:45 UTC");
  // with a measured value in the window the model is drawn, on its own panel
  await expect(page.getByTestId("research-model-collapsed")).toHaveCount(0);
  // Previous steps through pDA samples (not the visits between them); Next is at the record's end
  await expect(page.getByTestId("inspector-next")).toHaveAttribute("aria-disabled", "true");
  await page.getByTestId("inspector-prev").click();
  await expect(page.getByTestId("inspector-utc")).not.toHaveText("2022-08-31 14:45 UTC");
  await expect(page.getByTestId("research-inspector-live")).toContainText("Sample of 2022-");
});

test("1440 × 900, on load: the whole plot is in the first screen with the variable rail; so is the empty range with its actions", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  const plot = (await page.getByTestId("research-plot").boundingBox())!;
  expect(plot.width).toBeGreaterThanOrEqual(800);
  expect(plot.height).toBeGreaterThanOrEqual(360);
  expect(plot.y + plot.height).toBeLessThanOrEqual(900);
  const rail = (await page.getByTestId("research-rail").boundingBox())!;
  expect(rail.y).toBeGreaterThanOrEqual(0);
  expect(rail.y + rail.height).toBeLessThanOrEqual(plot.y);
  await expect(page.getByTestId("rail-temp")).toBeInViewport({ ratio: 1 });
  // Monterey Wharf has one more variable: its plot still ends inside the screen
  await open(page, "station=HABs-MontereyWharf&var=temp&range=1y");
  const mw = (await page.getByTestId("research-plot").boundingBox())!;
  expect(mw.y + mw.height).toBeLessThanOrEqual(900);
  // the empty range: the message and all four actions, without scrolling
  await open(page, MW_1Y);
  await expect(page.getByTestId("research-empty")).toBeInViewport({ ratio: 1 });
  for (const id of ["empty-latest", "empty-all", "empty-neighbour", "empty-source"]) await expect(page.getByTestId(id)).toBeInViewport({ ratio: 1 });
  const box = (await page.getByTestId("research-main").boundingBox())!;
  expect(box.y + box.height).toBeLessThanOrEqual(900);
});

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

test("one large plot, the variable rail as a radiogroup, a comparison on its own panel, the model apart", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  const box = (await page.getByTestId("research-plot").boundingBox())!;
  expect(box.width).toBeGreaterThanOrEqual(800);
  expect(box.height).toBeGreaterThanOrEqual(360);
  // the rail: arrow keys move the selection and the address
  const rail = page.getByRole("radiogroup", { name: /^Variable measured at Santa Cruz Wharf/ });
  await rail.getByRole("radio", { checked: true }).focus();
  await page.keyboard.press("ArrowDown");
  await expect.poll(() => params(page).var).toBe("pn_seriata");
  await expect(page.getByTestId("rail-pn_seriata")).toBeFocused();
  await expect(page.getByTestId("research-variable")).toContainText("Pseudo-nitzschia, seriata group");
  await page.keyboard.press("ArrowUp");
  await expect.poll(() => params(page).var).toBe("pDA");
  // a comparison variable is a second panel with its own axis, never a second axis on the first
  await page.getByTestId("research-compare").selectOption("temp");
  await expect.poll(() => params(page).cmp).toBe("temp");
  await expect(page.getByTestId("research-plot-cmp")).toBeVisible();
  await expect(page.getByTestId("research-plot").locator("[data-axis=y]")).toHaveCount(1);
  // the model: a separate plot, named a model, with its words
  await expect(page.getByTestId("research-plot-model")).toBeVisible();
  await expect(page.getByTestId("strip-model")).toContainText("Model · C-HARM P(pDA > 500 ng/L)");
  await expect(page.getByTestId("research-legend")).toContainText("Reported 0: at or below an unpublished limit, not absent");
  await expect(page.getByTestId("research-legend")).toContainText("Model: computed, not measured");
  // the address restores the whole view
  await page.reload();
  await expect(page.getByTestId("research-plot-cmp")).toBeVisible();
  await expect(page.getByTestId("rail-pDA")).toHaveAttribute("aria-checked", "true");
});

test("keyboard: focus the plot, step to a sample, Enter opens it in the inspector; Latest sample pins the newest", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
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
  // a reported 0 is inspected as a reported 0, never as "none"
  await page.getByRole("button", { name: "Close the details" }).click();
  await expect(page.getByTestId("research-inspector")).toHaveAttribute("data-empty", "");
  await page.getByTestId("range-latest").click();
  await expect.poll(() => params(page).day).toBe("2026-09-30");
  await expect(page.getByTestId("inspector-utc")).toContainText("2026-09-30");
});

test("a reported 0 is its own state in the table and the inspector", async ({ page }) => {
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  await page.getByTestId("disc-samples").getByRole("button").first().click();
  const zero = page.getByTestId("plot-table").locator("tbody tr", { hasText: "Reported 0" }).first();
  await zero.click();
  await expect(page.getByTestId("inspector-body")).toHaveAttribute("data-state", "reported_zero");
  await expect(page.getByTestId("inspector-value")).toHaveText("0 ng/mL");
  await expect(page.getByTestId("inspector-state")).toContainText("Not quantified, not absent");
});

test("phone, 390 px: a tap on the plot opens the sample; nothing scrolls sideways", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await open(page, "station=HABs-SantaCruzWharf&var=pDA&range=1y");
  const plot = page.getByTestId("research-plot");
  // the plot in the middle of the screen, clear of the header and the destinations, and the header settled
  await plot.evaluate((el) => el.scrollIntoView({ block: "center" }));
  await page.waitForTimeout(600);
  const box = (await plot.boundingBox())!;
  expect(box.width).toBeGreaterThanOrEqual(358);
  expect(box.height).toBeGreaterThanOrEqual(240);
  await page.touchscreen.tap(box.x + box.width * 0.7, box.y + box.height * 0.5);
  await expect(page.getByTestId("inspector-body")).toHaveAttribute("data-index", /\d+/);
  await expect(page.getByTestId("inspector-value")).toContainText("ng/mL");
  // the card is pinned above the destinations: in view without scrolling, the plot still above it
  await expect(page.getByTestId("inspector-value")).toBeInViewport();
  await expect(page.getByTestId("inspector-prev")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  // a link that carries a sample (`day=`) marks it on the plot and offers its details; the card pins on a tap
  await page.goto(`${OK}/research?station=HABs-MontereyWharf&var=pDA&from=2021-09-30&to=2022-09-30&day=2022-08-31`);
  await expect(page.getByTestId("research-plot")).toBeInViewport({ ratio: 0.9 });
  await expect(page.getByTestId("research-selected")).toBeVisible();
  await expect(page.getByTestId("inspector-value")).toBeHidden();
  await expect(page.getByTestId("research-sample-line")).toContainText("Aug 31, 2022: 0.052 ng/mL");
  await page.getByTestId("sample-details").scrollIntoViewIfNeeded();
  await page.getByTestId("sample-details").click();
  await expect(page.getByTestId("inspector-value")).toBeInViewport();
  await expect(page.getByTestId("inspector-value")).toHaveText("0.05220236 ng/mL");
  // the empty range on a phone keeps its actions inside the page
  await page.goto(`${OK}/research?${MW_1Y}`);
  await expect(page.getByTestId("empty-latest")).toBeVisible();
  for (const id of ["empty-latest", "empty-all", "empty-neighbour", "empty-source"]) {
    const b = (await page.getByTestId(id).boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.x + b.width).toBeLessThanOrEqual(390);
  }
  await ctx.close();
});

for (const [w, h] of [[1440, 900], [390, 844], [320, 640]] as const) {
  test(`axe at ${w} px: the measured view, the empty range and the inspector have no serious or critical violation`, async ({ browser }) => {
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
