import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * My Coast, v10 pass 2 (the review's failed composition, reworked): a decision row above the fold on
 * the trip's live waters, one evidence section, no stack of accordions, a worksheet whose answer leads
 * (gross as soon as catch and price exist, the missing list once, the inputs folded once complete).
 * The kept briefing offline: mycoast-kept-offline.spec.ts.
 */
const NOW = "2026-10-13T06:00:00Z";
const SANTA_CRUZ = `${OK}/coast?place=port-593&for=commercial&species=salmon`;

async function open(page: Page, url = SANTA_CRUZ) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
  await expect(page.getByTestId("decision-row")).toBeVisible();
}

test("1440: one decision row above the fold, beside a full-height map that stays while the column scrolls", async ({ page }) => {
  await open(page);
  const row = page.getByTestId("decision-row");
  const cells = ["glance-wind", "glance-seas", "glance-tide", "glance-notices-cell", "brief-econ"];
  const boxes = await Promise.all(cells.map((id) => row.getByTestId(id).boundingBox()));
  // one row: every reading starts on the same line, above the fold
  for (const b of boxes) {
    expect(Math.abs(b!.y - boxes[0]!.y)).toBeLessThan(2);
    expect(b!.y + b!.height).toBeLessThan(900);
  }
  // no empty "—" tiles for what is not measured nearby: one small line says so instead
  await expect(page.getByTestId("glance-waves")).toHaveCount(0);
  await expect(page.getByTestId("glance-water-t")).toHaveCount(0);
  await expect(row.getByTestId("glance-measured")).toBeVisible();
  // the notices are a count, unverified, linked to the list
  await expect(row.getByTestId("glance-notices-cell")).toContainText("not verified");
  await expect(row.getByTestId("glance-notices")).toHaveAttribute("href", "#official-h");
  // the map is the right side of the page, the whole window high, and still there after scrolling
  const map = page.getByTestId("locator");
  const m = (await map.boundingBox())!;
  expect(m.x).toBeGreaterThan(700);
  expect(m.height).toBeGreaterThanOrEqual(880);
  await page.mouse.wheel(0, 1400);
  await page.waitForTimeout(300);
  const after = (await map.boundingBox())!;
  expect(after.y).toBeGreaterThanOrEqual(-1);
  expect(after.y).toBeLessThan(5);
  await expect(page.getByTestId("locator-legend")).toContainText(/Chlorophyll|No satellite chlorophyll/);
});

test("one evidence section; the notices counted once; no stack of accordions", async ({ page }) => {
  await open(page);
  await expect(page.getByTestId("matters").locator(":scope > li").first()).toBeVisible();
  await expect(page.getByTestId("brief-now")).toHaveCount(1);
  await expect(page.getByText("More for this trip")).toHaveCount(0);
  await expect(page.getByTestId("brief-now").getByTestId("brief-official")).toBeVisible();
  await expect(page.getByTestId("brief-now").getByTestId("brief-measured")).toBeVisible();
  // the count of notices tied here is said once on the page, in the decision row
  const text = await page.locator("main").innerText();
  expect(text.match(/\b3 notices tied to these waters/g) ?? []).toHaveLength(0);
  expect((text.match(/tied here/g) ?? []).length).toBe(1);
  // the disclosures that remain sit where their subject is, never two in a row
  const folds = await page.locator('[data-testid$="-toggle"]').evaluateAll((els) => els.filter((e) => (e as HTMLElement).offsetParent !== null).map((e) => (e as HTMLElement).getBoundingClientRect().top + window.scrollY));
  for (let i = 1; i < folds.length; i++) expect(folds[i] - folds[i - 1], `folds ${i - 1} and ${i}`).toBeGreaterThan(20);
  // the season is two lines plus its links; CDFW's list is behind "Season details"
  await expect(page.getByTestId("seasons-all")).toBeHidden();
  await page.getByTestId("brief-seasons-toggle").click();
  await expect(page.getByTestId("seasons-all")).toBeVisible();
});

test("phone: place and trip, a one-paragraph season, then wind, seas, tide and notices on the first screen", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await open(page);
  for (const id of ["coast-title", "trip-summary", "glance-season", "glance-wind", "glance-seas", "glance-tide", "glance-notices-cell"]) {
    const b = await page.getByTestId(id).boundingBox();
    expect(b!.y + b!.height, id).toBeLessThan(844);
  }
  // the map comes after the readings, as a band
  const map = (await page.getByTestId("locator").boundingBox())!;
  expect(map.y).toBeGreaterThan((await page.getByTestId("glance-tide").boundingBox())!.y);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  await ctx.close();
});

test("worksheet: gross leads once catch and price exist; the missing list is said once; complete, the inputs fold to one line", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${SANTA_CRUZ}&sheet=trip`);
  const s = page.getByTestId("trip-economics-section");
  await expect(s.getByTestId("econ-worksheet")).toBeVisible();
  // the one framing line, above every number
  await expect(page.getByTestId("econ-hypothetical")).toContainText("Hypothetical: your numbers, not a forecast or permission to fish.");
  await expect(page.getByTestId("econ-hypothetical")).toContainText("Neritic holds no commercial season");
  await s.getByTestId("econ-catch-0-lb").fill("400");
  await s.getByTestId("econ-catch-0-price").fill("9");
  await expect(s.getByTestId("econ-sum-gross")).toHaveText("$3,600");
  const results = s.getByTestId("econ-results");
  await expect(results.getByTestId("econ-needs")).toHaveCount(1);
  expect((await results.innerText()).match(/Still needed/g) ?? []).toHaveLength(1);
  await expect(results.getByTestId("econ-summary").getByTestId("econ-need")).toHaveCount(0);
  // complete it, then reopen: the answer fills the page and the inputs are one line
  await s.getByTestId("econ-use-diesel").click();
  await s.getByTestId("econ-fuel-mode-gallons").click();
  await s.getByTestId("econ-fuel-gal").fill("40");
  await s.getByTestId("econ-crew-none").click();
  await expect(s.getByTestId("econ-sum-net")).toHaveText(/^\+\$/);
  await expect(s.getByTestId("econ-steps")).toBeVisible(); // completed while typing: the inputs stay open
  await page.reload();
  await expect(s.getByTestId("econ-results")).toHaveAttribute("data-folded", "true");
  await expect(s.getByTestId("econ-steps")).toHaveCount(0);
  const sum = s.getByTestId("econ-inputs-summary");
  await expect(sum).toContainText("400 lb at $9.00/lb");
  await expect(sum).toContainText("40 gal");
  await expect(sum).toContainText("no hired crew");
  await expect(s.getByTestId("econ-waterfall")).toBeVisible();
  await expect(s.getByTestId("econ-sensitivity")).toBeVisible();
  await s.getByTestId("econ-edit-inputs").click();
  await expect(s.getByTestId("econ-steps")).toBeVisible();
  await expect(s.getByTestId("econ-fuel-gal")).toHaveValue("40");
  // the briefing's reading: net, with the trip cost
  await page.getByTestId("econ-back").click();
  await expect(page.getByTestId("econ-glance-result")).toHaveAttribute("data-label", "net to boat");
});

test("v10 fix 2: the 60-hour lanes: a gap said once per span, nothing under the now line, tide turns inside the lane", async ({ page }) => {
  await open(page);
  const lanes = page.getByTestId("lanes");
  await lanes.scrollIntoViewIfNeeded();
  const model = page.getByTestId("lane-model");
  await expect(model.getByTestId("model-gap").first()).toBeVisible();
  // the fixture's run ends before this window: one span, its words once
  await expect(model.getByTestId("model-gap")).toHaveCount(1);
  expect(((await model.innerText()).match(/no run/g) ?? []).length).toBe(1);
  // the buoy's words start after the now line
  const now = await lanes.evaluate((el) => {
    const line = [...el.querySelectorAll("span")].find((s) => s.className.includes("w-px") && s.className.includes("bg-n-ink"));
    return line ? line.getBoundingClientRect().x : null;
  });
  expect(now).not.toBeNull();
  const measured = page.getByTestId("measured-text");
  if (await measured.count()) expect((await measured.boundingBox())!.x).toBeGreaterThan(now!);
  const gapText = model.getByTestId("model-gap").first().locator("span span");
  expect((await gapText.boundingBox())!.x).toBeGreaterThan(now!);
  // every tide turn's label lies inside the lane
  const tide = (await page.getByTestId("lane-tide").boundingBox())!;
  for (const l of await page.getByTestId("tide-turn-label").all()) {
    const b = (await l.boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(tide.x - 1);
    expect(b.x + b.width).toBeLessThanOrEqual(tide.x + tide.width + 1);
  }
});

test("v10 fix 2 (1440): the map legend clears the header chip; 320: the readings are one column of compact rows", async ({ page, browser }) => {
  await open(page);
  const chip = (await page.getByTestId("status-center").boundingBox())!;
  const legend = (await page.getByTestId("locator-legend").boundingBox())!;
  expect(legend.y).toBeGreaterThanOrEqual(chip.y + chip.height + 24);
  const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
  const p = await ctx.newPage();
  await open(p);
  const row = p.getByTestId("decision-row");
  const boxes = await Promise.all(["glance-wind", "glance-seas", "glance-tide", "glance-notices-cell"].map((id) => row.getByTestId(id).boundingBox()));
  for (let i = 1; i < boxes.length; i++) expect(boxes[i]!.y).toBeGreaterThan(boxes[i - 1]!.y + 10);
  for (const b of boxes) {
    expect(b!.width).toBeGreaterThan(280);
    expect(b!.height).toBeLessThan(90);
  }
  await ctx.close();
});
