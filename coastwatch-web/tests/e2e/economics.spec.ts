import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * v10 trip economics (src/v4/coast/economics): the briefing's glance and the worksheet view
 * (?sheet=trip). Every result appears as soon as its own inputs exist and otherwise names exactly what it
 * waits for, as links that focus the field. Nothing numeric is pre-filled (references are tapped in with
 * their source), the crew model has no default, recreational trips are cost-only, and the worksheet is
 * framed as hypothetical, never as permission to fish. Kept on the device.
 */
const NOW = "2026-10-13T06:00:00Z";
const PRINCETON = `${OK}/coast?place=port-452&for=commercial&species=dungeness`;
const SANTA_CRUZ = `${OK}/coast?place=port-593&for=commercial&species=salmon`;
const BANNED = /\b(safe|go\/no-go|good day|hotspot|expected catch|price forecast)\b/i;

async function openWorksheet(page: Page, url = PRINCETON) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${url}&sheet=trip`);
  const s = page.getByTestId("trip-economics-section");
  await expect(s.getByTestId("econ-worksheet")).toBeVisible();
  await expect(page.getByTestId("econ-kicker-head")).toContainText("Hypothetical: your numbers, not a forecast or permission to fish.");
  return s;
}

const formula = (page: Page, step: string) => page.getByTestId(`${step}-sum`);

/** Princeton Dungeness: the v6 worked example, typed and tapped as a user would. */
async function fillDungeness(page: Page) {
  const s = page.getByTestId("trip-economics-section");
  await s.getByTestId("econ-catch-0-lb").fill("1000");
  await s.getByTestId("econ-catch-0-price").fill("3.50");
  await s.getByTestId("econ-nm").fill("20");
  await s.getByTestId("econ-kt").fill("8");
  await s.getByTestId("econ-use-steamGph").click();
  await s.getByTestId("econ-hours").fill("8");
  await s.getByTestId("econ-use-fishingGph").click();
  await s.getByTestId("econ-fuel-price").fill("5");
  await s.getByTestId("econ-cost-0-usd").fill("200");
  await s.getByTestId("econ-crew-afterExpenses").click();
  await s.getByTestId("econ-use-crew-pct").first().click();
}

test("a fresh worksheet fills nothing in: references sit beside the fields with their source and are used only on a tap", async ({ page }) => {
  const s = await openWorksheet(page);
  await expect(s.getByTestId("econ-fishery")).toHaveValue("dungeness");
  for (const id of ["econ-steam-gph", "econ-fish-gph", "econ-catch-0-lb", "econ-catch-0-price", "econ-fuel-price", "econ-nm"]) await expect(s.getByTestId(id), id).toHaveValue("");
  // the crew model has no default: none of the four is chosen
  await expect(s.locator('[data-testid^="econ-crew-"][data-on]')).toHaveCount(0);
  await expect(s.getByTestId("econ-crew-unset")).toContainText("Nothing is assumed");
  const steam = s.getByTestId("econ-ref-steamGph");
  await expect(steam).toContainText("NOAA NWFSC-121, Table 31 (2008 survey)");
  await expect(steam).toContainText("Fleet average then, not a norm");
  await s.getByTestId("econ-use-steamGph").click();
  await expect(s.getByTestId("econ-steam-gph")).toHaveValue("7.8");
  await expect(steam).toContainText("basis: NOAA NWFSC-121");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("Past average, not today’s dock price");
  await expect(s.getByTestId("econ-catch-0-ref")).toContainText("$3.51/lb");
  await expect(s.getByTestId("econ-diesel-ref")).toContainText("$8.242");
});

test("Princeton Dungeness, complete: the engine's numbers, the waterfall, break-even and the 3 × 3 grid", async ({ page }) => {
  const s = await openWorksheet(page);
  await fillDungeness(page);
  // fuel 19.5 + 38.4 = 57.9 gal × $5 = $289.50; + $200 → $490
  await expect(s.getByTestId("econ-sum-cost")).toHaveText("$490");
  await expect(s.getByTestId("econ-r-gross")).toHaveText("$3,500");
  await expect(s.getByTestId("econ-r-crew")).toHaveText("−$846");
  await expect(s.getByTestId("econ-sum-net")).toHaveText("+$2,165");
  await expect(s.getByTestId("econ-r-net")).toHaveText("+$2,165");
  await expect(s.getByTestId("econ-sum-breakeven")).toHaveText("140 lb");
  await expect(s.getByTestId("econ-sens-0-0")).toHaveText("+$2,165");
  await expect(formula(page, "econ-g-vessel")).toContainText("Fuel cost $290 = 57.9 gal × $5.00/gal");
  await expect(s.getByTestId("econ-needs")).toHaveCount(0);
  // a negative is refused where it is typed
  await s.getByTestId("econ-nm").fill("-4");
  await expect(s.getByText("Can't be negative")).toBeVisible();
  expect(await s.innerText()).not.toMatch(BANNED);
});

test("Santa Cruz salmon, the owner's state: gross with its sum, exactly two missing inputs, each a link that focuses its field", async ({ page }) => {
  const s = await openWorksheet(page, SANTA_CRUZ);
  await s.getByTestId("econ-catch-0-lb").fill("400");
  await s.getByTestId("econ-catch-0-price").fill("9");
  await s.getByTestId("econ-use-diesel").click();
  await expect(s.getByTestId("econ-r-gross")).toHaveText("$3,600");
  await expect(formula(page, "econ-g-catch")).toContainText("Gross $3,600 = 400 lb × $9.00");
  const needs = s.getByTestId("econ-needs").getByTestId("econ-need");
  await expect(needs).toHaveText(["fuel used", "how crew is paid"]);
  await expect(s.getByTestId("econ-sum-net")).toHaveText("—");
  await needs.filter({ hasText: "fuel used" }).click();
  await expect(s.getByTestId("econ-nm")).toBeFocused();
  await needs.filter({ hasText: "how crew is paid" }).click();
  await expect(s.locator("#econ-in-crew-kind")).toBeFocused();
  // the briefing's glance says the same, and its links open the worksheet at that field
  await page.getByTestId("econ-back").click();
  const g = page.getByTestId("brief-econ");
  // v10 pass 2: one reading, the gross, then the one line of what net still needs
  await expect(g.getByTestId("econ-glance-result")).toContainText("$3,600");
  await expect(g.getByTestId("econ-glance-result")).toHaveAttribute("data-label", "gross");
  await expect(g.getByTestId("econ-glance-need")).toHaveText(["fuel used", "how crew is paid"]);
  await g.getByTestId("econ-glance-need").filter({ hasText: "how crew is paid" }).click();
  await expect(page).toHaveURL(/sheet=trip.*focus=crew\.kind/);
  await expect(page.locator("#econ-in-crew-kind")).toBeFocused();
});

test("closed species: the hypothetical framing and the season sit above every number, on the worksheet and the glance", async ({ page }) => {
  const s = await openWorksheet(page, SANTA_CRUZ);
  // v10 pass 2: the season and the hypothetical framing are one line, said once
  await expect(page.getByTestId("econ-official")).toContainText("closed on CDFW’s sport list");
  await expect(page.getByTestId("econ-official")).toContainText("Neritic holds no commercial season");
  const h = page.getByTestId("econ-hypothetical");
  await expect(h).toContainText("not a forecast or permission to fish");
  await expect(h.getByTestId("econ-closed")).toContainText("Ocean Salmon is closed");
  await expect(page.locator("main").getByText(/permission to fish/)).toHaveCount(1);
  const box = (id: string) => page.getByTestId(id).boundingBox();
  expect((await box("econ-hypothetical"))!.y).toBeLessThan((await box("econ-results"))!.y);
  await page.getByTestId("econ-back").click();
  await expect(page.getByTestId("glance-official")).toContainText("Nothing here says a trip is allowed");
});

test("crew model unset: partial results stand, net and the sensitivity wait for it; 'no hired crew' completes it", async ({ page }) => {
  const s = await openWorksheet(page, SANTA_CRUZ);
  await s.getByTestId("econ-catch-0-lb").fill("400");
  await s.getByTestId("econ-catch-0-price").fill("9");
  await s.getByTestId("econ-use-diesel").click();
  await s.getByTestId("econ-nm").fill("40");
  await s.getByTestId("econ-kt").fill("8");
  await s.getByTestId("econ-use-steamGph").click();
  await s.getByTestId("econ-hours").fill("10");
  await s.getByTestId("econ-use-fishingGph").click();
  await s.getByTestId("econ-cost-0-usd").fill("60");
  await s.getByTestId("econ-cost-1-usd").fill("40");
  // 40 nm ÷ 8 kt × 7.8 + 10 h × 3.6 = 75 gal × $8.242 = $618; + $100
  await expect(s.getByTestId("econ-sum-cost")).toHaveText("$718");
  await expect(s.getByTestId("econ-r-gross")).toHaveText("$3,600");
  await expect(s.getByTestId("econ-sum-net")).toHaveText("—");
  await expect(s.getByTestId("econ-needs").getByTestId("econ-need")).toHaveText(["how crew is paid"]);
  await expect(s.getByTestId("econ-sensitivity")).toContainText("Waits for");
  await expect(s.getByTestId("econ-waterfall").locator('[data-row="crew"]')).toContainText("waits for how crew is paid");
  await s.getByTestId("econ-crew-none").click();
  await expect(s.getByTestId("econ-sum-net")).toHaveText("+$2,882");
  await expect(s.getByTestId("econ-needs")).toHaveCount(0);
});

test("a complete hypothetical scenario: net, break-even against the catch, the 3 × 3 grid and fuel ±20 %", async ({ page }) => {
  const s = await openWorksheet(page, SANTA_CRUZ);
  for (const [id, v] of [["econ-catch-0-lb", "400"], ["econ-catch-0-price", "9"], ["econ-nm", "40"], ["econ-kt", "8"], ["econ-hours", "10"], ["econ-cost-0-usd", "60"], ["econ-cost-1-usd", "40"]]) await s.getByTestId(id).fill(v);
  await s.getByTestId("econ-use-diesel").click();
  await s.getByTestId("econ-use-steamGph").click();
  await s.getByTestId("econ-use-fishingGph").click();
  await s.getByTestId("econ-crew-afterExpenses").click();
  await s.getByTestId("econ-crew-pct").fill("30");
  await expect(s.getByTestId("econ-sum-net")).toHaveText("+$2,017");
  await expect(s.getByTestId("econ-sum-cost")).toHaveText("$718");
  await expect(s.getByTestId("econ-sum-breakeven")).toHaveText("80 lb");
  await expect(s.getByTestId("econ-breakeven-text")).toContainText("320 lb above");
  await expect(s.getByTestId("econ-breakeven-text")).toContainText("$1.80/lb");
  await expect(s.locator(`[data-testid^="econ-sens-"]:not([data-testid="econ-sensitivity"])`)).toHaveCount(9);
  await expect(s.getByTestId("econ-sens-0-0")).toHaveText("+$2,017");
  await expect(s.getByTestId("econ-fuel-sens")).toContainText("Fuel price −20 %");
  await expect(formula(page, "econ-g-crew")).toContainText("Crew pay $865 = 30 % × $2,882");
});

test("recreational: cost only, no price, crew or net; the gallons route and an optional split", async ({ page }) => {
  const s = await openWorksheet(page, `${OK}/coast?place=port-593&for=recreational&species=salmon`);
  await expect(page.getByTestId("econ-rec-note")).toContainText("cost only");
  for (const id of ["econ-catch-0-lb", "econ-catch-0-price", "econ-sum-net", "econ-g-crew", "econ-waterfall"]) await expect(s.getByTestId(id), id).toHaveCount(0);
  await s.getByTestId("econ-fuel-mode-gallons").click();
  await s.getByTestId("econ-fuel-gal").fill("30");
  await s.getByTestId("econ-fuel-price").fill("5");
  await expect(s.getByTestId("econ-sum-cost")).toHaveText("$150");
  await s.getByTestId("econ-anglers").fill("3");
  await expect(s.getByTestId("econ-sum-person")).toHaveText("$50");
  expect(await s.innerText()).not.toMatch(/net to boat|crew share|\$\/lb/i);
});

test("scenarios: save, save as new, compare only the differing inputs with result deltas; kept across reloads", async ({ page }) => {
  const s = await openWorksheet(page);
  await fillDungeness(page);
  await s.getByTestId("econ-scen-name").fill("Opener");
  await s.getByTestId("econ-scen-save").click();
  await s.getByTestId("econ-scen-dup").click();
  await s.getByTestId("econ-catch-0-lb").fill("800");
  await s.getByTestId("econ-scen-save").click();
  await s.getByTestId("econ-cmp-a").selectOption({ label: "Opener" });
  await s.getByTestId("econ-cmp-b").selectOption({ label: "Opener (copy)" });
  const diff = s.getByTestId("econ-cmp-diff");
  await expect(diff).toHaveCount(1);
  await expect(diff).toContainText("Dungeness crab, lb");
  await expect(s.getByTestId("econ-cmp-delta-gross")).toHaveText("−$700");
  await expect(s.getByTestId("econ-cmp-delta-tripCost")).toHaveText("0");
  await page.reload();
  const s2 = page.getByTestId("trip-economics-section");
  await expect(s2.getByTestId("econ-sum-cost")).toHaveText("$490");
  await expect(s2.getByTestId("econ-scenarios")).toContainText("2 of 20");
});

test("a scenario link carries numbers only and opens unsaved in the worksheet", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const s = await openWorksheet(page);
  await fillDungeness(page);
  await s.getByTestId("econ-cost-0").getByRole("textbox").first().fill("Skipper Bob's bait");
  await s.getByTestId("econ-share-copy").click();
  const link = await s.getByTestId("econ-share-url").inputValue();
  expect(link).toContain("#trip=");
  const payload = Buffer.from(link.split("#trip=")[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
  expect(payload).not.toMatch(/Bob|Princeton|Half Moon/);

  const other = await context.newPage();
  await other.clock.setFixedTime(new Date(NOW));
  await other.goto(link.replace("&sheet=trip", ""));
  const o = other.getByTestId("trip-economics-section");
  await expect(o.getByTestId("econ-shared")).toContainText("From a shared link, not saved");
  await expect(o.getByTestId("econ-r-gross")).toHaveText("$3,500");
  await expect(o.getByTestId("econ-sum-net")).toHaveText("+$2,165");
  await o.getByTestId("econ-shared-save").click();
  await expect(o.getByTestId("econ-shared")).toHaveCount(0);
});

test("damaged trip data is kept and reported; delete all clears only trip data", async ({ page }) => {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem("seeded")) {
      localStorage.setItem("nx-trip-v1:scenarios", "{broken");
      sessionStorage.setItem("seeded", "1");
    }
  });
  const s = await openWorksheet(page);
  await expect(s.getByTestId("econ-store-status")).toContainText("could not be read");
  await s.getByTestId("econ-data").locator("summary").click();
  await s.getByTestId("econ-delete-all").click();
  await s.getByTestId("econ-delete-confirm").click();
  await expect(s.getByTestId("econ-data-msg")).toContainText("deleted");
  const keys = await page.evaluate(() => Object.keys(localStorage).filter((k) => k.startsWith("nx-trip-v1:")));
  expect(keys).toEqual([]);
  expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("research");
});

for (const width of [390, 320]) {
  test(`phone ${width}: results first, a result bar above the dock, no sideways scroll, 44 px targets, accessible`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    const s = await openWorksheet(page);
    // the decision order is the reading order: the result block before the first input
    expect((await s.getByTestId("econ-summary").boundingBox())!.y).toBeLessThan((await s.getByTestId("econ-catch-0-lb").boundingBox())!.y);
    await fillDungeness(page);
    await s.getByTestId("econ-g-crew").scrollIntoViewIfNeeded();
    const bar = page.getByTestId("econ-bar");
    await expect(bar).toBeVisible();
    await expect(bar).toContainText("+$2,165");
    await expect(bar).toContainText(/hypothetical/i);
    const box = (await bar.boundingBox())!;
    const dock = (await page.getByTestId("nav-dock").boundingBox())!;
    expect(Math.abs(box.y + box.height - dock.y)).toBeLessThanOrEqual(2);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    const small = await s.locator("button:visible, select:visible, input:visible:not([type=checkbox]):not([type=radio]):not([type=file])").evaluateAll((els) => els.map((e) => e.getBoundingClientRect()).filter((r) => r.height < 44).length);
    expect(small).toBe(0);
    const axe = await new AxeBuilder({ page }).include("#trip-worksheet").analyze();
    expect(axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical")).toEqual([]);
  });
}

test("the briefing's glance: fixed height before the device's numbers are read, then what they work out to", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(PRINCETON);
  const g = page.getByTestId("brief-econ");
  await expect(g.getByTestId("econ-kicker")).toHaveText("Scenario from your inputs, not a forecast.");
  await expect(g.getByTestId("econ-glance-empty")).toContainText("Nothing entered yet");
  await g.getByTestId("econ-open").click();
  await expect(page).toHaveURL(/sheet=trip/);
  await expect(page.getByTestId("econ-worksheet")).toBeVisible();
  await page.getByTestId("econ-catch-0-lb").fill("1000");
  await page.getByTestId("econ-catch-0-price").fill("3.5");
  await page.goBack();
  await expect(page.getByTestId("coast-title")).toContainText("Princeton");
  await expect(g.getByTestId("econ-glance-result")).toContainText("$3,500");
});

test("rock crab: CDFW's 2024 statewide value sits beside the price as history, copied in only on a tap", async ({ page }) => {
  const s = await openWorksheet(page, `${OK}/coast?place=port-550&for=commercial&species=rock-crab`);
  const ref = s.getByTestId("econ-catch-0-ref");
  await expect(ref).toContainText("CDFW history, not today’s price");
  await expect(ref).toContainText("$2.35/lb");
  await expect(ref).toContainText("Past average, not today’s dock price");
  await expect(s.getByTestId("econ-catch-0-price")).toHaveValue("");
  await s.getByTestId("econ-catch-0-use").click();
  await expect(s.getByTestId("econ-catch-0-price")).toHaveValue("2.35");
  await expect(ref).toContainText("basis: CDFW history");
});
