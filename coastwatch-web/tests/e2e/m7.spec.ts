import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** Fisheries & Economic Exposure (M7): one annual chart, the year beside it, one table that opens
 *  in place, a reference map without values, and the methods. Every number is checked against the
 *  fixture dataset. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const fish = JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith("fisheries-"))!), "utf8"));
type G = { id: string; tier: number; annual: { year: number; dollars_real: number; dollars_nominal: number; pounds: number }[] };
const groups = fish.groups as G[];
const val = (g: G, y: number, k: "dollars_real" | "dollars_nominal" | "pounds") => g.annual.find((a) => a.year === y)![k];
const M = (v: number) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${(v / 1e3).toFixed(0)}k`);
const LB = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M lb` : `${(v / 1e3).toFixed(0)}k lb`);

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(url);
  await expect(page.getByTestId("fisheries-page")).toBeVisible();
}

test.describe("Fisheries (M7)", () => {
  test("the story opens with the notices, the definition and three figures, then the chart", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    // the definition and the figures sit side by side on wide screens, so only their position
    // after the notices and before the chart is fixed
    const order = await page.evaluate(() =>
      ["fisheries-official", "fisheries-figures", "landings-chart", "species-table", "fisheries-limits", "port-level-unavailable", "fisheries-methods"].map(
        (id) => document.querySelector(`[data-testid="${id}"]`)!.getBoundingClientRect().top + window.scrollY,
      ),
    );
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    const top = async (id: string) => (await page.getByTestId(id).boundingBox())!.y;
    expect(await top("exposure-definition")).toBeGreaterThan(await top("fisheries-official"));
    expect(await top("exposure-definition")).toBeLessThan(await top("landings-chart"));
    await expect(page.locator("h1")).toHaveText("What California’s toxin\u2011affected fisheries have landed");
    await expect(page.getByTestId("fisheries-official")).toContainText("Official notices that name these species: 5");
    await expect(page.getByTestId("fisheries-official").getByTestId("official-verification")).toContainText("Not verified");
    // no KPI cards: the figures are type on the page
    await expect(page.getByTestId("fisheries-figures").locator(".rounded-lg")).toHaveCount(0);
  });

  test("bar totals equal the sum of the published group values, each year and unit", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const t1 = groups.filter((g) => g.tier === 1);
    for (const y of fish.years as number[]) await expect(page.getByTestId(`bar-total-${y}`)).toHaveText(M(t1.reduce((s, g) => s + val(g, y, "dollars_real"), 0)));
    await page.getByTestId("unit-pounds").click();
    await page.getByTestId("tier-12").click();
    for (const y of [2015, 2018, 2024]) await expect(page.getByTestId(`bar-total-${y}`)).toHaveText(LB(groups.reduce((s, g) => s + val(g, y, "pounds"), 0)));
    await expect(page.getByTestId("chart-unit")).toContainText("pounds landed");
    await expect(page.getByTestId("chart-notes")).toContainText("Bivalve pounds are meat weight");
    await expect(page.getByTestId("legend-bivalves")).toContainText("meat weight");
  });

  test("the 2015–16 closures are bracketed by date across the 2015 and 2016 bars, as context only", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const box = async (id: string) => (await page.getByTestId(id).boundingBox())!;
    const chart = await page.getByTestId("landings-chart").boundingBox();
    const [b15, b16] = [await box("year-2015"), await box("year-2016")];
    const dg = page.getByTestId("event-dungeness-2015-16");
    const x0 = Number(await dg.getAttribute("data-x0")) + chart!.x;
    const x1 = Number(await dg.getAttribute("data-x1")) + chart!.x;
    // 6 Nov 2015 is 85% through the 2015 slot; 26 May 2016 is 40% through the 2016 slot
    expect(x0).toBeGreaterThan(b15.x + b15.width * 0.8);
    expect(x0).toBeLessThan(b15.x + b15.width);
    expect(x1).toBeGreaterThan(b16.x + b16.width * 0.35);
    expect(x1).toBeLessThan(b16.x + b16.width * 0.45);
    await expect(dg).toContainText("Dungeness crab season delayed for domoic acid");
    await expect(page.getByTestId("event-rock-crab-2015-16")).toContainText("rock crab closed");
    await expect(page.getByTestId("chart-notes")).toContainText("do not say how much of any change");
    // selecting 2016 quotes the dated record with its source
    await page.getByTestId("year-2016").click();
    const ev = page.getByTestId("year-event");
    await expect(ev).toHaveCount(2);
    await expect(ev.first()).toContainText("26 May 2016");
    await expect(ev.first()).toContainText("not a measured effect");
    await expect(ev.first().getByRole("link")).toHaveAttribute("href", /fisheries\.legislature\.ca\.gov/);
  });

  test("a year is chosen from the chart or the keyboard; figures, panel, table and link follow", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await page.getByTestId("year-2016").click();
    await expect(page).toHaveURL(/year=2016/);
    const t1 = groups.filter((g) => g.tier === 1);
    const tot = t1.reduce((s, g) => s + val(g, 2016, "dollars_real"), 0);
    await expect(page.getByTestId("figure-year")).toContainText(M(tot));
    await expect(page.getByTestId("year-panel")).toHaveAttribute("data-year", "2016");
    await expect(page.getByTestId("year-total")).toContainText(M(tot));
    await expect(page.getByTestId("species-value-dungeness_crab")).toHaveText(M(val(groups[0], 2016, "dollars_real")));
    await page.getByTestId("year-2016").press("ArrowLeft");
    await expect(page.getByTestId("year-2015")).toBeFocused();
    await expect(page.getByTestId("year-2015")).toHaveAttribute("aria-pressed", "true");
    await page.keyboard.press("End");
    await expect(page.getByTestId(`year-${fish.years.at(-1)}`)).toHaveAttribute("aria-pressed", "true");
    await expect(page).not.toHaveURL(/year=/);
  });

  test("choosing a group shows it alone on its own scale and opens its row", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await page.getByTestId("legend-rock_crab").click();
    await expect(page).toHaveURL(/group=rock_crab/);
    await expect(page.locator("#chart-h")).toHaveText(/^Rock crabs: landed value by year/);
    await expect(page.getByTestId("chart-unit")).toContainText("alone, on its own scale");
    const shown = await page.getByTestId("landings-chart").locator("rect[data-group]").evaluateAll((r) => [...new Set(r.map((x) => x.getAttribute("data-group")))]);
    expect(shown).toEqual(["rock_crab"]);
    await expect(page.getByTestId("event-dungeness-2015-16")).toHaveCount(0);
    await expect(page.getByTestId("event-rock-crab-2015-16")).toHaveCount(1);
    await expect(page.getByTestId("species-open-rock_crab")).toHaveAttribute("aria-expanded", "true");
    await page.getByTestId("legend-all").click();
    await expect(page.getByTestId("landings-chart")).toHaveAttribute("data-focus", "");
  });

  test("a group's detail: why it is here, its notices (not verified), revenue and pounds on separate scales", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await expect(page.getByTestId("species-row-bivalves")).toHaveAttribute("data-dim", "true");
    await page.getByTestId("species-open-bivalves").click();
    await expect(page).toHaveURL(/tiers=all/);
    await expect(page).toHaveURL(/group=bivalves/);
    await expect(page.getByTestId("species-row-bivalves")).toHaveAttribute("data-dim", "false");
    const d = page.getByTestId("species-detail-bivalves");
    await expect(d).toContainText("mostly aquaculture");
    await expect(page.getByTestId("species-notices-bivalves")).toContainText("not verified by a person");
    await expect(page.getByTestId("species-notices-bivalves")).toContainText("Annual quarantine of sport-harvested mussels");
    await expect(page.getByTestId("detail-revenue-bivalves")).toContainText("Revenue, 2024 dollars");
    await expect(page.getByTestId("detail-pounds-bivalves")).toContainText("pounds of meat (without shell)");
    await expect(d).toContainText("separate measures on separate scales");
    await expect(d).toContainText("Rows without a published value (left out, not zero)");
    await page.getByRole("button", { name: "Read the notices ›" }).click();
    await expect(page.getByTestId("official-drawer")).toBeVisible();
  });

  test("the table sorts with announced order", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const ids = () => page.locator('[data-testid^="species-row-"]').evaluateAll((r) => r.map((x) => x.getAttribute("data-testid")!.replace("species-row-", "")));
    const byLatest = [...groups].sort((a, b) => val(b, 2024, "dollars_real") - val(a, 2024, "dollars_real")).map((g) => g.id);
    expect(await ids()).toEqual(byLatest);
    await page.getByTestId("sort-name").click();
    await expect(page.locator('th[aria-sort="ascending"]')).toContainText("Species group");
    expect((await ids())[0]).toBe("bivalves");
    await page.getByTestId("sort-mean").click();
    await expect(page.locator('th[aria-sort="descending"]')).toContainText("10-yr avg");
  });

  test("a deep link restores the view; unknown parameters fall back", async ({ page }) => {
    await open(page, `${OK}/fisheries?unit=nominal&tiers=all&group=spiny_lobster&year=2019`);
    await expect(page.getByTestId("unit-nominal")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("tier-12")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("year-2019")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("#chart-h")).toHaveText(/^Spiny lobster/);
    await open(page, `${OK}/fisheries?unit=x&tiers=y&group=z&year=1`);
    await expect(page.getByTestId("unit-real")).toHaveAttribute("aria-checked", "true");
    await expect(page.getByTestId("year-2024")).toHaveAttribute("aria-pressed", "true");
  });

  test("where landings are reported: CDFW points by port area, no values, no boundaries", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const map = page.getByTestId("port-area-map");
    // the map and its points load only when the section nears the screen
    await expect(map).toHaveCount(0);
    await page.locator("#geo-h").scrollIntoViewIfNeeded();
    await expect(map).toBeVisible();
    expect(await map.locator("circle").count()).toBeGreaterThan(100);
    await expect(map.locator("text")).toContainText(["Eureka", "Fort Bragg", "Bodega Bay", "San Francisco", "Monterey", "Morro Bay", "Santa Barbara", "Los Angeles", "San Diego"]);
    expect(await map.locator("svg").textContent()).not.toMatch(/\$|\blb\b|%/);
    await expect(map).toContainText("no area boundaries");
    await expect(map.getByRole("link", { name: /CDFW Marine Region/ })).toHaveAttribute("href", /gis\.data\.ca\.gov/);
    await expect(page.getByTestId("port-level-unavailable")).toContainText("does not split the statewide totals");
  });

  test("what the figures can and cannot tell you", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const l = page.getByTestId("fisheries-limits");
    await expect(l).toContainText("an overlap in time does not show a cause");
    await expect(l).toContainText("Profit or income");
    await expect(l).toContainText("Fishing effort");
    await expect(page.getByTestId("dollars-note")).toContainText("not seafood prices or fishing costs");
  });

  test("with every disclosure open, no sentence claims safety, openness or losses unless negated", async ({ page }) => {
    await open(page, `${OK}/fisheries?year=2016&group=rock_crab`);
    for (const d of await page.locator("details").all()) await d.evaluate((n) => ((n as HTMLDetailsElement).open = true));
    const text = await page.locator("main").innerText();
    const bad = text
      .split(/(?<=[.!?])\s+|\n+/)
      .filter((s) => /\b(safe|open|all clear|losse?s?|caused?)\b/i.test(s) && !/\b(not|no|never|isn't)\b/i.test(s) && !/Whale.?safe/i.test(s));
    expect(bad, bad.join(" | ")).toEqual([]);
  });

  test("no serious accessibility violations with a detail open, in pounds", async ({ page }) => {
    await open(page, `${OK}/fisheries?group=bivalves&unit=pounds`);
    await page.waitForLoadState("networkidle");
    const r = await new AxeBuilder({ page }).analyze();
    expect(r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => v.id)).toEqual([]);
  });
});

test.describe("Fisheries (M7) on a phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  test("tap a bar to choose its year; rows open in place; nothing overflows", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    const svg = page.getByTestId("landings-chart").locator("svg");
    await svg.scrollIntoViewIfNeeded();
    const b = (await svg.boundingBox())!;
    const band = (b.width - 54) / 10;
    await page.touchscreen.tap(b.x + 50 + band * 1.5, b.y + b.height * 0.85);
    await expect(page.getByTestId("year-2016")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("year-2016")).toHaveText("’16");
    await page.getByTestId("species-open-dungeness_crab").tap();
    await expect(page.getByTestId("species-detail-dungeness_crab")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  });
});

test.describe("Fisheries (M7) before the chart is measured", () => {
  // Without JavaScript the chart never measures its width, which is the state every phone sees for
  // the first moments of a visit: nothing may overflow, and the server-rendered story is all there.
  test.use({ viewport: { width: 390, height: 844 }, javaScriptEnabled: false });
  test("the server-rendered page fits a phone and carries the story", async ({ page }) => {
    await page.goto(`${OK}/fisheries`);
    await expect(page.getByTestId("landings-chart")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    await expect(page.getByTestId("exposure-definition")).toContainText("not a prediction of losses");
    await expect(page.getByTestId("figure-year")).toContainText("$");
  });
});
