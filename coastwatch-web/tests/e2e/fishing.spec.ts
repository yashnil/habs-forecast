import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * v5 fishermen audit: the trip briefing answers "is it in season here, what is the sea doing, what has
 * it been worth" at a glance, each answer quoted from its agency with its date, and never as Neritic's
 * verdict (no "open to fish", no "safe", no go/no-go).
 */
const NOW = "2026-10-13T06:00:00Z";
const VERDICTS = /\b(safe to (eat|go|fish|harvest)|open to fish|all clear|good to go|no-go\b(?! from Neritic)|best spot)/i;

test("at a glance: rows for the fishery, the sea, the trip and the gaps, each one tap from its evidence", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=recreational&species=dungeness`);
  const g = page.getByTestId("brief-glance");
  await expect(g).toBeVisible();
  await expect(g.getByTestId("glance-season")).toContainText("Dungeness Crab");
  await expect(g.getByTestId("glance-season-list")).toHaveText(/on CDFW.s closed list/);
  await expect(g.getByTestId("glance-official")).toContainText("scheduled to reopen on November 7, 2026");
  await expect(g.getByTestId("glance-notices")).toContainText("not verified");
  // v6 P0 3: the dock value is history, labelled so, and the species' own fishery's only
  await expect(g.getByTestId("glance-worth")).toContainText("Dock value, 2024 (history)");
  await expect(g.getByTestId("glance-worth")).toContainText("Dungeness Crab");
  await expect(g.getByTestId("glance-worth")).toContainText("Not today's price");
  for (const [tile, target] of [["glance-season", "#seasons-h"], ["glance-sea-link", "#sea-h"], ["glance-notices", "#official-h"], ["glance-worth", "#worth-h"], ["glance-econ", "#trip-economics"], ["glance-unknown", "#unknown-h"]] as const) {
    await expect(g.getByTestId(tile)).toHaveAttribute("href", target);
    await expect(page.locator(target)).toHaveCount(1);
  }
  expect(await page.locator("main").innerText()).not.toMatch(VERDICTS);
});

test("v6 P0 2: a commercial crabber never sees the sport reopen date as the season", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
  const o = page.getByTestId("glance-official");
  await expect(o).toContainText("Sport list only. Commercial opener: see CDFW commercial / in-season page");
  await expect(o).not.toContainText("November 7");
  await expect(o.getByTestId("glance-season-list")).toHaveCount(0);
  await expect(o.getByRole("link", { name: "CDFW commercial" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Commercial");
  await expect(o.getByRole("link", { name: "in-season page" })).toHaveAttribute("href", /wildlife\.ca\.gov\/Fishing\/Ocean\/Regulations\/inseason/);
  // the sport row in Seasons stays closed for a commercial reader, marked "(sport)"
  await expect(page.getByTestId("seasons-mine").locator('li[data-testid^="season-dungeness"] details').first()).not.toHaveAttribute("open", "");
});

test("v6 P0 3: a species with no fishery of its own shows no neighbour's value and no port total", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  for (const sp of ["tunas", "groundfish"]) {
    await page.goto(`${OK}/coast?place=port-550&species=${sp}`);
    const w = page.getByTestId("glance-worth");
    await expect(w).toContainText("no species-level reference");
    await expect(w).not.toContainText(/Bluefin|Bigeye|Sablefish|Petrale|port area/);
  }
});

test("seasons quote CDFW's region page with its date, the reader's species first", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&species=groundfish`);
  const s = page.getByTestId("brief-seasons");
  await expect(s).toContainText("Central");
  await expect(s).toContainText("October 13, 2026");
  // v6: one source mark here; the page-level disclosure says no person has reviewed Neritic's copy
  await expect(s.getByTestId("seasons-source")).toContainText("Neritic’s copy of");
  await expect(page.getByTestId("matters-provenance")).toContainText("No person has reviewed that copy yet");
  const mine = s.getByTestId("seasons-mine");
  await expect(mine.getByTestId("season-rockfish")).toHaveAttribute("data-mark", "open");
  // CDFW's groundfish table for the trip day, quoted
  await expect(mine.getByTestId("season-groundfish-row").first()).toContainText("Open all depths");
  await expect(s.getByRole("link", { name: /CDFW Central region page/ })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Regulations/Fishing-Map/central");
  // every entry CDFW lists stays one tap away
  await s.getByTestId("seasons-all").locator("summary").first().click();
  expect(await s.getByTestId("seasons-all").locator("li").count()).toBeGreaterThan(20);
  // a commercial-only species says the sport list is not its rules
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=squid`);
  await expect(page.getByTestId("seasons-commercial")).toContainText("sport fishing");
});

test("sea conditions: the forecaster's words, and the reader's own limits compared with them", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  const sea = page.getByTestId("brief-sea");
  await expect(sea.getByTestId("sea-period").first()).toBeVisible();
  await expect(sea).toContainText("never a go or no-go from Neritic");
  // the trip day's period is matched by its own date: Monday night's "Tonight" for a Monday trip
  await expect(sea.getByTestId("sea-period").first()).toHaveAttribute("data-chosen", "true");
  await expect(sea.getByTestId("sea-over")).toHaveCount(0);
  await sea.getByTestId("sea-limits-edit").click();
  await sea.getByTestId("sea-limit-wind").fill("1");
  await expect(sea.getByTestId("sea-over").first()).toContainText("above your own limit");
  await expect(page.getByTestId("glance-over")).toContainText("1 kt");
  // kept on this device only
  await page.reload();
  await expect(page.getByTestId("brief-sea").getByTestId("sea-limits")).toContainText("wind 1 kt");
  await page.getByTestId("sea-limits-edit").click();
  await page.getByTestId("sea-limit-wind").fill("");
  await expect(page.getByTestId("sea-over")).toHaveCount(0);
  // the same limits are set in place from the glance
  await page.getByTestId("glance-limits-set").click();
  await page.getByTestId("glance-limit-seas").fill("2");
  await expect(page.getByTestId("glance-over")).toContainText("2 ft");
});

test("what it was worth: CDFW's dock values with year and source, history not price", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-452&species=dungeness`);
  await page.getByTestId("brief-worth-toggle").click();
  const w = page.getByTestId("worth");
  await expect(w.getByTestId("worth-species")).toContainText("2024");
  await expect(w.getByTestId("worth-species")).toContainText("2023");
  await expect(w.getByTestId("worth-port-area")).toContainText("San Francisco port area");
  await expect(w).toContainText("History, not this season");
  await expect(w.getByRole("link", { name: "CDFW Marine Region, 2024 By The Numbers" })).toHaveAttribute("href", /nrm\.dfg\.ca\.gov/);
  // a species without a CDFW figure says so, never a number
  await page.goto(`${OK}/coast?place=port-452&species=salmon`);
  await page.getByTestId("brief-worth-toggle").click();
  await expect(page.getByTestId("worth-species-none")).toContainText("No species-level reference");
});

test("species first: the landing asks what you fish and carries it into the briefing", async ({ page }) => {
  await page.goto(`${OK}/coast`);
  await page.getByTestId("coast-pick-halibut").click();
  await expect(page.getByTestId("coast-pick-halibut")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("coast-start-port-550")).toHaveAttribute("href", /species=halibut/);
  await page.getByTestId("coast-start-port-550").click();
  await expect(page.getByTestId("trip-species")).toHaveValue("halibut");
  await expect(page.getByTestId("seasons-mine").getByTestId("season-halibut")).toBeVisible();
});

test("fisheries: each species shows CDFW's lists region by region and its dock value", async ({ page }) => {
  // the species panel is the Coastal layout (the suite's stored mode is Research)
  await page.addInitScript(() => window.localStorage.setItem("cw-mode-v1", "coastal"));
  await page.goto(`${OK}/fisheries`);
  await page.getByTestId("fish-species-groundfish").click();
  const f = page.getByTestId("fish-seasons");
  for (const r of ["northern", "mendocino", "san-francisco", "central", "southern"]) await expect(f.getByTestId(`fish-season-${r}`)).toBeVisible();
  await expect(f.getByTestId("fish-season-southern")).toContainText("alert");
  await page.getByTestId("fish-species-dungeness").click();
  await expect(page.getByTestId("fish-worth")).toContainText("2024: Dungeness Crab $49.7M");
});

test("phone: the glance fits at 390 and 320 without sideways scroll, and passes axe", async ({ browser }) => {
  for (const width of [390, 320]) {
    const ctx = await browser.newContext({ viewport: { width, height: 760 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=port-201&for=recreational&species=dungeness`);
    await expect(page.getByTestId("brief-glance")).toBeVisible();
    const box = await page.getByTestId("glance-season").boundingBox();
    expect(box!.y).toBeLessThan(760);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    const r = await new AxeBuilder({ page }).include('[data-testid="brief-glance"]').include('[data-testid="brief-seasons"]').include('[data-testid="brief-sea"]').analyze();
    const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
    await ctx.close();
  }
});
