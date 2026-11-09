import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { M2, OK } from "./ports";

/**
 * Today at your port (Neritic private pilot): /today on fixture data at phone and desktop widths.
 * Official status first, never a green "safe"; measured tests as dots against the limit; the sea from
 * recorded NOAA answers; the bloom outlook with its caveats; whale rules; what changed; offline save;
 * stale and missing-data states; accessibility (axe: no violations at all).
 */
const MOSS = `${OK}/today?place=port-592`;
const SAFETY = /\b(safe|all clear|good to go|green light)\b/i;

async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

async function ready(page: Page) {
  await expect(page.getByTestId("today-port")).toBeVisible();
  // the sea has loaded (recorded NOAA answers)
  await expect(page.getByTestId("sea-lanes")).toBeVisible({ timeout: 15_000 });
}

for (const [label, viewport] of [["phone", { width: 390, height: 844 }], ["desktop", { width: 1440, height: 900 }]] as const) {
  test.describe(label, () => {
    test.use({ viewport, ...(label === "phone" ? { isMobile: true, hasTouch: true } : {}) });

    test("answers first: the port, the official status, then the sea, tests, bloom and whale rules", async ({ page }) => {
      await page.goto(MOSS);
      await ready(page);
      await expect(page.locator("h1")).toHaveCount(1);
      await expect(page.locator("h1")).toHaveText("Moss Landing");
      // the lede leads with the official delay, in words, with its cause
      await expect(page.getByTestId("today-lede")).toContainText("Dungeness crab (commercial)");
      await expect(page.getByTestId("today-lede")).toContainText("humpback whale entanglements");
      // the status strip: red delay first, the calmest state is "None on record", records not verified
      const dungeness = page.getByTestId("status-dungeness_commercial");
      await expect(dungeness).toHaveAttribute("data-severity", "red");
      await expect(dungeness.getByTestId("status-word")).toHaveText("Delayed");
      await expect(dungeness.getByTestId("not-verified")).toBeVisible();
      await expect(dungeness).toContainText("Next review Dec 15");
      await expect(page.getByTestId("status-groundfish").getByTestId("status-word")).toHaveText("None on record");
      await expect(page.getByTestId("status-groundfish")).toHaveAttribute("data-severity", "neutral");
      await expect(page.getByTestId("status-anchovy_sardine")).toHaveAttribute("data-severity", "amber");
      // the order on the page follows the hierarchy
      const order = await page.locator("[data-area]").evaluateAll((els) => els.map((e) => e.getAttribute("data-area")));
      expect(order).toEqual(["status", "sea", "tests", "bloom", "whale", "changed", "actions"]);
      // the sea: forecast lanes (a table), measured buoy, predicted tide
      await expect(page.getByTestId("sea-lanes").locator("thead th")).toHaveCount(10);
      await expect(page.getByTestId("sea-now")).toContainText("4.9");
      await expect(page.getByTestId("tide-curve")).toBeVisible();
      // the tests: a plain sentence first, then the chart with a text alternative
      await expect(page.getByTestId("tests-lede")).toHaveText("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
      await expect(page.getByTestId("toxin-chart").first()).toHaveAttribute("role", "img");
      // the bloom outlook: C-HARM as conditions, the outlook as an indicator with skill from the data file
      await expect(page.getByTestId("bloom-lede")).toContainText("probability of elevated particulate domoic acid in the water");
      await expect(page.getByTestId("outlook-words")).toContainText("a bloom indicator, not a toxin forecast");
      await expect(page.getByTestId("outlook-words")).toContainText("0.754");
      await expect(page.getByTestId("bloom-timeline")).toBeVisible();
      // whale rules
      await expect(page.getByTestId("whale-lede")).toContainText("Zone 4");
      await expect(page.getByTestId("whale-next")).toContainText("Dec 15");
      await expect(page.getByTestId("whale-next")).toContainText("December 15-18");
      // honesty: nothing is called safe, and the actions carry their readiness labels
      expect(await page.locator("main").innerText()).not.toMatch(SAFETY);
      await expect(page.getByTestId("readiness-alerts")).toContainText(/Not yet connected|Pilot demo|Live/);
      await expect(page.getByTestId("action-ask")).toHaveAttribute("href", "/ask?place=port-592");
      await expect(page.getByTestId("action-alert")).toHaveAttribute("href", "/account?place=port-592");
      expect(await axe(page)).toEqual([]);
    });

    test("Today is the first workspace in the navigation", async ({ page }) => {
      await page.goto(MOSS);
      await expect(page.getByTestId("today-port")).toBeVisible();
      const nav = page.getByTestId(label === "phone" ? "nav-dock" : "nav-bar");
      await expect(nav.locator("a").first()).toHaveText("Today");
      await expect(nav.getByTestId("nav-today")).toHaveAttribute("aria-current", "page");
      if (label === "phone") {
        const boxes = await nav.locator("a").evaluateAll((as) => as.map((a) => a.getBoundingClientRect()).map((b) => [b.width, b.height]));
        expect(boxes).toHaveLength(4);
        for (const [w, h] of boxes) expect(w >= 44 && h >= 44).toBe(true);
      }
    });
  });
}

test("a status row opens its detail sheet; Escape closes it and returns focus", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  const row = page.getByTestId("status-dungeness_commercial");
  await row.click();
  const sheet = page.getByTestId("status-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByRole("heading", { level: 2 })).toHaveText("Dungeness crab (commercial)");
  await expect(sheet).toContainText("Not verified: no person has checked this record");
  await expect(sheet).toContainText("Oregon border to Point Conception");
  expect(await axe(page)).toEqual([]);
  await page.keyboard.press("Escape");
  await expect(sheet).toBeHidden();
  await expect(row).toBeFocused();
});

test("the reader's fisheries: edit the list, kept on this device", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  await page.getByTestId("fisheries-edit").click();
  await page.getByTestId("pick-market_squid").uncheck();
  await page.getByTestId("pick-spiny_lobster").check();
  await page.getByTestId("fisheries-save").click();
  await expect(page.getByTestId("status-market_squid")).toHaveCount(0);
  await expect(page.getByTestId("status-spiny_lobster")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("status-spiny_lobster")).toBeVisible();
  await expect(page.getByTestId("status-market_squid")).toHaveCount(0);
  // a link's own list wins
  await page.goto(`${MOSS}&f=groundfish`);
  await expect(page.getByTestId("status-strip").locator("li")).toHaveCount(1);
});

test("what changed: the first visit says so; the next compares by fact", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  await expect(page.getByTestId("changed-first")).toBeVisible();
  // pretend the last visit saw no anchovy restriction and an older C-HARM run
  await page.evaluate(() => {
    const k = "nx-today-seen-v1:port-592";
    const s = JSON.parse(localStorage.getItem(k)!);
    s.facts = s.facts.filter((f: { id: string }) => f.id !== "status:anchovy_sardine").map((f: { id: string; as_of: string }) => (f.id === "charm:pda" ? { ...f, as_of: "2026-10-01" } : f));
    localStorage.setItem(k, JSON.stringify(s));
  });
  await page.reload();
  await ready(page);
  await expect(page.getByTestId("changed-summary")).toContainText("2 changes");
  await expect(page.getByTestId("changed-list")).toContainText("Anchovy & sardine");
});

test("no port: a picker with the pilot ports first; then Today reopens the last port", async ({ page }) => {
  await page.goto(`${OK}/today`);
  await expect(page.getByTestId("today-picker")).toBeVisible();
  await expect(page.locator("h1")).toHaveText("Choose your port");
  const pilots = page.locator('[data-testid^="pick-port-"]');
  await expect(pilots).toHaveCount(4);
  await expect(pilots.first()).toContainText("Moss Landing");
  expect(await axe(page)).toEqual([]);
  await page.getByTestId("port-filter").fill("Crescent");
  await expect(page.getByRole("link", { name: /Crescent City/ })).toBeVisible();
  await pilots.nth(2).click();
  await expect(page.locator("h1")).toHaveText("Santa Cruz");
  await page.goto(`${OK}/today`);
  await expect(page).toHaveURL(/\/today\?place=port-593$/);
  await expect(page.getByTestId("nav-bar").getByTestId("nav-today")).toHaveAttribute("href", "/today?place=port-593");
  // an unknown port is said plainly
  await page.goto(`${OK}/today?place=port-99999`);
  await expect(page.getByTestId("today-unknown-port")).toContainText("port-99999");
});

test("stale data is flagged in place: an old C-HARM run and its freshness", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  await expect(page.getByTestId("charm-stale")).toContainText("C-HARM run of Oct 8");
  await expect(page.getByTestId("today-freshness")).toContainText("C-HARM run Oct 8 (old)");
});

test("missing sidecars: tests, whale rules, status and outlook say they are unavailable, never empty or open", async ({ page }) => {
  await page.goto(`${M2}/today?place=port-592`);
  await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
  await expect(page.getByTestId("tests-unavailable")).toBeVisible();
  await expect(page.getByTestId("whale-unavailable")).toBeVisible();
  await expect(page.getByTestId("outlook-missing")).toBeVisible();
  await expect(page.getByTestId("today-lede")).toContainText("Fishery status could not be loaded");
  await expect(page.getByTestId("status-dungeness_commercial").getByTestId("status-word")).toHaveText("Status not available");
  await expect(page.getByTestId("status-strip")).not.toContainText("None on record");
  expect(await axe(page)).toEqual([]);
});

test("save for offline: the page and its sea reopen with no connection, marked as a saved copy", async ({ browser }) => {
  test.setTimeout(90_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(MOSS);
  await ready(page);
  await page.getByTestId("save-offline").click();
  await expect(page.getByTestId("save-status")).toContainText("Saved", { timeout: 30_000 });
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await ctx.setOffline(true);
  await page.goto(MOSS);
  await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
  await expect(page.getByTestId("today-saved-copy")).toContainText("Saved copy");
  await expect(page.getByTestId("status-dungeness_commercial")).toBeVisible();
  await expect(page.getByTestId("sea-lanes")).toBeVisible({ timeout: 15_000 });
  await ctx.setOffline(false);
  await ctx.close();
});
