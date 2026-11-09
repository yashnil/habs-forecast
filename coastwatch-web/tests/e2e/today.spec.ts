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
      // no cloud credentials on the e2e servers: said plainly, with what does work
      await expect(page.getByTestId("readiness-alerts")).toHaveText("Not yet connected: accounts, push and text. Alerts work in the app on this device.");
      await expect(page.getByTestId("readiness-ask")).toContainText("Pilot demo: answers from the facts on this page");
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

test("every fact has an address: a citation link lands on its card and marks it", async ({ page }) => {
  // a fact in a card on the page (the Dungeness crab test)
  await page.goto(`${MOSS}#fact-${encodeURIComponent("test:bma-g-monterey-bay-dungeness-crab")}`);
  await ready(page);
  const hit = page.locator('[data-fact-hit="test:bma-g-monterey-bay-dungeness-crab"]');
  await expect(hit).toBeVisible();
  await expect(hit).toHaveAttribute("data-testid", "test-bma-g-monterey-bay-dungeness-crab");
  await expect(hit).toBeInViewport();
  // a fishery the reader does not follow (recreational Dungeness is not in the default list): shown for this visit
  await page.goto(`${MOSS}#fact-${encodeURIComponent("status:dungeness_recreational")}`);
  await ready(page);
  const row = page.locator('[data-fact-hit="status:dungeness_recreational"]');
  await expect(row).toBeInViewport();
  await expect(row.getByTestId("status-dungeness_recreational")).toContainText("crab traps prohibited");
  expect(await page.evaluate(() => localStorage.getItem("nx-today-fisheries-v1"))).toBeNull();
  // a forecast period inside the sea lanes marks the forecast, once the sea has loaded
  await page.goto(`${MOSS}#fact-${encodeURIComponent("weather:nws:2026-10-13:day")}`);
  await ready(page);
  await expect(page.locator('[data-fact-hit="weather:nws:2026-10-13:day"]')).toContainText("W 10–15 kt");
  // every fact id on the page is an address, each exactly once
  const ids = await page.locator('[id^="fact-"]').evaluateAll((els) => els.map((e) => e.id));
  expect(new Set(ids).size).toBe(ids.length);
  for (const id of ["fact-status:dungeness_commercial", "fact-whale:ramp:zone-4", "fact-charm:pda", "fact-outlook:chl8d", "fact-weather:nws", "fact-tide:next", "fact-sea:buoy", "fact-age:data", "fact-notice:cdph-2026-annual-mussel-quarantine"]) expect(ids).toContain(id);
});

test("Ask Neritic opens in place; a citation closes the sheet and lands on the card", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  await page.getByTestId("action-ask").click();
  const sheet = page.getByTestId("ask-sheet");
  await expect(sheet).toBeVisible();
  await expect(page).toHaveURL(/\/today\?place=port-592$/);
  await expect(sheet.getByRole("heading", { level: 2, name: "Ask Neritic" })).toBeVisible();
  await expect(sheet.getByTestId("ask-readiness")).toContainText("Pilot demo");
  await sheet.getByTestId("ask-suggestions").getByRole("button", { name: "What are the whale rules?" }).click();
  const a = sheet.getByTestId("ask-answer");
  await expect(a).toContainText("RAMP Zone 4");
  expect(await axe(page)).toEqual([]);
  await a.getByTestId("ask-chip").first().click();
  await a.getByTestId("ask-see-today").first().click();
  await expect(sheet).toBeHidden();
  await expect(page).toHaveURL(/#fact-whale%3Aramp%3Azone-4$/);
  await expect(page.locator('[data-fact-hit="whale:ramp:zone-4"]')).toBeInViewport();
  await expect(page.locator('[data-fact-hit="whale:ramp:zone-4"]').getByTestId("whale-lede")).toBeVisible();
});

test("each section asks once whether it was useful; the answer is kept on this device", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  const asks = page.getByTestId("feedback");
  await expect(asks).toHaveCount(5);
  expect(await asks.evaluateAll((els) => els.map((e) => e.getAttribute("data-fact")))).toEqual(["status:dungeness_commercial", "weather:nws", "test:bma-g-monterey-bay-dungeness-crab", "charm:pda", "whale:ramp:zone-4"]);
  const tests = page.getByTestId("today-tests").getByTestId("feedback");
  await tests.getByTestId("feedback-yes").click();
  await expect(tests).toContainText("Thanks — saved on this device.");
  await page.reload();
  await ready(page);
  await expect(page.getByTestId("today-tests").getByTestId("feedback-yes")).toHaveAttribute("aria-pressed", "true");
});

test("the top bar carries the alerts bell and the pilot's pages; Today runs the device's alert check", async ({ page }) => {
  // a device that follows Moss Landing
  await page.addInitScript(() => {
    if (sessionStorage.getItem("seeded")) return;
    sessionStorage.setItem("seeded", "1");
    localStorage.setItem("nrt-pilot-account-v1", JSON.stringify({ v: 1, role: null, fisheries: [], home_place_id: "port-592", vessel_length_class: null, age_13_plus: null, saved_places: ["port-592"], alerts: [{ place_id: "port-592", triggers: ["status_change", "new_toxin_result", "ramp_action"], channels: ["in_app"] }], sms: null, created_at: "2026-10-12T15:00:00.000Z" }));
  });
  await page.goto(MOSS);
  await ready(page);
  await expect(page.getByTestId("alerts-bell")).toHaveAccessibleName("Alerts");
  await page.getByTestId("more-menu").click();
  await expect(page.getByTestId("more-account")).toHaveAttribute("href", "/account");
  await expect(page.getByTestId("more-ask")).toHaveAttribute("href", "/ask");
  await page.keyboard.press("Escape");
  // the first visit kept a baseline; pretend it saw an older whale-rule action, then come back
  await page.waitForFunction(() => !!JSON.parse(localStorage.getItem("nrt-pilot-alert-snap-v1") ?? "{}")["port-592"]);
  await page.evaluate(() => {
    const k = "nrt-pilot-alert-snap-v1";
    const s = JSON.parse(localStorage.getItem(k)!);
    const id = "whale:ramp:zone-4";
    s["port-592"].facts[id] = { v: "00000000", sig: { ...s["port-592"].facts[id].sig, headline: "Whale rules, RAMP Zone 4: no action on record." } };
    localStorage.setItem(k, JSON.stringify(s));
  });
  await page.reload();
  await ready(page);
  await expect(page.getByTestId("alerts-unread")).toHaveText("1");
  await page.getByTestId("alerts-bell").click();
  const panel = page.getByTestId("alerts-panel");
  await expect(panel.getByTestId("inbox-item")).toContainText("Moss Landing: Whale rules, RAMP Zone 4");
  // made from the test fixture: marked, never passed off as live
  await expect(panel.getByTestId("inbox-sample")).toContainText("Sample data");
  expect(await axe(page)).toEqual([]);
});

test("known gaps are said plainly: a wave buoy's missing wind comes from the next buoy, and sources that differ say so", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  // the nearest buoy (46236) has no anemometer: the wind is from the nearest that has one, named with its distance
  await expect(page.getByTestId("wind-now")).toContainText("6");
  await expect(page.getByTestId("wind-where")).toHaveText("at buoy 46092, 22 km");
  await expect(page.getByTestId("today-sea")).toContainText("it has no wind sensor, so wind is from buoy 46092, 22 km away");
  await expect(page.getByTestId("sea-now")).toContainText("4.9");
  // mussels: the notices registry lists the quarantine, the curated status says CDPH announced its end.
  // The row keeps the more restrictive level and flags the disagreement; the sheet shows both, each with its date
  const row = page.getByTestId("status-bivalve_shellfish");
  await expect(row).toHaveAttribute("data-severity", "red");
  await expect(row.getByTestId("sources-differ")).toBeVisible();
  await row.click();
  const sheet = page.getByTestId("status-sheet");
  await expect(sheet.getByTestId("conflict")).toContainText("It shows the more restrictive one and does not choose between them");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("Closed: Annual quarantine of sport-harvested mussels");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("in effect since May 1 · registry last read Oct 8 · not verified");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("shown");
  await expect(sheet.getByTestId("conflict-record")).toContainText("Watch: CDPH announced the 2026 mussel quarantine ended Oct 31");
  await expect(sheet.getByTestId("conflict-record")).toContainText("checked Oct 12 · not verified");
  expect(await axe(page)).toEqual([]);
  await page.keyboard.press("Escape");
  // a fishery whose sources agree carries no such flag
  await expect(page.getByTestId("status-anchovy_sardine").getByTestId("sources-differ")).toHaveCount(0);
});

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
  // a page that was never saved: the offline page offers the saved Today by the port's name
  await page.goto(`${OK}/research`);
  await expect(page.getByTestId("offline-page")).toBeVisible();
  const saved = page.getByTestId("offline-today");
  await expect(saved).toContainText("Open Today at Moss Landing");
  await expect(saved).toContainText("as it was then");
  await saved.click();
  await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
  await expect(page.getByTestId("today-saved-copy")).toBeVisible();
  await ctx.setOffline(false);
  await ctx.close();
});
