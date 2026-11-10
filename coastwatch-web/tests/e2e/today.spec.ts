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

/** A Fold's own trigger (its body may hold buttons of its own). */
const foldTrigger = (page: Page, id: string) => page.getByTestId(id).locator("button[aria-expanded]").first();

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
      // (the sector is said once: "Dungeness crab: commercial season delayed …")
      await expect(page.getByTestId("today-lede")).toContainText("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
      await expect(page.getByTestId("today-lede")).not.toContainText("(commercial): Commercial");
      // the status strip: red delay first, the calmest state is "None on record", records not verified
      const dungeness = page.getByTestId("status-dungeness_commercial");
      await expect(dungeness).toHaveAttribute("data-severity", "red");
      await expect(dungeness.getByTestId("status-word")).toHaveText("Delayed");
      // the lede quotes the record, so it says "Not verified" itself, with whose record and when it was read
      const cue = page.getByTestId("lede-cue");
      await expect(cue).toContainText("CDFW · checked Oct 12");
      await expect(cue.getByTestId("lede-not-verified")).toBeVisible();
      await expect(cue.getByTestId("lede-not-verified")).toBeInViewport();
      // the group header counts what is not verified, including rows with no record at all (7 fisheries + the RAMP record)
      await expect(page.getByTestId("status-not-verified")).toHaveText("8 of 8 not verified");
      // "none on record" collapses to one line that carries its own caveat, on that line
      await expect(page.getByTestId("status-none-line")).toContainText("None on record for rock crab, groundfish and market squid. Not a promise they are open.");
      await foldTrigger(page, "status-none").click();
      await expect(page.getByTestId("status-groundfish").getByTestId("status-word")).toHaveText("None on record");
      await expect(page.getByTestId("status-groundfish")).toContainText("Not a promise it is open.");
      await expect(page.getByTestId("status-groundfish")).toHaveAttribute("data-severity", "neutral");
      await expect(page.getByTestId("status-anchovy_sardine")).toHaveAttribute("data-severity", "amber");
      // the sources differ on the sport-harvest quarantine: never stated as settled, in the row or anywhere it is counted
      const mussels = page.getByTestId("status-bivalve_shellfish");
      await expect(mussels.getByTestId("status-word")).toHaveText("Status unclear");
      await expect(mussels).toContainText("Sport harvest: sources differ, awaiting a person's review.");
      await expect(mussels).not.toContainText("Sport harvest closed");
      // the glance row, unboxed, four items, each linked to its chapter
      const glance = page.getByTestId("today-glance");
      await expect(glance.locator("a")).toHaveCount(4);
      await expect(glance).toContainText("4 flagged");
      await expect(glance).toContainText("not verified");
      await expect(glance).toContainText("5.5 of 30 ppm");
      await expect(glance).toContainText("chlorophyll, not toxin");
      await expect(page.getByTestId("glance-sea")).toHaveAttribute("href", "#sea");
      // a season that has simply ended is closed, said calmly: a barred disc and "Season closed", not the alarm triangle
      const salmon = page.getByTestId("status-salmon_commercial");
      await expect(salmon.getByTestId("status-word")).toHaveText("Season closed");
      await expect(salmon.locator('svg[data-mark="season"]')).toBeVisible();
      await expect(salmon).toContainText("season has concluded");
      // the order on the page follows the hierarchy
      const order = await page.locator("[data-area]").evaluateAll((els) => els.map((e) => e.getAttribute("data-area")));
      expect(order).toEqual(["status", "sea", "tests", "bloom", "actions"]);
      // the sea: measured figures, the 36-hour strip with the tide on the same axis; the forecasters' periods fold away
      await foldTrigger(page, "sea-more").click();
      await expect(page.getByTestId("sea-periods").locator("thead th")).toHaveCount(10);
      await expect(page.getByTestId("sea-now")).toContainText("4.9");
      await expect(page.getByTestId("tide-curve")).toBeVisible();
      // the tests: a plain sentence first, then the chart with a text alternative
      await expect(page.getByTestId("tests-lede")).toHaveText("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
      await expect(page.getByTestId("toxin-strip")).toHaveAttribute("role", "img");
      await expect(page.getByTestId("toxin-strip").locator("desc")).toContainText("against the 30 ppm action level");
      await expect(page.getByTestId("toxin-chart").first()).toHaveAttribute("role", "img");
      // the bloom outlook, top-down: (a) NOAA's toxin-conditions model with its run date, then (b) Neritic's 8-day chlorophyll outlook
      await expect(page.getByTestId("bloom-lede")).toContainText("probability of elevated particulate domoic acid in the water");
      await expect(page.getByTestId("charm-run")).toHaveText("run Oct 8");
      await expect(page.getByTestId("bloom-timeline")).toBeVisible();
      const outlook = page.getByTestId("outlook");
      // every day of this run has passed: it is a past run, never an outlook, and it steps back behind the 8-day outlook
      expect(await page.getByTestId("today-bloom").locator('[data-testid="bloom-charm"], [data-testid="outlook"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")))).toEqual(["outlook", "bloom-charm"]);
      await expect(page.getByTestId("bloom-charm")).toContainText("past run");
      await expect(page.getByTestId("bloom-lede")).toContainText("Those days have passed.");
      // (b) leads with a sentence a fisherman can read: higher than, about, or lower than usual for the season
      await expect(page.getByTestId("outlook-say")).toHaveText("Next 8 days (Oct 13–20): chlorophyll near Moss Landing is expected to run higher than usual for this time of year, about 1.6× the typical level.");
      await expect(page.getByTestId("outlook-say").locator("b")).toHaveText("higher than usual");
      await expect(outlook).toHaveAttribute("data-level", "higher");
      // the same thing as one scale, one tap down, with a text alternative that says everything the drawing says
      await foldTrigger(page, "bloom-more").click();
      const scale = page.getByTestId("outlook-scale");
      await expect(scale).toBeVisible();
      await expect(scale).toHaveAttribute("role", "img");
      await expect(scale.locator("desc")).toHaveText(/The outlook is 1\.6× the usual chlorophyll for this time of year: higher than usual\. Shaded band: its typical error, a factor of 2 either way, from 0\.\d× to 3\.\d× usual\./);
      // what it is, how sure, and a neutral use, in its detail; the heading says "chlorophyll, not toxin" at every width
      await expect(page.getByTestId("outlook")).toContainText("8-day chlorophyll outlook · not toxin");
      await expect(page.getByTestId("outlook-words")).toHaveText("A bloom indicator, not a toxin forecast: chlorophyll is algae of every kind. It is one 8-day average, not a day-by-day forecast, and its typical error is about a factor of 2 either way. A higher-than-usual outlook is a reason to watch the next CDPH test results, not a sign of toxin.");
      // the held-out skill, read from the data file, is in the same fold, not in the sentence
      const skill = page.getByTestId("outlook-skill");
      await expect(page.getByTestId("outlook-words")).not.toContainText("RMSE");
      await expect(skill).toContainText("Held-out RMSE ln(mg m-3): 0.754 for this outlook, against 0.85 climatology and 0.892 persistence (n = 185,000; 2019 to 2021-06). Lower is better.");
      await expect(page.getByTestId("outlook-numbers")).toContainText("about 3.2 mg/m³ against a usual 2.0 mg/m³");
      // whale rules
      await expect(page.getByTestId("whale-lede")).toContainText("Zone 4");
      await page.getByTestId("status-whale").click();
      await expect(page.getByTestId("whale-sheet").getByTestId("whale-next")).toContainText("Dec 15");
      await expect(page.getByTestId("whale-next")).toContainText("December 15-18");
      await page.keyboard.press("Escape");
      await expect(page.getByTestId("whale-sheet")).toBeHidden();
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
  expect(JSON.parse((await page.evaluate(() => localStorage.getItem("nx-port-v1")))!).fisheries).toEqual([]);
  // a citation to a fact inside a fold opens the fold, even one the reader shut by hand
  await page.goto(`${MOSS}#fact-${encodeURIComponent("status:rock_crab")}`);
  await ready(page);
  const rock = page.locator('[data-fact-hit="status:rock_crab"]');
  await expect(rock).toBeInViewport();
  await expect(page.getByTestId("status-rock_crab")).toBeVisible();
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

test("the briefing asks once whether it was useful, at the end; the answer is kept on this device", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  const asks = page.getByTestId("feedback");
  await expect(asks).toHaveCount(1);
  const end = page.getByTestId("today-actions").getByTestId("feedback");
  await expect(end).toHaveAttribute("data-fact", "status:dungeness_commercial");
  await end.getByTestId("feedback-yes").click();
  await expect(end).toContainText("Thanks — saved on this device.");
  await page.reload();
  await ready(page);
  await expect(page.getByTestId("today-actions").getByTestId("feedback-yes")).toHaveAttribute("aria-pressed", "true");
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
  await expect(page.getByTestId("wind-where")).toContainText("buoy 46092");
  await expect(page.getByTestId("wind-where")).toContainText("22 km away");
  await expect(page.getByTestId("today-sea")).toContainText("it has no wind sensor, so wind is from buoy 46092, 22 km away");
  await expect(page.getByTestId("sea-now")).toContainText("4.9");
  // mussels: the notices registry lists the quarantine, the curated status says CDPH announced its end.
  // The row keeps the more restrictive level and flags the disagreement; the sheet shows both, each with its date
  const row = page.getByTestId("status-bivalve_shellfish");
  await expect(row).toHaveAttribute("data-severity", "red");
  await expect(row.getByTestId("sources-differ")).toBeVisible();
  await row.click();
  const sheet = page.getByTestId("status-sheet");
  await expect(sheet.getByTestId("sheet-word")).toHaveText("Status unclear");
  await expect(sheet).toContainText("Not verified");
  await expect(sheet).toContainText("This is a rule for sport harvest, not a closure of a commercial fishery.");
  await expect(sheet.getByTestId("conflict")).toContainText("does not choose between them");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("Closed: Annual quarantine of sport-harvested mussels");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("in effect since May 1 · registry last read Oct 8 · not verified");
  await expect(sheet.getByTestId("conflict-notice")).toContainText("the more restrictive");
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
  await expect(page.getByTestId("status-spiny_lobster")).toHaveCount(1);
  await page.reload();
  await expect(page.getByTestId("status-spiny_lobster")).toHaveCount(1);
  await expect(page.getByTestId("status-market_squid")).toHaveCount(0);
  // a link's own list wins
  await page.goto(`${MOSS}&f=groundfish`);
  await expect(page.getByTestId("status-strip").locator("li")).toHaveCount(1);
});

test("since your last visit: nothing on the first visit, one line once something changed", async ({ page }) => {
  await page.goto(MOSS);
  await ready(page);
  await page.waitForFunction(() => !!localStorage.getItem("nx-today-seen-v1:port-592"));
  await expect(page.getByTestId("today-changed")).toHaveCount(0);
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
  await expect(page.locator("h1")).toHaveText("Where do you fish?");
  await expect(page.getByTestId("ocean-stage")).toBeVisible();
  const pilots = page.locator('[data-testid^="pick-port-"]');
  await expect(pilots).toHaveCount(4);
  await expect(pilots.first()).toContainText("Moss Landing");
  expect(await axe(page)).toEqual([]);
  await pilots.nth(2).click();
  await expect(page.locator("h1")).toHaveText("Santa Cruz");
  await page.goto(`${OK}/today`);
  await expect(page).toHaveURL(/\/today\?place=port-593$/);
  await expect(page.getByTestId("nav-bar").getByTestId("nav-today")).toHaveAttribute("href", "/today?place=port-593");
  // an unknown port is said plainly
  await page.goto(`${OK}/today?place=port-99999`);
  await expect(page.getByTestId("today-unknown-port")).toContainText("port-99999");
});

test("a none-on-record lede and glance carry their caveat where they say it, above the fold", async ({ page }) => {
  await page.goto(`${MOSS}&f=groundfish`);
  await ready(page);
  const lede = page.getByTestId("today-lede");
  await expect(lede).toContainText("No closure or restriction on record");
  await expect(lede).toContainText("Not a promise they are open");
  await expect(lede).toBeInViewport();
  await expect(page.getByTestId("glance-status")).toContainText("None flagged");
  await expect(page.getByTestId("glance-status")).toContainText("not a promise they are open");
  // no record behind the row: its sheet states the review state, it never reads as checked
  await expect(page.getByTestId("status-not-verified")).toContainText("of");
  await foldTrigger(page, "status-none").click();
  await page.getByTestId("status-groundfish").click();
  await expect(page.getByTestId("status-sheet")).toContainText("Not verified: Neritic holds no record for this fishery here, so no person has checked anything");
});

test("the hero: the reader's waters as a stage, no invented curves, the advisory's end as NWS said it", async ({ page }) => {
  // the recorded answers have no advisory here: one with no end time, then one with an end time
  await page.route("**/api/conditions*", async (route) => {
    const r = await route.fetch();
    const j = await r.json();
    j.alerts = { status: "ok", source: "NOAA National Weather Service", url: "https://example.org", fetchedAt: "2026-10-13T06:00:00.000Z", data: [{ event: "Small Craft Advisory", headline: "Small craft advisory in effect through this evening", onset: null, ends: null, url: "https://example.org/a" }] };
    await route.fulfill({ response: r, json: j });
  });
  await page.goto(MOSS);
  await ready(page);
  await expect(page.locator(".today-hero-iso")).toHaveCount(0);
  await expect(page.getByTestId("ocean-stage")).toBeVisible();
  await expect(page.getByTestId("stage-alt")).toContainText("not toxin");
  await expect(page.getByTestId("today-sealine")).toContainText("small craft advisory");
  await expect(page.getByTestId("sea-alert")).toHaveText(/Small Craft Advisory · no end time given/);
  await foldTrigger(page, "sea-more").click();
  await expect(page.getByTestId("sea-alert-words")).toContainText("NWS’s headline: Small craft advisory in effect through this evening");
  // NWS's wind in its own order: a later change is said as a change
  await expect(page.getByTestId("today-sealine")).toContainText("W 10–15 kt, then NW 5–10 kt … · seas 3–5 ft");
  await expect(page.getByTestId("today-sealine")).not.toContainText("N 5–15");
  await expect(page.getByTestId("today-sea")).toContainText("W 10–15 kt, then NW 5–10 kt, then N · seas 3–5 ft");
  await page.unroute("**/api/conditions*");
  await page.route("**/api/conditions*", async (route) => {
    const r = await route.fetch();
    const j = await r.json();
    j.alerts = { status: "ok", source: "NOAA National Weather Service", url: "https://example.org", fetchedAt: "2026-10-13T06:00:00.000Z", data: [{ event: "Gale Warning", headline: "Gale warning", onset: null, ends: "2026-10-13T15:00:00-07:00", url: "https://example.org/b" }] };
    await route.fulfill({ response: r, json: j });
  });
  await page.reload();
  await ready(page);
  await expect(page.getByTestId("sea-alert")).toContainText("Gale Warning · until Oct 13, 3:00 PM");
  expect(await axe(page)).toEqual([]);
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
  // an official notice still leads (this dataset has the notices registry), and the missing list is said right after it
  await expect(page.getByTestId("today-lede")).toContainText("annual quarantine of sport-harvested mussels (sport harvest only).");
  await expect(page.getByTestId("lede-caution")).toHaveText("The rest of Neritic's season list did not load. Check CDFW and CDPH before you go.");
  await expect(page.getByTestId("today-status")).toContainText("Neritic’s list of seasons and restrictions did not load. Unknown is not open");
  // one statement for the fisheries Neritic cannot speak to, by name: never a row that reads like a status, never "open"
  const unknown = page.getByTestId("status-unknown");
  await expect(unknown.getByTestId("status-word")).toHaveText("Status not available");
  await expect(unknown).toContainText("Dungeness crab (commercial) · Rock crab · Salmon (commercial) · Groundfish");
  await expect(unknown).toContainText("Unknown is not open.");
  await expect(page.getByTestId("status-dungeness_commercial")).toHaveCount(0);
  await expect(page.getByTestId("status-strip")).not.toContainText("None on record");
  await expect(page.getByTestId("status-strip")).not.toContainText(/\b(is|are) open\b/i);
  await expect(page.getByTestId("today-freshness")).toContainText("Status not available");
  await expect(page.getByTestId("today-freshness")).not.toContainText("not available (old)");
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
  // offline the stage is the poster, never a broken map: no canvas, and it says the lines are not the reader's waters
  await expect(page.getByTestId("ocean-stage")).toHaveAttribute("data-state", "poster-only");
  await expect(page.getByTestId("stage-poster")).toBeVisible();
  await expect(page.locator("canvas.maplibregl-canvas")).toHaveCount(0);
  await expect(page.getByTestId("stage-poster-note")).toContainText("not your waters");
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

test("the bloom glance names the outlook's own period, so a period already under way never reads as the next 8 days; the chapter opens with its sentence", async ({ page }) => {
  await page.goto(MOSS);
  const glance = page.getByTestId("glance-bloom");
  await expect(glance).toContainText(/[A-Z][a-z]{2} \d+/);
  await expect(glance).not.toContainText("next 8 days");
  await expect(glance).toContainText("chlorophyll, not toxin");
  await expect(page.getByTestId("today-bloom").locator(".today-say")).toContainText("That says nothing about toxin.");
});
