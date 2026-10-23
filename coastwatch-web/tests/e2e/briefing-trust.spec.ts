import { expect, test } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * My Coast's scientific trust (v4), restored from the v2 coast and coastal suites: which notices
 * apply is decided by official geography and the trip; unresolved and unknown things are said; an
 * offshore value is never put at the harbour; old runs and missing samples are honest gaps.
 */
const NOW = "2026-10-13T06:00:00Z";
const settled = (page: import("@playwright/test").Page) => expect(page.getByTestId("matters").locator(":scope > li").first()).toBeVisible();

test("Crescent City: its notices, not verified; what no person has reviewed; no recent toxin sample", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-201`);
  await settled(page);
  // the official restrictions are folded below "Before you go", with their count; opened, each is there
  await expect(page.getByTestId("brief-official-toggle")).toContainText(/Official restrictions\s*\d/);
  await page.getByTestId("brief-official-toggle").click();
  await expect(page.getByTestId("notice-official:cdfw-2026-razor-clam-del-norte")).toContainText("Not verified");
  await expect(page.getByTestId("notice-official:cdph-2026-sn26-020-razor-clam-del-norte")).toBeVisible();
  await expect(page.getByTestId("unknowns")).toContainText("no person has reviewed these associations");
  await expect(page.getByTestId("unknowns")).toContainText(/no toxin measurement from the last 30 days within 40 km/i);
});

test("Santa Cruz: an offshore value is context, never a forecast for the harbour", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-593`);
  await settled(page);
  await expect(page.getByTestId("unknowns")).toContainText("C-HARM gives no toxin probability for these waters");
  await expect(page.getByTestId("unknowns")).toContainText("is not a forecast here");
  await expect(page.getByTestId("unknowns")).toContainText(/its nearest value is [\d.]+ km away/);
  // no model value is listed among the evidence that describes these waters
  await expect(page.getByTestId("brief-measured").locator('[data-node^="charm:"]')).toHaveCount(0);
});

test("official geography decides: a notice outside the waters is not listed; an unresolved one is said and checked", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-593`);
  await settled(page);
  await expect(page.getByTestId("notice-official:cdph-2026-sn26-018-monterey-bivalves")).toHaveCount(0);
  await page.goto(`${OK}/coast?place=port-605`);
  await settled(page);
  await expect(page.getByTestId("unknowns")).toContainText("is not settled by");
  await expect(page.getByTestId("brief-checks")).toContainText(/Confirm whether “.*”/);
});

test("a stale C-HARM run is said in what is not known, never presented as current", async ({ page }) => {
  await page.clock.setFixedTime(new Date("2026-10-12T20:00:00Z"));
  await page.goto(`${FAILED}/coast?place=port-593`);
  await settled(page);
  await expect(page.getByTestId("unknowns")).toContainText("The latest C-HARM run is not current");
});

test("the species leads the official headline, links to the agency, and says whose copy it is", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=recreational&species=anchovy-sardine`);
  await settled(page);
  const claim = page.getByTestId("claim-notices");
  await expect(claim).toContainText("anchovy or sardine");
  // v6: a statement is one line; its source opens on tap
  await claim.getByTestId("claim-open-notices").click();
  await expect(claim.getByRole("link").first()).toHaveAttribute("href", /^https:\/\/(wildlife\.ca\.gov|www\.cdph\.ca\.gov)\//);
  await expect(page.getByTestId("matters-provenance")).toContainText("No person has reviewed that copy yet, so the agency pages are the authority");
  // every id on the page is unique
  const ids = await page.locator("[id]").evaluateAll((els) => els.map((e) => e.id));
  expect(new Set(ids).size).toBe(ids.length);
});

test("changing the species changes the official headline and the checks at once, and goes in the link", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-201`);
  await settled(page);
  await page.getByTestId("trip-species").selectOption("razor-clam");
  await expect(page).toHaveURL(/species=razor-clam/);
  await expect(page.getByTestId("claim-notices")).toContainText("razor clam");
  await expect(page.getByTestId("brief-checks")).toContainText("CDPH");
  const opts = await page.getByTestId("trip-day").locator("option").evaluateAll((o) => o.map((x) => (x as HTMLOptionElement).value));
  await page.getByTestId("trip-day").selectOption(opts.at(-1)!);
  await expect(page).toHaveURL(new RegExp(`date=${opts.at(-1)}`));
});

test("Dungeness, commercial, Crescent City: the season status leads, quoted with its CDFW link; the closure line dials", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-201&for=commercial&species=dungeness`);
  await settled(page);
  await expect(page.getByTestId("matters").locator(":scope > li").first()).toHaveAttribute("data-testid", "claim-season");
  await page.getByTestId("claim-open-season").click();
  await expect(page.getByTestId("claim-season").getByRole("link").first()).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
  // the agency's own words, quoted in full, beside the headline (never only "season status")
  await expect(page.getByTestId("claim-season").getByTestId("claim-quote")).toContainText("Season is");
  await expect(page.getByTestId("claim-season")).toHaveAttribute("data-tone", /alert|note/);
  await expect(page.getByTestId("checks").locator(":scope > li").first()).toContainText("Confirm the season status on CDFW's page");
  const tel = page.getByTestId("brief-checks").getByTestId("check-tel").first();
  await expect(tel).toHaveAttribute("href", /^tel:\+?1?8316492883$/);
  // the number is shown once, as the agency writes it
  await expect(tel).toHaveText("(831) 649-2883");
  expect(await page.getByTestId("checks").locator(":scope > li").count()).toBeLessThanOrEqual(5);
});

test("Shelter Cove: no station within 60 km is an honest gap, said once, with what to check", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-215`);
  await settled(page);
  const hab = page.getByTestId("claim-hab");
  await expect(hab).toHaveAttribute("data-kind", "gap");
  await expect(hab).toContainText("No recent HAB sample within 60 km");
  await expect(hab).toContainText("No sample is not a clean result");
});

test("a point in Monterey Bay: the notices between the agency's latitudes, not verified", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=point-36.800,-121.950`);
  await settled(page);
  await expect(page.getByTestId("notice-official:cdph-2026-sn26-019-anchovy-central-coast")).toContainText("Not verified");
});

test("a shellfish harvester's CDPH site link opens that site with its notices, on a phone", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=site-clam-beach&for=shellfish`);
  await expect(page.getByTestId("coast-title")).toHaveText("Clam Beach, Humboldt");
  await expect(page.getByTestId("trip-summary")).toContainText("Shellfish");
  await expect(page.getByTestId("notice-official:cdfw-2024-razor-clam-humboldt")).toContainText("Not verified");
  await ctx.close();
});

test("a link's trip beats the purpose saved on this device; without one, the saved purpose applies", async ({ browser }) => {
  const ctx = await browser.newContext({ storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-purpose-v1", value: JSON.stringify({ purpose: "commercial" }) }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-201&for=recreational`);
  await expect(page.getByTestId("trip-activity")).toHaveValue("recreational");
  await page.goto(`${OK}/coast?place=port-201`);
  await expect(page.getByTestId("trip-activity")).toHaveValue("commercial");
  await ctx.close();
});
