import { expect, test, type BrowserContext } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * Every official notice (test-migration.md, "Missing in v12" item 3): one page under Today, reached from Today's
 * notices row, the Others row and Alerts. Each notice carries its agency, action, dates, source link and "not
 * verified"; a record an agency release appears to lift is "sources differ", never lifted (T38, T40).
 */
test.use({ storageState: { cookies: [], origins: [] } });

const MOSS = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });
async function seed(ctx: BrowserContext) {
  await ctx.addInitScript((port) => {
    if (sessionStorage.getItem("__seeded")) return;
    sessionStorage.setItem("__seeded", "1");
    localStorage.setItem("nx-port-v1", port);
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-tour-v1", "dismissed");
  }, MOSS);
}

test.beforeEach(async ({ context }) => seed(context));

test("Today's notices row opens every notice for the port and the coast, each with agency, action, dates, source and 'not verified'", async ({ page }) => {
  await page.goto(`${OK}/today?place=port-592`);
  // v14: the ticket's Notices tab, its count on the tab
  const row = page.getByTestId("nx-today").getByRole("tab", { name: /^Notices/ });
  await expect(row).toContainText(/Notices\s*\d+/);
  await row.click();
  await expect(page.getByTestId("nx-today").getByRole("tabpanel")).toContainText(/official notices? · not verified/i);
  await page.getByTestId("nx-notices-link").click();
  await expect(page).toHaveURL(/\/notices\?place=port-592$/);
  const list = page.getByTestId("nx-notices");
  await expect(list.getByRole("heading", { level: 1 })).toHaveText(/notices? appl(y|ies) at Moss Landing/);
  // the tab stays on Today
  await expect(page.getByRole("navigation", { name: "Destinations" }).first().getByRole("link", { name: "Today" })).toHaveAttribute("aria-current", "page");
  // the statewide mussel quarantine applies here, with its agency, action, dates and source
  const mussels = page.locator('[data-notice="cdph-2026-annual-mussel-quarantine"]');
  await expect(mussels).toContainText("CDPH · Quarantine · Sport harvest · statewide");
  await expect(mussels).toContainText("In effect since May 1, 2026.");
  await expect(mussels.getByRole("link", { name: /SN26-008/ })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx");
  await expect(mussels.getByRole("button", { name: /not verified/ })).toBeVisible();
  // the Del Norte razor clam records are listed, elsewhere on the coast
  await expect(page.getByTestId("nx-notices-elsewhere")).toContainText("Del Norte");
  // every notice says "not verified" on its stamp; nothing claims to be verified
  const n = await page.getByTestId("nx-notice").count();
  expect(n).toBeGreaterThan(5);
  expect(await list.getByRole("button", { name: /not verified/ }).count()).toBe(n);
  await expect(list).not.toContainText(/\bverified\b(?<!not verified)/);
  // no notice is marked lifted or expired (a record's own words may still mention lifting)
  for (const meta of await list.locator("[data-notice] > p:first-child").allInnerTexts()) expect(meta).not.toMatch(/lifted|expired|reopen/i);
  await expect(list).not.toContainText(/\b(has been|was|is now) (lifted|expired)\b/i);
});

test("an expected end that has passed is said, never treated as an end", async ({ page }) => {
  await page.goto(`${OK}/notices?place=port-592`);
  // the fixture clock is Oct 13; the mussel quarantine's expected end (Oct 31) is still ahead
  await expect(page.locator('[data-notice="cdph-2026-annual-mussel-quarantine"]')).toContainText("Expected to end Oct 31, 2026; not an end until CDPH announces it.");
});

test("a record an agency release appears to lift is 'sources differ', quoting both, never lifted", async ({ page }) => {
  await page.goto(`${FAILED}/notices?place=port-592`);
  const differ = page.getByTestId("nx-notices-differ");
  await expect(differ).toContainText("Sources differ · not verified");
  await expect(differ).toContainText("Do not eat sport-harvested bivalve shellfish from Monterey County");
  await expect(differ.getByRole("link", { name: "CDPH release SN26-021" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx");
  // the disputed record is not also listed as applying here
  await expect(page.getByTestId("nx-notices-here").locator('[data-notice="cdph-2026-sn26-018-monterey-bivalves"]')).toHaveCount(0);
});

test("Alerts and the Others row reach the same list; without a port the saved one opens", async ({ page }) => {
  await page.goto(`${OK}/alerts?place=port-592`);
  await expect(page.getByTestId("alerts-notices")).toHaveAttribute("href", "/notices?place=port-592");
  await page.goto(`${OK}/notices`);
  await expect(page).toHaveURL(/\/notices\?place=port-592$/);
  await expect(page.getByTestId("nx-notices").getByRole("heading", { level: 1 })).toContainText("Moss Landing");
});

test("Today's Notices tab never shows a notice as in effect where the sources differ: the season record contradicts it, or an agency release appears to lift it (T40)", async ({ page, context }) => {
  // the season record says the mussel quarantine ended Oct 31 while the registry lists it: "sources differ", the same
  // words the Fisheries tab ("Status unclear") and /notices use, never an active notice
  await page.goto(`${FAILED}/today?place=port-592`);
  await page.getByTestId("nx-today").getByRole("tab", { name: /^Notices/ }).click();
  const panel = page.getByTestId("nx-today").getByRole("tabpanel");
  const mussels = page.locator('[data-today-notice="cdph-2026-annual-mussel-quarantine"]');
  await expect(mussels).toHaveAttribute("data-differ", "1");
  await expect(mussels).toContainText("Sources differ: CDPH's season record says");
  await expect(mussels).toContainText("Awaiting a person's review");
  await expect(panel).toContainText(/where sources differ/);
  // the record an agency release appears to lift (Monterey County bivalves) is not listed as in effect here at all
  await expect(page.locator('[data-today-notice="cdph-2026-sn26-018-monterey-bivalves"]')).toHaveCount(0);
  // the same at the port whose own fishery says "Status unclear"
  await context.addInitScript(() => localStorage.setItem("nx-port-v1", JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-550", fisheries: ["bivalve_shellfish"], vessel: null })));
  await page.goto(`${FAILED}/today?place=port-550`);
  await page.getByTestId("nx-today").getByRole("tab", { name: /^Notices/ }).click();
  await expect(page.locator('[data-today-notice="cdph-2026-annual-mussel-quarantine"]')).toHaveAttribute("data-differ", "1");
  await expect(page.locator('[data-today-notice="cdph-2026-annual-mussel-quarantine"]')).toContainText("Awaiting a person's review");
});
