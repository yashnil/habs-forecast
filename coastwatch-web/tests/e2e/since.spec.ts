import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import type { Snapshot } from "../../src/lib/briefingSnapshot";
import { NODATA, OK } from "./ports";

/**
 * "Since you last checked" and kept briefings (fishermen-first workstream), on My Coast in Coastal
 * mode (mounted under the answer line). Fixture data; nothing is fetched from a third party.
 */
const CRAB = "/coast?place=port-201&for=commercial&species=dungeness";
const coastal = (origin: string) => ({ origin, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] });
test.use({ storageState: { cookies: [], origins: [coastal(OK), coastal(NODATA)] } });
const SEASON = "cdfw-dungeness-2026-27-season-status";

async function seedPreviousVisit(page: Page, lastSeenAt: string, edit: (snap: Snapshot) => void) {
  const snap: Snapshot = await (await page.request.get(`${OK}/api/briefing?place=port-201&for=commercial&species=dungeness`)).json();
  edit(snap);
  const entry = { key: "port-201|commercial|dungeness", place: "port-201", placeName: "Crescent City", audience: "commercial", species: "dungeness", speciesLabel: "Dungeness crab", lastSeenAt, snapshot: snap };
  await page.goto(`${OK}/about`);
  await page.evaluate((v) => {
    localStorage.setItem("cw-waters-v1", JSON.stringify([v]));
    sessionStorage.clear();
  }, entry);
}

test.describe("since you last checked", () => {
  test("a first visit makes no claims; it says the waters are remembered on this device", async ({ page }) => {
    await page.goto(`${OK}${CRAB}`);
    const s = page.getByTestId("since-last-check");
    await expect(s).toHaveAttribute("data-state", "first");
    await expect(page.getByTestId("since-first")).toContainText("Remembered on this device");
    await expect(page.getByTestId("since-list")).toHaveCount(0);
    await expect(page.getByTestId("since-checked")).toContainText("CoastWatch last checked the agencies' pages");
  });

  test("a return visit lists the agency's changed statement first, and an old notice as newly listed by CoastWatch, never as new", async ({ page }) => {
    await seedPreviousVisit(page, "2026-10-07T16:00:00Z", (snap) => {
      snap.official.statements.find((s) => s.id === SEASON)!.statement = "Commercial Fishery: Season status to be announced.";
      // the Del Norte razor-clam notices were not listed at the last visit; the agency dated one before it
      // not listed at the last visit: the Del Norte razor-clam notices (agency-dated Oct 9) and the
      // statewide mussel quarantine (agency-dated May 1, long before the visit)
      snap.official.notices = snap.official.notices.filter((n) => !/razor-clam-del-norte|annual-mussel-quarantine/.test(n.id));
      snap.official.notices.push({ ...snap.official.notices[0], id: "old-removed-notice", title: "An old notice", status: "active", forYou: true });
    });
    await page.goto(`${OK}${CRAB}`);
    await expect(page.getByTestId("since-last-check")).toHaveAttribute("data-state", "changes");
    const first = page.getByTestId("since-list").locator("li").first();
    await expect(first).toHaveAttribute("data-testid", "since-item-agency-statement");
    await expect(first).toContainText("now says: “Commercial Fishery: Season is closed");
    await expect(first).toContainText("At your last check it said: “Commercial Fishery: Season status to be announced.”");
    await expect(page.getByTestId("since-item-removed")).toContainText("does not know why");
    // other species' items wait behind "more"
    await page.getByTestId("since-more").click();
    const razor = page.getByTestId("since-list").locator("li", { hasText: "razor clams from Del Norte" });
    await expect(razor.first()).toContainText("Other species here");
    await expect(razor.first()).toContainText("after your last check");
    const mussels = page.getByTestId("since-list").locator("li", { hasText: "quarantine" });
    await expect(mussels).toHaveAttribute("data-testid", "since-item-listed-late");
    await expect(mussels).toContainText("Newly listed by CoastWatch");
    await expect(mussels).toContainText("Not new from the agency");
    const text = (await page.getByTestId("since-list").innerText()).replace(/\s+/g, " ");
    expect(text).not.toMatch(/\b(is|are) (open|safe)\b|safe to eat|good fishing/i);
    // a reload in the same session compares with the same earlier visit
    await page.reload();
    await expect(page.getByTestId("since-item-agency-statement")).toBeVisible();
  });

  test("with storage blocked the page renders without errors and claims nothing", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.addInitScript(() => {
      for (const k of ["localStorage", "sessionStorage"])
        Object.defineProperty(window, k, {
          get() {
            throw new DOMException("blocked", "SecurityError");
          },
        });
    });
    await page.goto(`${OK}${CRAB}`);
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    await page.waitForTimeout(1500);
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
    await expect(page.getByTestId("since-none")).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test("remembered activity and species: one tap continues with them", async ({ page }) => {
    await page.goto(`${OK}${CRAB}`);
    await expect(page.getByTestId("since-last-check")).toBeVisible();
    await page.goto(`${OK}/coast?place=port-201`);
    const resume = page.getByTestId("since-resume");
    await expect(resume).toHaveText("Continue with Commercial fishing · Dungeness crab");
    await resume.click();
    await expect(page).toHaveURL(/\/coast\?place=port-201&for=commercial&species=dungeness/);
  });

  test("forget on this device removes what was remembered", async ({ page }) => {
    await page.goto(`${OK}${CRAB}`);
    await page.getByTestId("since-forget").click();
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
    expect(await page.evaluate(() => localStorage.getItem("cw-waters-v1"))).toBeNull();
  });

  test("when CoastWatch data are unavailable nothing is listed or claimed", async ({ page }) => {
    await page.goto(`${NODATA}${CRAB}`);
    await page.waitForTimeout(1500);
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
    expect((await page.request.get(`${NODATA}/api/briefing?place=port-201`)).status()).toBe(503);
  });

  test("Research mode does not show the block", async ({ page }) => {
    await page.goto(`${OK}${CRAB}&mode=research`);
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    await page.waitForTimeout(1500);
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
  });
});

test.describe("kept briefing", () => {
  test("keeps a dated copy that opens offline as a file, with its time, sources and warning", async ({ page, context }) => {
    await page.goto(`${OK}${CRAB}`);
    await page.getByTestId("kept-make").click();
    await expect(page.getByTestId("kept-done")).toContainText("Kept ");
    const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("kept-download").click()]);
    expect(dl.suggestedFilename()).toMatch(/^coastwatch-crescent-city-\d{4}-\d{2}-\d{2}\.html$/);
    const file = test.info().outputPath(dl.suggestedFilename());
    await dl.saveAs(file);
    const html = readFileSync(file, "utf8");
    expect(html).toContain("not a legal authorization to fish or harvest, and not proof that seafood is safe");
    expect(html).toContain("https://wildlife.ca.gov/Conservation/Marine/Whale-Safe-Fisheries");
    expect(html).not.toMatch(/<script|src=/i);

    await context.setOffline(true);
    const offline = await context.newPage();
    await offline.goto(`file://${file}`);
    await expect(offline.locator("h1")).toHaveText("Crescent City · Commercial fishing · Dungeness crab");
    await expect(offline.locator(".warn")).toContainText("This is a saved copy, not current information");
    await expect(offline.locator("body")).toContainText(/Saved [A-Z][a-z]{2} \d{1,2}, \d{4}/);
    await expect(offline.locator("body")).not.toContainText(/\bcurrent status\b|\btoday\b|\bnow open\b/i);
    await context.setOffline(false);
  });

  test("the print view keeps the warning and hides its buttons when printed", async ({ page }) => {
    await page.goto(`${OK}${CRAB}`);
    await page.getByTestId("kept-make").click();
    await page.getByTestId("kept-open").click();
    await expect(page).toHaveURL(/\/coast\/kept\?id=/);
    await expect(page.getByTestId("kept-title")).toHaveText("Crescent City · Commercial fishing · Dungeness crab");
    await expect(page.getByTestId("kept-warning")).toContainText("Agency actions after it was saved may replace anything in it");
    await page.emulateMedia({ media: "print" });
    await expect(page.getByTestId("kept-print")).toBeHidden();
    await expect(page.getByTestId("kept-warning")).toBeVisible();
  });
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test("touch targets are at least 44 px and the block has no serious accessibility violations", async ({ page }) => {
    await page.goto(`${OK}${CRAB}`);
    const s = page.getByTestId("since-last-check");
    await expect(s).toBeVisible();
    for (const el of await s.locator("a, button").all()) {
      const b = await el.boundingBox();
      expect(b!.height, await el.innerText()).toBeGreaterThanOrEqual(44);
    }
    const r = await new AxeBuilder({ page }).include("[data-testid=since-last-check]").analyze();
    expect(r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  });
});
