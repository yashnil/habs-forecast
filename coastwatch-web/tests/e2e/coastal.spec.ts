import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * Coastal mode (the default for anyone not set up for research): My Coast as a briefing
 * (official status, next steps, conditions) with the science one tap away, and a basemap-first
 * Ocean Map. Ocean conditions come from recorded NOAA responses (tests/conditions-fixtures.json).
 */
const NOW = "2026-10-08T20:00:00Z";
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

/** A visitor with no saved presentation (so Coastal) who has seen the welcome. */
async function coastal(browser: Browser, opts: object = { viewport: { width: 1440, height: 900 } }) {
  const ctx = await browser.newContext({ ...opts, storageState: { cookies: [], origins: [{ origin: OK, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }, { origin: FAILED, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }] }] } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  return { ctx, page };
}

async function noUnsafeClaims(page: Page) {
  const text = await page.locator("main").innerText();
  const bad = text
    .split(/(?<=[.!?])\s+|\n+/)
    .filter((s) => /\b(safe|open|all clear|good fishing)\b/i.test(s) && !/\b(not|no|never|does not|isn't|nothing)\b/i.test(s) && !/Whale.?safe/i.test(s));
  expect(bad, bad.join(" | ")).toEqual([]);
}

/** My Coast in Coastal mode is a trip plan in chapters; open one as a reader would. */
async function openChapter(page: import("@playwright/test").Page, id: "target" | "official" | "conditions" | "before") {
  const ch = page.getByTestId(`chapter-${id}`);
  if ((await ch.getAttribute("data-open")) == null) await page.getByTestId(`chapter-head-${id}`).click();
  await expect(ch).toHaveAttribute("data-open", "true");
}

/** Next steps show five rows; the rest are one tap away. */
async function moreSteps(page: import("@playwright/test").Page) {
  await openChapter(page, "before");
  await expect(page.getByTestId("next-steps")).toBeVisible();
  const more = page.getByTestId("next-steps-more");
  if (await more.count()) await more.click();
}

test.describe("Coastal mode", () => {
  test("My Coast is a trip plan: what you are after, what officially applies, conditions, before you go; the science is one tap away", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-550&for=recreational&species=anchovy-sardine`);
    await expect(page.getByTestId("mode-switch-navy").getByTestId("mode-coastal")).toHaveAttribute("aria-pressed", "true");
    // a link that names the activity opens on what officially applies; the other chapters keep their answers
    await expect(page.getByTestId("chapter-official")).toHaveAttribute("data-open", "true");
    await expect(page.getByTestId("chapter-target")).toContainText("Recreational · Anchovy or sardine");
    // official status for the chosen species, each notice with its agency source
    await expect(page.getByTestId("brief-official-headline")).toHaveText("2 notices cover these waters for anchovy or sardine");
    // the answer comes first, one line per kind of knowing, each led by its mark
    await expect(page.getByTestId("answer-official")).toContainText("2 notices cover these waters for anchovy or sardine");
    await expect(page.getByTestId("answer-official").locator("[data-mark=official]")).toHaveCount(1);
    await expect(page.getByTestId("answer-model")).toHaveAttribute("data-kind", "gap");
    await expect(page.getByTestId("brief-notice-cdfw-2026-anchovy-take-restriction-monterey-bay").getByRole("link", { name: /CDFW source/ })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
    // next steps: every step has a source; regulation, species and conditions steps are there
    await openChapter(page, "before");
    const steps = page.getByTestId("next-steps").locator('[data-testid^="step-"]');
    // the first five show as rows; the rest are one tap away (conditions readings stay in their card)
    await expect(steps).toHaveCount(5);
    await page.getByTestId("next-steps-more").click();
    expect(await steps.count()).toBeGreaterThanOrEqual(5);
    // why each step is suggested opens in place
    await page.getByTestId("why-read-notices").click();
    await expect(page.getByTestId("step-read-notices")).toContainText("Read the 2 notices that cover these waters");
    for (const s of await steps.all()) await expect(s.locator('a[href^="https://"], a[href^="tel:"]')).toHaveCount(1);
    await expect(page.getByTestId("step-read-notices")).toContainText("Read the 2 notices that cover these waters");
    await expect(page.getByTestId("step-species-rules")).toContainText("anchovy or sardine");
    await expect(page.getByTestId("step-region-central")).toContainText("Central region");
    // conditions from NOAA, each with its source and time; the folded chapter already says the forecast
    await expect(page.getByTestId("chapter-conditions")).toContainText("W wind 10 to 15 kt");
    await openChapter(page, "conditions");
    await expect(page.getByTestId("cond-marine")).toContainText("W wind 10 to 15 kt");
    await expect(page.getByTestId("cond-marine")).toContainText("NWS PZZ535");
    await expect(page.getByTestId("cond-tides")).toContainText("Monterey");
    await expect(page.getByTestId("cond-forecast")).toContainText("No toxin forecast for these waters");
    // the science sits in a closed details element, in the page
    const details = page.getByTestId("coast-details");
    await expect(details).not.toHaveAttribute("open", "");
    await expect(page.getByTestId("coast-measured")).toBeHidden();
    await details.locator("summary").first().click();
    await expect(page.getByTestId("coast-measured")).toBeVisible();
    await noUnsafeClaims(page);
    await ctx.close();
  });

  test("choosing a species re-orders, never hides, and goes in the link", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-201`);
    // no activity in the link: the plan starts with what you are after
    await expect(page.getByTestId("chapter-target")).toHaveAttribute("data-open", "true");
    await page.getByTestId("coast-species-razor-clam").click();
    await expect(page).toHaveURL(/species=razor-clam/);
    await expect(page.getByTestId("answer-official")).toContainText("razor clams");
    await page.getByTestId("journey-next").click();
    await expect(page.getByTestId("brief-official-headline")).toHaveText(/notices? covers? these waters for razor clams/);
    // notices for other species are one tap away, not gone
    await expect(page.getByTestId("brief-others")).toContainText("more for other species here");
    await ctx.close();
  });

  test("an honest gap: no station within 60 km at Shelter Cove; the step says what is not known", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-215`);
    await openChapter(page, "conditions");
    await expect(page.getByTestId("cond-buoy")).toContainText("No NDBC station reports within 60 km");
    await expect(page.getByTestId("cond-buoy")).toHaveAttribute("data-state", "none");
    await moreSteps(page);
    await expect(page.getByTestId("step-toxin-gap")).toBeVisible();
    await ctx.close();
  });

  test("a NOAA outage reads in plain words, with the technical reason kept for anyone who looks", async ({ browser }) => {
    // a point on the water with no recorded NOAA responses: every live request fails like an outage
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=point-36.700,-122.300`);
    await openChapter(page, "conditions");
    const marine = page.getByTestId("cond-marine");
    await expect(marine).toContainText("did not answer just now");
    await expect(marine).toHaveAttribute("title", /HTTP 503/);
    await expect(marine).not.toContainText("HTTP 503");
    await moreSteps(page);
    await expect(page.getByTestId("step-marine-gap")).toContainText("Check the NWS marine forecast directly");
    await ctx.close();
  });

  test("switching to Research opens the science, is kept on the device, and a link's mode wins without being saved", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    await page.goto(`${OK}/coast?place=port-550`);
    await page.getByTestId("mode-switch-navy").getByTestId("mode-research").click();
    await expect(page.getByTestId("coast-details")).toHaveCount(0);
    await expect(page.getByTestId("coast-measured")).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("research");
    await page.getByTestId("mode-switch-navy").getByTestId("mode-coastal").click();
    await expect(page.getByTestId("coast-details")).toBeVisible();
    // a shared research link shows Research here, but the visitor's own choice stays Coastal
    await page.goto(`${OK}/coast?place=port-550&mode=research`);
    await expect(page.getByTestId("mode-switch-navy").getByTestId("mode-research")).toHaveAttribute("aria-pressed", "true");
    expect(await page.evaluate(() => localStorage.getItem("cw-mode-v1"))).toBe("coastal");
    await ctx.close();
  });

  test("the Ocean Map opens on the basemap in Coastal mode; a link naming a layer still shows it", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    // the depth tint cross-fades (raster-opacity), it is never toggled
    const visible = (id: string) => page.evaluate((i) => (window as unknown as { __cwMap: { getPaintProperty: (x: string, p: string) => number } }).__cwMap.getPaintProperty(i, "raster-opacity") > 0, id);
    await page.goto(`${OK}/?region=monterey_bay`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded: () => boolean } }).__cwMap?.loaded());
    await expect.poll(() => visible("relief-sea-tint")).toBe(true);
    await expect(page.getByTestId("forecast-panel")).toHaveCount(0);
    await page.goto(`${OK}/?region=monterey_bay&layer=forecast`);
    await expect(page.getByTestId("forecast-panel")).toBeVisible();
    await expect.poll(() => visible("relief-sea-tint")).toBe(false);
    await ctx.close();
  });

  test("a failed update is never labelled Current, even for a recent run", async ({ browser }) => {
    const { ctx, page } = await coastal(browser);
    // the failed fixture's last run was issued on the fixture day; its later update failed
    await page.goto(`${FAILED}/?region=monterey_bay&layer=forecast`);
    const badge = page.getByTestId("forecast-panel").getByTestId("freshness");
    await expect(badge).toHaveAttribute("data-state", "unconfirmed");
    await expect(badge).toContainText("Not confirmed");
    await expect(page.getByTestId("update-failed")).toContainText("a newer run may exist");
    await ctx.close();
  });

  test("reading the device's mode never re-mounts the Bloom chart or the Fisheries history (Research)", async ({ browser }) => {
    // regression: the Research layout used to replace the Coastal tree after the first render, so
    // the chart was detached and drawn again (a flash for researchers; CI caught a detached element)
    const { ctx, page } = await coastal(browser);
    await page.addInitScript(() => localStorage.setItem("cw-mode-v1", "research"));
    for (const [url, id] of [["/bloom?range=3m", "primary-chart"], ["/fisheries", "landings-chart"]] as const) {
      await page.goto(`${OK}${url}`);
      const el = await page.getByTestId(id).elementHandle();
      await page.waitForTimeout(1500);
      expect(await el!.evaluate((n) => n.isConnected), `${id} stayed in the page`).toBe(true);
      await expect(page.getByTestId(id)).toBeVisible();
    }
    await ctx.close();
  });

  for (const [name, opts] of [["desktop", { viewport: { width: 1440, height: 900 } }], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing on the Coastal briefing`, async ({ browser }) => {
      const { ctx, page } = await coastal(browser, opts);
      await page.goto(`${OK}/coast?place=site-clam-beach&for=shellfish&species=razor-clam`);
      await expect(page.getByTestId("brief-official")).toBeVisible();
      await openChapter(page, "conditions");
      await expect(page.getByTestId("cond-marine")).not.toContainText("Loading");
      await page.waitForTimeout(600);
      const r = await new AxeBuilder({ page }).include("main").analyze();
      expect(r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
      await ctx.close();
    });
  }
});
