import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/** My Coast: one location's official information, measurements, forecast availability and observations. */
const NOW = "2026-10-08T20:00:00Z";
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

async function open(page: Page, url: string, now = NOW) {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
  await expect(page.getByTestId("coast-title")).toBeVisible();
}

/** Every rendered sentence with "safe", "open" or "clear" must be negated (as in m3/m8). */
async function assertNoUnsafeClaims(page: Page) {
  const text = await page.locator("main").innerText();
  const bad = text
    .split(/(?<=[.!?])\s+|\n+/)
    .filter((s) => /\b(safe|open|all clear)\b/i.test(s) && !/\b(not|no|never|does not|isn't|nothing)\b/i.test(s) && !/Whale.?safe/i.test(s));
  expect(bad, bad.join(" | ")).toEqual([]);
}

test.describe("My Coast", () => {
  test("a shared link opens the waters: official geography first, then measurements, forecast availability and observations", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-201`);
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    const geo = page.getByTestId("coast-notices-geo");
    await expect(geo.getByTestId("coast-notice-cdfw-2026-razor-clam-del-norte")).toBeVisible();
    await expect(geo.getByTestId("coast-notice-cdph-2026-sn26-020-razor-clam-del-norte")).toContainText("Inside the agency's outline of Del Norte County");
    await expect(geo.getByTestId("coast-notice-cdph-rock-crab-viscera-north-coast-nci")).toContainText("Inside the notice's official latitudes");
    for (const n of await page.getByTestId("coast-notice-review").all()) await expect(n).toHaveText("Match to these waters not reviewed");
    await expect(page.getByTestId("coast-glance")).toContainText("not verified");
    await expect(page.getByTestId("coast-measured-none")).toContainText("No monitoring station within 40 km");
    await expect(page.getByTestId("glance-measured")).toContainText("No toxin measurement within 40 km");
    await expect(page.getByTestId("coast-model")).toContainText("C-HARM v3.1");
    await expect(page.getByTestId("coast-region-northern")).toContainText("Northern region");
    await expect(page.getByTestId("coast-unknowns")).toContainText("no person has reviewed these associations");
    await assertNoUnsafeClaims(page);
  });

  test("forecast availability never puts an offshore value at the location; the map draws the cell it came from", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-593`);
    const pda = page.getByTestId("coast-model-pda");
    await expect(pda).toBeVisible();
    const here = pda.getByTestId("coast-model-here");
    const nearest = pda.getByTestId("coast-model-nearest");
    if (await nearest.count()) {
      await expect(here).toContainText("No model value");
      await expect(nearest).toContainText("at that cell, not here");
      await expect(nearest).toContainText("km");
    }
    await expect
      .poll(() =>
        page.evaluate(() => {
          const m = (window as unknown as { __cwMap?: { getStyle: () => { sources: Record<string, { data?: { features?: { properties: { kind: string } }[] } }> } } }).__cwMap;
          return m?.getStyle().sources["coast-overlay"]?.data?.features?.map((f) => f.properties.kind) ?? [];
        }),
      )
      .toEqual(expect.arrayContaining(["target", "cell"]));
  });

  test("choosing waters with the keyboard updates the link; Back returns to the previous waters", async ({ page }) => {
    await open(page, `${OK}/coast`);
    await expect(page.getByTestId("coast-landing")).toBeVisible();
    await page.getByTestId("place-input").fill("Monterey");
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/coast\?place=port-550$/);
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
    await page.getByTestId("place-input").fill("Clam");
    await page.getByTestId("place-option-site-clam-beach").click();
    await expect(page).toHaveURL(/place=site-clam-beach/);
    await expect(page.getByTestId("coast-notice-cdfw-2024-razor-clam-humboldt")).toHaveAttribute("data-tier", "geographic");
    await page.goBack();
    await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
    await page.goBack();
    await expect(page.getByTestId("coast-landing")).toBeVisible();
  });

  test("clicking water on the map chooses those waters by coordinates", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-593`);
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    const box = (await page.getByTestId("coast-map-desktop").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.8);
    await expect(page).toHaveURL(/place=point-\d+\.\d{3}(,|%2C)-\d+\.\d{3}/);
    await expect(page.getByTestId("coast-subtitle")).toContainText("Point chosen on the map");
  });

  test("the fishing filter folds other notices without hiding them, and keeps the link", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-201`);
    await page.getByTestId("coast-for-commercial").click();
    await expect(page).toHaveURL(/for=commercial/);
    await expect(page.getByTestId("coast-notices-other")).toContainText("more for other kinds of fishing");
    const total = await page.locator('[data-testid^="coast-notice-"][data-tier]').count();
    expect(total).toBeGreaterThanOrEqual(4);
    await expect(page.getByTestId("coast-agencies")).toContainText("CDFW commercial fishing");
  });

  test("home port is separate from the waters, remembered on this device, and can be cleared", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-593`);
    await page.getByTestId("coast-home-set").click();
    await expect(page.getByTestId("coast-home")).toContainText("Home port: Santa Cruz");
    await page.goto(`${OK}/coast?place=port-201`);
    await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
    await expect(page.getByTestId("coast-home")).toContainText("Home port: Santa Cruz");
    await page.getByTestId("coast-home-open").click();
    await expect(page).toHaveURL(/place=port-593/);
    await page.getByTestId("coast-home-clear").click();
    await expect(page.getByTestId("coast-home-set")).toBeVisible();
  });

  test("an unknown place in a link says so instead of guessing", async ({ page }) => {
    await open(page, `${OK}/coast?place=Santa%20Cruz`);
    await expect(page.getByTestId("coast-unknown-place")).toContainText("not one CoastWatch knows");
    await expect(page.getByTestId("coast-landing")).toBeVisible();
  });

  test("a stale C-HARM run is flagged, not presented as current", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-593`, "2026-10-12T20:00:00Z");
    await expect(page.getByTestId("coast-model-run")).toBeVisible();
    await expect(page.getByTestId("coast-model-stale")).toBeVisible();
    await expect(page.getByTestId("coast-unknowns")).toContainText("The latest C-HARM run is not current");
  });

  test("reachable from the masthead, the phone tab bar and the Ocean Map's port panel", async ({ page, browser }) => {
    await open(page, `${OK}/coast`);
    await expect(page.getByTestId("nav-coast")).toHaveAttribute("aria-current", "page");
    const ctx = await browser.newContext(PHONE);
    const ph = await ctx.newPage();
    await open(ph, `${OK}/coast?place=port-593`);
    await expect(ph.getByTestId("tab-coast")).toHaveAttribute("aria-current", "page");
    await ctx.close();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?port=593`);
    const link = page.getByTestId("port-open-coast");
    if (await link.count()) await expect(link).toHaveAttribute("href", "/coast?place=port-593");
  });

  test("phone: the summary is on the first screen, no sideways scroll, and every tab fits at 320 px", async ({ browser }) => {
    for (const width of [390, 320]) {
      const ctx = await browser.newContext({ viewport: { width, height: width === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });
      const page = await ctx.newPage();
      await open(page, `${OK}/coast?place=port-593`);
      const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, vw: document.documentElement.clientWidth }));
      expect(m.sw).toBeLessThanOrEqual(m.vw);
      if (width === 390) {
        const g = (await page.getByTestId("coast-glance").boundingBox())!;
        const tab = (await page.getByTestId("tabbar").boundingBox())!;
        expect(g.y + g.height).toBeLessThanOrEqual(tab.y + 1);
      }
      for (const t of await page.locator('[data-testid^="tab-"]').all()) {
        const b = (await t.boundingBox())!;
        expect(b.x).toBeGreaterThanOrEqual(0);
        expect(b.x + b.width).toBeLessThanOrEqual(width + 0.5);
      }
      await ctx.close();
    }
  });

  for (const [name, opts] of [["desktop", { viewport: { width: 1440, height: 900 } }], ["phone", PHONE]] as const) {
    test(`${name}: axe finds nothing on My Coast`, async ({ browser }) => {
      const ctx = await browser.newContext(opts);
      const page = await ctx.newPage();
      await open(page, `${OK}/coast?place=port-201`);
      await expect(page.getByTestId("coast-model-pda").or(page.getByTestId("coast-model-error"))).toBeVisible();
      const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
      await ctx.close();
    });
  }
});
