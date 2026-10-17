import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * My Coast (v4): an answer-first trip briefing. What matters now first; the evidence that describes
 * these waters apart from what is far away or old; official restrictions marked not verified; what is
 * not known; then what to check. The trip (activity, species, day) lives in the link.
 */
const NOW = "2026-10-13T06:00:00Z";

test("the briefing answers first, in order, and every statement says its kind", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  const order = await page.locator('[data-testid^="brief-"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
  expect(order.slice(0, 2)).toEqual(["brief-now", "brief-changes"]);
  expect(order).toEqual(expect.arrayContaining(["brief-measured", "brief-official", "brief-unknown", "brief-checks"]));
  await expect(page.getByTestId("matters")).toBeVisible();
  const kinds = await page.locator('[data-testid="matters"] [data-kind]').evaluateAll((els) => els.map((e) => e.getAttribute("data-kind")));
  expect(kinds.length).toBeGreaterThanOrEqual(3);
  for (const k of kinds) expect(["official", "measured", "model", "satellite", "conditions", "gap"]).toContain(k);
  await expect(page.getByTestId("brief-official")).toContainText("Not verified");
  await expect(page.getByTestId("unknowns")).toContainText("Toxin in seafood is not measured by Neritic");
  // nothing is called safe; chlorophyll is never toxin
  const text = await page.locator("main").innerText();
  expect(text).not.toMatch(/\b(safe to eat|safe to harvest|all clear)\b/i);
  // an official statement has no place of its own: "Show on map" opens the harbour's waters
  await page.locator('[data-testid="matters"] > li[data-kind="official"]').first().getByRole("button", { name: "Show on map" }).click();
  await expect(page).toHaveURL(/sel=harbor(%3A|:)550/);
  await expect(page.getByTestId("inspect-title")).toHaveText("Monterey");
});

test("the trip controls write the link, and the link reproduces the trip", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-201`);
  await page.getByTestId("trip-activity").selectOption("commercial");
  await expect(page).toHaveURL(/for=commercial/);
  await page.getByTestId("trip-species").selectOption("dungeness");
  await expect(page).toHaveURL(/species=dungeness/);
  await page.reload();
  await expect(page.getByTestId("trip-activity")).toHaveValue("commercial");
  await expect(page.getByTestId("trip-species")).toHaveValue("dungeness");
  await expect(page.getByTestId("brief-checks")).toContainText("Dungeness crab");
});

test("an unknown place says so and offers places; the landing lists harbours", async ({ page }) => {
  await page.goto(`${OK}/coast?place=Santa%20Cruz`);
  await expect(page.getByTestId("coast-unknown-place")).toContainText("not one Neritic knows");
  await expect(page.getByTestId("coast-landing").getByTestId("coast-start-port-550")).toBeVisible();
});

test("on a phone the trip is one line, the answer is near the top, and nothing scrolls sideways", async ({ browser }) => {
  for (const width of [390, 320]) {
    const ctx = await browser.newContext({ viewport: { width, height: 760 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
    await expect(page.getByTestId("trip-summary")).toContainText("Commercial fishing");
    await expect(page.getByTestId("trip-controls")).toBeHidden();
    await expect(page.getByTestId("brief-now")).toBeVisible();
    const top = (await page.getByTestId("brief-now").boundingBox())!.y;
    expect(top, `${width}px: "What matters now" starts within the first screen`).toBeLessThan(560);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBe(0);
    await page.getByTestId("trip-change").click();
    await expect(page.getByTestId("trip-controls")).toBeVisible();
    await ctx.close();
  }
});

test("no serious accessibility violations on the briefing (desktop and phone)", async ({ browser }) => {
  for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const ctx = await browser.newContext({ viewport: vp });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=port-550`);
    await expect(page.getByTestId("matters").locator("li").first()).toBeVisible();
    const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").analyze();
    const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
    await ctx.close();
  }
});
