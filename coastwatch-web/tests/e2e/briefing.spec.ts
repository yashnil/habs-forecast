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
  // v10 pass 2: the decision (season, the readings, the trip's economics, the checks), the 60 hours, then
  // the one evidence section (statements, what they rest on, what is not known)
  const order = await page.locator('[data-testid^="brief-"]:not([data-testid$="-toggle"])').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));
  expect(order.indexOf("brief-glance")).toBeLessThan(order.indexOf("brief-econ"));
  expect(order.indexOf("brief-econ")).toBeLessThan(order.indexOf("brief-checks"));
  expect(order.indexOf("brief-checks")).toBeLessThan(order.indexOf("brief-lanes"));
  expect(order.indexOf("brief-lanes")).toBeLessThan(order.indexOf("brief-now"));
  expect(order).toEqual(expect.arrayContaining(["brief-measured", "brief-official", "brief-unknown", "brief-checks"]));
  for (const id of ["brief-evidence", "brief-official", "brief-unknown"]) expect(order.indexOf(id)).toBeGreaterThan(order.indexOf("brief-now"));
  await expect(page.getByTestId("matters")).toBeVisible();
  const kinds = await page.locator('[data-testid="matters"] [data-kind]').evaluateAll((els) => els.map((e) => e.getAttribute("data-kind")));
  expect(kinds.length).toBeGreaterThanOrEqual(3);
  for (const k of kinds) expect(["official", "measured", "model", "satellite", "conditions", "gap"]).toContain(k);
  await expect(page.getByTestId("brief-official")).toContainText("Not verified");
  await expect(page.getByTestId("unknowns")).toContainText("Toxin in seafood is not measured by Neritic");
  // nothing is called safe; chlorophyll is never toxin
  const text = await page.locator("main").innerText();
  expect(text).not.toMatch(/\b(safe to eat|safe to harvest|all clear)\b/i);
  // v6: at most three one-line statements up front, the rest one tap away; each opens in place
  expect(await page.locator('[data-testid="matters"] > li:visible').count()).toBeLessThanOrEqual(3);
  // an official statement has no place of its own: "Show on map" opens the harbour's waters
  const official = page.locator('[data-testid="matters"] > li[data-kind="official"]').first();
  await official.locator('[data-testid^="claim-open-"]').click();
  await official.getByRole("button", { name: "Show on map" }).click();
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

test("v7 first visit: one question and one place search (an ARIA combobox over every harbour and site), no species chips, no directory", async ({ page }) => {
  await page.goto(`${OK}/coast`);
  await expect(page.getByTestId("coast-title")).toHaveText("Where are you fishing?");
  await expect(page.locator('[data-testid^="coast-pick-"]')).toHaveCount(0);
  expect(await page.getByTestId("coast-landing").locator("a").count()).toBeLessThanOrEqual(6);
  const box = page.getByTestId("coast-place-search").getByRole("combobox");
  await expect(box).toHaveAttribute("aria-expanded", "false");
  await box.fill("santa");
  await expect(box).toHaveAttribute("aria-expanded", "true");
  const list = page.getByRole("listbox", { name: "Places" });
  await expect(list.getByRole("option", { name: /Santa Cruz/ }).first()).toBeVisible();
  // by county too: a harvest site or harbour in Humboldt
  await box.fill("humboldt");
  await expect(list.getByRole("option").first()).toBeVisible();
  // keyboard: arrow to the first result, Enter opens its briefing
  await box.fill("Monter");
  await box.press("ArrowDown");
  await box.press("ArrowUp");
  await box.press("Enter");
  await expect(page).toHaveURL(/place=port-550/);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
});

test("v7: a bare /coast with a saved home port opens the home briefing with its trip", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("cw-home-port", JSON.stringify({ place: "port-201", for: "commercial", species: "dungeness" })));
  await page.goto(`${OK}/coast`);
  await expect(page).toHaveURL(/place=port-201.*for=commercial.*species=dungeness/);
  await expect(page.getByTestId("coast-title")).toHaveText("Crescent City");
  await expect(page.getByTestId("coast-home")).toContainText("Your home port");
});

test("on a phone the trip is one line, the answer is near the top, and nothing scrolls sideways", async ({ browser }) => {
  for (const width of [390, 320]) {
    const ctx = await browser.newContext({ viewport: { width, height: 760 }, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
    await expect(page.getByTestId("trip-summary")).toContainText("Commercial fishing");
    await expect(page.getByTestId("trip-controls")).toBeHidden();
    // v5: the answer is the at-a-glance block (season, sea, notices, worth); "What matters now" follows it
    await expect(page.getByTestId("brief-glance")).toBeVisible();
    const top = (await page.getByTestId("brief-glance").boundingBox())!.y;
    expect(top, `${width}px: the at-a-glance answer starts within the first screen`).toBeLessThan(560);
    expect((await page.getByTestId("brief-now").boundingBox())!.y).toBeGreaterThan(top);
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

test("one count of notices: the top bar, the glance and the notices section agree, and say what names the species", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550&for=commercial&species=dungeness`);
  await expect(page.getByTestId("matters")).toBeVisible();
  const here = Number(await page.getByTestId("notices-count").innerText());
  expect(here).toBeGreaterThan(0);
  // v10 pass 2: the decision row counts them once (unverified), and says what names the species
  const cell = page.getByTestId("glance-notices-cell");
  await expect(page.getByTestId("glance-notices")).toHaveText(new RegExp(`^${here}\\s*tied here$`));
  await expect(cell).toContainText(/Not verified · (none|\d+) names? Dungeness crab/i);
  await expect(page.getByTestId("brief-official")).toContainText("Not verified");
  // never "none here" when notices are tied to these waters
  expect(await page.locator("main").innerText()).not.toMatch(/None (in Neritic's list is )?tied to these waters/);
  const listed = await page.getByTestId("brief-official").locator('li[data-testid^="notice-official:"]').count();
  expect(listed).toBe(here);
});
