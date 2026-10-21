import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * The Explore shell (v4 final pass): panels that fit a phone, a map that stays open while a place is
 * read, controls that never cover the timebar, search that finds what people call places, and a
 * home port that Explore opens on.
 */
const NOW = "2026-10-08T20:00:00Z";
type M = { loaded: () => boolean; getCenter: () => { lat: number; lng: number } };
const mapReady = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: M }).__cwMap?.loaded());
const phone = (width = 390) => ({ viewport: { width, height: width === 320 ? 640 : 844 }, isMobile: true, hasTouch: true });
type Box = { x: number; y: number; width: number; height: number };
const overlap = (a: Box, b: Box) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

test("phone: the status centre opens inside the screen, under the top bar", async ({ browser }) => {
  for (const w of [320, 390]) {
    const ctx = await browser.newContext(phone(w));
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}/?layer=forecast`);
    await page.getByTestId("status-center-compact").click();
    const b = (await page.getByTestId("status-panel").boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(0);
    expect(b.x + b.width).toBeLessThanOrEqual(w);
    expect(b.y + b.height).toBeLessThanOrEqual(w === 320 ? 640 : 844);
    await ctx.close();
  }
});

test("desktop: the dock's layer button opens the layer list upward and Escape closes it (not the place); the credits stay in the map's margin", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&lead=1`);
  // the default state: no permanent deck, the dock's button names the layer and its kind
  const capsule = page.getByTestId("layer-capsule");
  await expect(capsule).toBeVisible();
  await expect(capsule).toContainText("NOAA bloom forecast");
  await expect(capsule).toContainText("Model");
  await expect(page.getByTestId("layer-deck")).toBeHidden();
  await capsule.click();
  await expect(page.getByTestId("layer-deck")).toBeVisible();
  // the popover rises from the dock: its bottom is above the dock's top
  const deck = (await page.getByTestId("layer-deck").boundingBox())!;
  const dock = (await page.getByTestId("timebar-deck").boundingBox())!;
  expect(deck.y + deck.height).toBeLessThanOrEqual(dock.y);
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("layer-deck")).toBeHidden();
  // with a place open, Escape closes the list first and leaves the place
  await page.goto(`${OK}/?layer=forecast&lead=1&sel=harbor:550`);
  await expect(page.getByTestId("inspect-panel")).toBeVisible();
  await expect(page.getByTestId("attribution-line")).toContainText("© OpenStreetMap contributors");
  await capsule.click();
  await expect(page.getByTestId("layer-deck")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("layer-deck")).toBeHidden();
  await expect(page.getByTestId("inspect-panel")).toBeVisible();
  // the dock, the tools and the panel never overlap, and the disclosure line is never under any of them
  const panel = (await page.getByTestId("inspect-panel").locator("xpath=ancestor::aside").boundingBox())!;
  const dock2 = (await page.getByTestId("timebar-deck").boundingBox())!;
  const tools = (await page.getByTestId("map-controls").boundingBox())!;
  const line = (await page.getByTestId("registry-disclosure").boundingBox())!;
  for (const [a, b, what] of [[dock2, tools, "dock/tools"], [dock2, panel, "dock/panel"], [tools, panel, "tools/panel"], [line, dock2, "line/dock"], [line, panel, "line/panel"]] as const) expect(overlap(a, b), what).toBe(false);
});

test("tablet and short screens: the map's tools never cover the dock, with or without a place open", async ({ browser }) => {
  for (const viewport of [{ width: 834, height: 1112 }, { width: 1024, height: 768 }, { width: 844, height: 390 }]) {
    const ctx = await browser.newContext({ viewport, hasTouch: true });
    const page = await ctx.newPage();
    await page.clock.setFixedTime(new Date(NOW));
    for (const url of ["/?layer=forecast&lead=1", "/?layer=forecast&lead=1&sel=harbor:550"]) {
      await page.goto(`${OK}${url}`);
      const deck = page.getByTestId("timebar-deck");
      await expect(deck).toBeVisible();
      const d = (await deck.boundingBox())!;
      for (const b of await page.getByTestId("map-controls").locator("button").all()) {
        if (!(await b.isVisible())) continue;
        expect(overlap(d, (await b.boundingBox())!), `${viewport.width}x${viewport.height} ${url}`).toBe(false);
      }
    }
    await ctx.close();
  }
});

test("search: another name finds the harbour, and no match says what to do next", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  const input = page.getByTestId("place-search-input");
  await input.fill("Pillar Point");
  const list = page.getByTestId("place-search-list");
  await expect(list.getByRole("option").first()).toContainText("Princeton / Half Moon Bay");
  await expect(list.getByRole("option").first()).toContainText("also called Pillar Point");
  await input.fill("Atlantis");
  await expect(page.getByTestId("place-search-empty")).toContainText("No harbour, station or region matches");
  await page.getByTestId("place-search-clear").click();
  await expect(list.getByRole("option").first()).toContainText("Monterey");
  // a pick moves focus to the place's panel; closing it returns focus to the search, not the page top
  await list.getByRole("option").first().click();
  await expect(page.getByTestId("inspect-panel")).toBeFocused();
  await page.getByTestId("inspect-close").click();
  await expect(page.locator("#place-search")).toBeFocused();
});

test("a home port: Explore opens on its region, search offers it first, and My Coast goes to its briefing", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(() => window.localStorage.setItem("cw-home-port", JSON.stringify({ place: "port-201", for: "commercial" })));
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/`);
  await mapReady(page);
  const c = await page.evaluate(() => (window as unknown as { __cwMap: M }).__cwMap.getCenter());
  expect(c.lat).toBeGreaterThan(39.9); // the north coast, not the whole state
  await page.getByTestId("place-search-input").click();
  await expect(page.getByTestId("place-search-list").getByRole("option").first()).toContainText("Your home port");
  // the whole trip, not only the place (v7): the reader lands on their own activity and species
  await expect(page.getByTestId("nav-bar").getByTestId("nav-coast")).toHaveAttribute("href", "/coast?place=port-201&for=commercial");
  // a link with its own place is not overridden
  await page.goto(`${OK}/?sel=harbor:550`);
  await mapReady(page);
  await expect(page.getByTestId("inspect-title")).toHaveText("Monterey");
  await ctx.close();
});

test("the area tool's letter works only with the map focused", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast`);
  await mapReady(page);
  await page.getByTestId("recenter").focus();
  await page.keyboard.press("a");
  await expect(page.getByTestId("area-overlay")).toHaveCount(0);
  await page.locator(".maplibregl-canvas").focus();
  await page.keyboard.press("a");
  await expect(page.getByTestId("area-overlay")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("area-overlay")).toHaveCount(0);
});
