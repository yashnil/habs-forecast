import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/** M8: the About page and its routes, the first-visit introduction, self-hosted map glyphs, and the
 *  phone layouts (Bloom's first value above the fold, Fisheries' section links, compact notices). */
const NOW = "2026-10-08T20:00:00Z";
async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
}
const PHONE = { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true };

/** Every rendered sentence that says "safe", "open" or "loss" must be negated (as in m3.spec.ts). */
async function assertNoUnsafeClaims(page: Page) {
  const text = await page.locator("body").innerText();
  const bad = text
    .split(/(?<=[.!?])\s+|\n+/)
    .filter((s) => /\b(safe|open|all clear|losse?s?)\b/i.test(s) && !/\b(not|no|never|does not|isn't)\b/i.test(s) && !/Whale.?safe/i.test(s));
  expect(bad, bad.join(" | ")).toEqual([]);
}

test.describe("About CoastWatch", () => {
  test("is in the masthead and says what CoastWatch is, its four kinds of information, its limits and the research", async ({ page }) => {
    await open(page, `${OK}/bloom`);
    await page.getByTestId("nav-about").click();
    await expect(page).toHaveURL(/\/about$/);
    await expect(page.getByTestId("nav-about")).toHaveAttribute("aria-current", "page");
    const kinds = page.getByTestId("about-kinds");
    for (const k of ["Official", "Model", "Measured", "Historical"]) await expect(kinds).toContainText(k, { ignoreCase: true });
    await expect(page.getByTestId("about-cannot")).toContainText("does not mean an area is open or seafood is safe");
    const pub = page.getByTestId("publication");
    await expect(pub).toContainText("Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System");
    await expect(page.getByTestId("publication-doi")).toHaveAttribute("href", "https://doi.org/10.33422/ccgconf.v2i2.1619");
    await expect(page.getByTestId("research")).toContainText("matches, but does not beat");
    await expect(page.getByTestId("research-relation")).toContainText("CoastWatch does not run those research models");
    await expect(page.getByTestId("research-relation")).toContainText("not been peer reviewed or reviewed by an independent HAB scientist");
    await expect(page.getByTestId("about-review")).toContainText("have not been checked by a person");
    await assertNoUnsafeClaims(page);
  });

  test("reading pages end with a footer to About, the sources and the paper", async ({ page }) => {
    for (const path of ["/bloom", "/fisheries", "/sources", "/about"]) {
      await open(page, `${OK}${path}`);
      const f = page.getByTestId("site-footer");
      await expect(f.getByRole("link", { name: "About" })).toHaveAttribute("href", "/about");
      await expect(f.getByRole("link", { name: /Research paper/ })).toHaveAttribute("href", "https://doi.org/10.33422/ccgconf.v2i2.1619");
    }
  });

  test("on a phone About is a tab", async ({ browser }) => {
    const ctx = await browser.newContext(PHONE);
    const page = await ctx.newPage();
    await open(page, `${OK}/fisheries`);
    await page.getByTestId("tab-about").click();
    await expect(page).toHaveURL(/\/about$/);
    await expect(page.getByTestId("tab-about")).toHaveAttribute("aria-current", "page");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
    await ctx.close();
  });
});

test.describe("first visit", () => {
  test("the map introduces itself once, links to About, and stays dismissed", async ({ browser }) => {
    // a first-time visitor: no stored state (the suite's default marks the introduction as seen)
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: { cookies: [], origins: [] } });
    const page = await ctx.newPage();
    await open(page, OK);
    const card = page.getByTestId("intro-card");
    await expect(card).toBeVisible();
    await expect(card).toContainText("not an official source");
    for (const k of ["Model", "Measured", "Official"]) await expect(card).toContainText(k, { ignoreCase: true });
    await expect(card.getByTestId("intro-about")).toHaveAttribute("href", "/about");
    // official information keeps its place above it
    const chip = await page.getByTestId("place-chip").boundingBox();
    expect((await card.boundingBox())!.y).toBeGreaterThan(chip!.y + chip!.height - 1);
    await card.getByTestId("intro-dismiss").click();
    await expect(card).toHaveCount(0);
    await page.reload();
    await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
    await page.waitForTimeout(300);
    await expect(page.getByTestId("intro-card")).toHaveCount(0);
    await ctx.close();
  });
});

test.describe("map labels", () => {
  test("label glyphs come from the site, so port markers draw while the font server is slow", async ({ browser }) => {
    test.setTimeout(90_000);
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    // OpenFreeMap's fonts answer only after 40 s; its tiles are normal
    await ctx.route(/tiles\.openfreemap\.org\/fonts\//, async (r) => {
      await new Promise((s) => setTimeout(s, 40_000));
      await r.continue().catch(() => {});
    });
    const page = await ctx.newPage();
    const local: string[] = [];
    page.on("response", (r) => {
      if (/\/fonts\/Noto%20Sans%20(Regular|Italic)\/\d+-\d+\.pbf$/.test(r.url()) && r.url().startsWith(OK)) local.push(`${r.status()}`);
    });
    const t0 = Date.now();
    await open(page, OK);
    await page.waitForFunction(
      () => {
        const m = (window as unknown as { __cwMap?: { getLayer(id: string): unknown; queryRenderedFeatures(o: unknown): unknown[] } }).__cwMap;
        return !!m?.getLayer("ports-circle") && m.queryRenderedFeatures({ layers: ["ports-circle"] }).length > 0;
      },
      null,
      { timeout: 25_000 },
    );
    expect(Date.now() - t0).toBeLessThan(30_000);
    expect(local.length).toBeGreaterThan(0);
    expect(local.every((s) => s === "200")).toBe(true);
    await ctx.close();
  });
});

test.describe("phone layouts", () => {
  test("bloom: the first measured value is on the first screen", async ({ browser }) => {
    const ctx = await browser.newContext(PHONE);
    const page = await ctx.newPage();
    await open(page, `${OK}/bloom`);
    const first = page.getByTestId("quantity-index").locator("[data-testid^=quantity-]").first();
    await expect(first).toBeVisible();
    const box = (await first.boundingBox())!;
    const bar = (await page.getByTestId("tabbar").boundingBox())!;
    // its name and value (the top of the column) sit above the tab bar without scrolling
    expect(box.y + 90).toBeLessThan(bar.y);
    // the notices row stays compact and keeps the state and the safety sentence
    const strip = page.getByTestId("bloom-official");
    expect((await strip.boundingBox())!.height).toBeLessThan(150);
    await expect(strip.getByTestId("official-verification")).toHaveAttribute("data-state", "unverified");
    await expect(strip).toContainText("A missing notice does not mean an area is open or seafood is safe.");
    await ctx.close();
  });

  test("fisheries: section links reach every part; how to read the chart is one tap away", async ({ browser }) => {
    const ctx = await browser.newContext(PHONE);
    const page = await ctx.newPage();
    await open(page, `${OK}/fisheries`);
    const jump = page.getByTestId("fisheries-jump");
    await expect(jump).toBeVisible();
    await jump.getByRole("link", { name: "Methods" }).click();
    await expect(page.locator("#methods-h")).toBeInViewport();
    const notes = page.getByTestId("chart-notes");
    await expect(notes.locator("summary")).toHaveText("How to read this chart");
    await notes.locator("summary").click();
    await expect(notes).toContainText("Each bar is one calendar year");
    // what the figures do not show stays open
    await expect(page.getByTestId("fisheries-limits")).toContainText("They do not show");
    await ctx.close();
  });
});

test.describe("keyboard and structure", () => {
  for (const path of ["/", "/bloom", "/fisheries", "/sources", "/about"]) {
    test(`${path}: one h1, and the first Tab offers a skip to the main content`, async ({ page }) => {
      await open(page, `${OK}${path}`);
      await expect(page.locator("h1")).toHaveCount(1);
      await page.keyboard.press("Tab");
      const skip = page.getByTestId("skip-link");
      await expect(skip).toBeFocused();
      await expect(skip).toBeVisible();
      await page.keyboard.press("Enter");
      await expect(page.locator("#main")).toBeFocused();
    });
  }
});

test("phone: 'All California' in Places keeps the sheet open on the list of regions", async ({ browser }) => {
  const ctx = await browser.newContext(PHONE);
  const page = await ctx.newPage();
  await open(page, OK);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: unknown }).__cwMap);
  await page.getByTestId("place-button").click();
  await page.getByTestId("nav-all").click();
  await page.getByTestId("region-sf_bay_farallones").click();
  await expect(page.getByTestId("place-button")).toContainText("San Francisco");
  await ctx.close();
});
