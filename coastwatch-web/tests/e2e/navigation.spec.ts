import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { openLayers } from "./helpers";
import { OK } from "./ports";

/**
 * Navigation and what every page says about itself (v4), restored from the v2 shell, release, tour
 * and welcome suites: the notices control on every page; v2 links keep working; the research-preview
 * and test-data disclosures; reference pages keep their provenance; keyboard routes through controls.
 */
const NOW = "2026-10-08T20:00:00Z";
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const official = JSON.parse(readFileSync(path.join(FIX, manifest.official_url), "utf8"));
const active = official.registry.records.filter((r: { status: string }) => r.status === "active");

test("every page has the notices control with the count and the unverified state, and it opens every notice", async ({ page }) => {
  for (const p of ["/", "/coast?place=port-550", "/research?place=port-550", "/bloom", "/fisheries", "/sources", "/about"]) {
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}${p}`);
    const b = page.getByTestId("notices-button");
    await expect(b, p).toHaveAttribute("data-state", "unverified");
    await b.click();
    const panel = page.getByTestId("notices-panel");
    if (await page.getByTestId("notices-here").count()) {
      // a place is open: its own notices lead, the rest follow; together, every active notice once
      const here = Number(await page.getByTestId("notices-count").innerText());
      await expect(panel.getByTestId("notices-local"), p).toContainText(`· ${here}`);
      await expect(panel.locator('li[data-testid^="notice-"]:not([data-disputed])'), p).toHaveCount(active.length);
    } else {
      await expect(page.getByTestId("notices-count"), p).toHaveText(String(active.length));
      await expect(page.getByTestId("notices-current").locator(":scope > li"), p).toHaveCount(active.length);
    }
    await page.keyboard.press("Escape");
  }
});

test("Explore, My Coast and Research say they are a research preview, and label recorded test data", async ({ page }) => {
  for (const p of ["/?layer=forecast", "/coast?place=port-550", "/research?place=port-550"]) {
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}${p}`);
    // the long form links to About; Explore's compact form says the words only
    await expect(page.locator('[data-testid^="preview-disclosure"]:visible').first(), p).toContainText("Research preview");
    await expect(page.getByTestId("fixture-banner").first(), p).toContainText("Test data");
  }
  for (const p of ["/bloom", "/fisheries", "/sources", "/about"]) {
    await page.goto(`${OK}${p}`);
    await expect(page.getByTestId("footer-disclaimer"), p).toContainText("does not replace CDFW or CDPH");
  }
});

test("v2 links open the same view: a port, a region, a research-mode briefing", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?region=monterey_bay&port=550&layer=forecast`);
  await expect(page.getByTestId("inspect-title")).toHaveText("Monterey");
  // v10: the layer index (always in view on a wide screen) lights the layer and names it honestly
  await expect(page.getByTestId("layer-deck")).toBeVisible();
  await expect(page.getByTestId("layer-forecast")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-forecast")).toContainText("Bloom & toxin probability");
  await expect(page.getByTestId("layer-forecast")).toContainText("Model · NOAA C-HARM");
  await page.goto(`${OK}/coast?place=port-550&mode=research`);
  await expect(page).toHaveURL(/\/research\?place=port-550/);
});

test("About is reachable from every workspace and states the kinds of information, the limits and the research", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await page.getByRole("button", { name: "More pages" }).click();
  await page.getByRole("link", { name: /About/ }).first().click();
  await expect(page).toHaveURL(/\/about$/);
  await expect(page.getByTestId("about-kinds")).toContainText("Official");
  await expect(page.getByTestId("about-cannot")).toContainText("does not mean an area is open or seafood is safe");
  await expect(page.getByTestId("publication-doi")).toHaveAttribute("href", "https://doi.org/10.33422/ccgconf.v2i2.1619");
  await expect(page.getByTestId("research")).toContainText("matches, but does not beat");
  // the old tour link now opens the map itself
  await expect(page.getByTestId("about-tour")).toHaveAttribute("href", "/");
});

test("Sources lists every source with its status, and says when a date is inferred", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/sources`);
  for (const id of ["charm", "satellite_chl", "hf_radar", "official"]) await expect(page.getByTestId(`source-${id}`)).toBeVisible();
  await expect(page.getByTestId("source-charm")).toContainText("date inferred: NOAA does not publish it");
});

test("the bloom page links each station's original data on SCCOOS ERDDAP, encoded, and says it is not the chart", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/bloom`);
  const link = page.getByTestId("source-csv-link");
  const href = (await link.getAttribute("href"))!;
  expect(href).toMatch(/^https:\/\/erddap\.sccoos\.org\/erddap\/tabledap\/HABs-[A-Za-z]+\.csv\?time,latitude,longitude,/);
  expect(href).not.toMatch(/["<>\s]|%25/);
  await expect(link).toHaveAttribute("rel", /noopener/);
  await page.goto(`${OK}/bloom?station=HABs-MorroBayFrontBay`);
  await expect(page.getByTestId("source-csv-link")).toHaveAttribute("href", /\/tabledap\/HABs-MorroBayFrontBay\.csv\?/);
});

test("the layer list moves with the arrow keys, and county lines and names are always drawn (no switch)", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=none`);
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  await openLayers(page);
  await page.getByTestId("layer-none").focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("layer-forecast")).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("layer-forecast")).toBeFocused();
  await expect(page.getByTestId("county-note")).toContainText("do not define fishery closures");
  // part of the basemap: no checkbox or map control, and every county layer is on
  await expect(page.getByTestId("county-lines")).toHaveCount(0);
  await expect(page.getByTestId("county-labels")).toHaveCount(0);
  await expect(page.getByTestId("counties-toggle")).toHaveCount(0);
  const vis = () =>
    page.evaluate(() => {
      const m = (window as unknown as { __cwMap: { getLayer(l: string): unknown; getLayoutProperty(l: string, p: string): string | undefined } }).__cwMap;
      return ["county-borders-casing", "county-borders", "county-labels"].map((l) => (m.getLayer(l) ? (m.getLayoutProperty(l, "visibility") ?? "visible") : "missing"));
    });
  await expect.poll(vis).toEqual(["visible", "visible", "visible"]);
  // a choice remembered by an earlier version is ignored
  await page.evaluate(() => localStorage.setItem("cw-counties-v1", JSON.stringify({ lines: false, labels: false })));
  await page.reload();
  await page.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded());
  await expect.poll(vis).toEqual(["visible", "visible", "visible"]);
  // names are placed with collision handling, under towns and harbours (which claim their place first)
  const order = await page.evaluate(() => {
    const ids = (window as unknown as { __cwMap: { getStyle(): { layers: { id: string }[] } } }).__cwMap.getStyle().layers.map((l) => l.id);
    return { county: ids.indexOf("county-labels"), city: ids.indexOf("place-city"), ports: ids.indexOf("ports-label") };
  });
  expect(order.county).toBeLessThan(order.city);
  expect(order.county).toBeLessThan(order.ports);
});

test("with a harbour selected and on a phone, the notices stay one tap away", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:593`);
  await expect(page.getByTestId("registry-disclosure").first()).toBeVisible();
  await page.getByTestId("notices-button").click();
  await expect(page.getByTestId("notices-panel").locator('[data-testid^="notice-"]').first()).toBeVisible();
  await ctx.close();
});

test("the first Tab on each workspace offers to skip past the bar", async ({ page }) => {
  for (const p of ["/", "/coast?place=port-550", "/research?place=port-550"]) {
    await page.clock.setFixedTime(new Date(NOW));
    await page.goto(`${OK}${p}`);
    await page.keyboard.press("Tab");
    const skip = page.getByTestId("skip-link");
    await expect(skip, p).toBeFocused();
    await expect(skip, p).toBeVisible();
    const target = (await skip.getAttribute("href"))!.slice(1);
    await expect(page.locator(`#${target}`), p).toHaveCount(1);
  }
});
