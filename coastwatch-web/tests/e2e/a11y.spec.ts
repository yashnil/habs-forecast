import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Accessibility across v4's workspaces and states (axe-core, WCAG 2.x A/AA): no serious or critical
 * violation on Explore with a harbour open, the notices panel, Research's tools, My Coast, and the
 * same on a phone. The map's WebGL canvas is excluded (its content is said in text beside it).
 */
const NOW = "2026-10-08T20:00:00Z";
const mapReady = (p: Page) => p.waitForFunction(() => !!(window as unknown as { __cwMap?: { loaded(): boolean } }).__cwMap?.loaded(), null, { polling: 250 });
async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

for (const [label, viewport] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]] as const) {
  test.describe(label, () => {
    test.use({ viewport, ...(label === "phone" ? { isMobile: true, hasTouch: true } : {}) });

    test("Explore with a harbour open", async ({ page }) => {
      await page.clock.setFixedTime(new Date(NOW));
      await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
      await mapReady(page);
      expect(await serious(page)).toEqual([]);
    });

    test("the official notices panel", async ({ page }) => {
      await page.clock.setFixedTime(new Date(NOW));
      await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
      await mapReady(page);
      await page.getByTestId("notices-button").click();
      await expect(page.getByTestId("notices-panel")).toBeVisible();
      expect(await serious(page)).toEqual([]);
    });

    test("Research with its map tools", async ({ page }) => {
      await page.clock.setFixedTime(new Date(NOW));
      await page.goto(`${OK}/research?station=HABs-SantaCruzWharf`);
      await page.getByTestId("research-tab-grid").click();
      await page.getByTestId("research-grid").scrollIntoViewIfNeeded();
      expect(await serious(page)).toEqual([]);
    });

    test("My Coast briefing", async ({ page }) => {
      await page.clock.setFixedTime(new Date(NOW));
      await page.goto(`${OK}/coast?place=port-593&for=commercial&species=dungeness`);
      await page.getByTestId("matters").waitFor();
      expect(await serious(page)).toEqual([]);
    });
  });
}

test("daylight mode: switched from the ⋯ menu, kept on this device, applied before the first paint, darker tokens, decks over the map on light paper, no axe violation", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("coast-title")).toHaveText("Monterey");
  const token = (name: string) => page.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), name);
  expect(await token("--nx-accent")).toBe("#086a85");
  await page.getByTestId("more-menu").click();
  const sw = page.getByTestId("daylight-toggle");
  await expect(sw).toHaveAttribute("role", "switch");
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");
  await expect(page.locator("html")).toHaveAttribute("data-daylight", "");
  expect(await page.evaluate(() => localStorage.getItem("nx-daylight-v1"))).toBe("1");
  for (const [n, v] of [["--nx-accent", "#04566c"], ["--nx-official", "#7f3a00"], ["--nx-model", "#55309a"], ["--nx-measured", "#055a50"], ["--nx-conditions", "#164f78"], ["--nx-ok", "#0f5e3b"], ["--nx-stale", "#6b4500"], ["--nx-down", "#922a16"], ["--nx-gap", "#45565f"], ["--nx-line", "#8a9ba5"]]) expect(await token(n), n).toBe(v);
  await page.keyboard.press("Escape");
  expect(await serious(page)).toEqual([]);
  // kept: a reload has it before any script of the app runs (no darker flash first)
  await page.reload({ waitUntil: "commit" });
  await page.waitForLoadState("domcontentloaded");
  expect(await page.evaluate(() => document.documentElement.hasAttribute("data-daylight"))).toBe(true);
  // Explore: the decks over the map read on light paper; the map stays the map
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  await mapReady(page);
  const deckBg = await page.getByTestId("explore").evaluate((el) => getComputedStyle(el).getPropertyValue("--nx-surface").trim());
  expect(deckBg).toMatch(/^#fff(fff)?$/);
  expect(await serious(page)).toEqual([]);
  // and off again, for good
  await page.getByTestId("more-menu").click();
  await page.getByTestId("daylight-toggle").click();
  await expect(page.locator("html")).not.toHaveAttribute("data-daylight", "");
  expect(await page.evaluate(() => localStorage.getItem("nx-daylight-v1"))).toBeNull();
  await ctx.close();
});
