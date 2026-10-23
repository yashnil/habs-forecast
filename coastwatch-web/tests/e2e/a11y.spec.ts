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
