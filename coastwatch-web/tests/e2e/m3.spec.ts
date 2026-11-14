import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

const FIX = path.resolve(__dirname, "../fixture-data/v1");
const read = (prefix: string) => JSON.parse(readFileSync(path.join(FIX, readdirSync(FIX).find((f) => f.startsWith(prefix))!), "utf8"));
const fish = read("fisheries-");
const NOW = "2026-10-08T20:00:00Z";

async function open(page: Page, url: string, now = NOW) {
  await page.clock.setFixedTime(new Date(now));
  await page.goto(url);
}

/** Every rendered sentence that says "safe", "open" or "loss" must be negated. */
async function assertNoUnsafeClaims(page: Page) {
  const text = await page.locator("body").innerText();
  const sentences = text.split(/(?<=[.!?])\s+|\n+/);
  const bad = sentences.filter(
    (s) => /\b(safe|open|all clear|losse?s?)\b/i.test(s) && !/\b(not|no|never|does not|isn't)\b/i.test(s) && !/Whale.?safe/i.test(s),
  );
  expect(bad, bad.join(" | ")).toEqual([]);
}


test.describe("Fisheries & Economic Exposure", () => {
  test("figures match the published data; units and tiers switch; port level unavailable", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await expect(page.getByTestId("exposure-definition")).toContainText("not a prediction of losses");
    await expect(page.getByTestId("port-level-unavailable")).toContainText("Port-level values are not available");
    await page.getByTestId("port-level-why").locator("summary").click();
    await expect(page.getByTestId("port-level-why")).toContainText("permission");
    const y = fish.years.at(-1);
    const t1 = fish.groups.filter((g: { tier: number }) => g.tier === 1);
    const total = (gs: typeof t1, k: string) => gs.reduce((s: number, g: { annual: { year: number }[] }) => s + ((g.annual.find((a) => a.year === y) as Record<string, number>)[k] ?? 0), 0);
    const fmt = (v: number) => `$${(v / 1e6).toFixed(1)}M`;
    await expect(page.getByTestId("figure-year")).toContainText(fmt(total(t1, "dollars_real")));
    await page.getByTestId("tier-12").click();
    await expect(page.getByTestId("figure-year")).toContainText(fmt(total(fish.groups, "dollars_real")));
    await page.getByTestId("unit-nominal").click();
    await expect(page.getByTestId("figure-year")).toContainText(fmt(total(fish.groups, "dollars_nominal")));
    await page.getByTestId("values-disclosure").locator("summary").click();
    await expect(page.getByTestId("withheld-row")).toContainText("not attributed");
    await page.getByTestId("deflator").locator("summary").click();
    await expect(page.getByTestId("deflator")).toContainText("missing M10");
    await expect(page.getByTestId("data-through")).toContainText(String(y));
    await assertNoUnsafeClaims(page);
  });
});

test.describe("accessibility", () => {
  for (const [name, url] of [
    ["map", "/"],
    ["bloom", "/bloom"],
    ["fisheries", "/fisheries"],
    ["sources", "/sources"],
    ["about", "/about"],
  ] as const) {
    test(`${name}: no serious or critical axe violations`, async ({ page }) => {
      await open(page, `${OK}${url}`);
      await page.waitForLoadState("networkidle");
      const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
      const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
    });
  }
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  test("fisheries: nothing overflows horizontally", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    await expect(page.getByTestId("fisheries-figures")).toBeVisible();
    await expect(page.getByTestId("landings-chart")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  });
});
