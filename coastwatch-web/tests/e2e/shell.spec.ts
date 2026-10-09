import { readFileSync } from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

/** P0 shell: masthead, official pill and drawer, mobile tab bar, fonts, light/dark themes. */
const FIX = path.resolve(__dirname, "../fixture-data/v1");
const manifest = JSON.parse(readFileSync(path.join(FIX, "manifest.json"), "utf8"));
const official = JSON.parse(readFileSync(path.join(FIX, manifest.official_url), "utf8"));
const active = official.registry.records.filter((r: { status: string }) => r.status === "active");

const OK = "http://localhost:3200";
const NODATA = "http://localhost:3202";
const PAGES = [
  ["map", "/"],
  ["bloom", "/bloom"],
  ["fisheries", "/fisheries"],
  ["sources", "/sources"],
] as const;

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date("2026-10-08T20:00:00Z"));
  await page.goto(url);
  await expect(page.getByTestId("official-pill")).not.toHaveAttribute("data-state", "checking");
}

test.describe("masthead and official drawer", () => {
  for (const [name, url] of PAGES) {
    test(`${name}: the official pill shows the count and verification state and opens every notice`, async ({ page }) => {
      await open(page, `${OK}${url}`);
      const pill = page.getByTestId("official-pill");
      await expect(pill).toContainText(String(active.length));
      await expect(pill).toContainText("Not verified");
      await pill.click();
      const drawer = page.getByRole("dialog", { name: /active notices in California/ });
      await expect(drawer).toBeVisible();
      for (const r of active) await expect(drawer.getByTestId(`notice-${r.id}`)).toBeVisible();
      await expect(drawer.getByTestId("official-verification")).toHaveAttribute("data-state", "unverified");
      await expect(drawer).toContainText("does not mean an area is open");
      await expect(drawer.locator("a[href^='tel:']")).toHaveCount(2);
      await page.keyboard.press("Escape");
      await expect(drawer).toHaveCount(0);
      await expect(pill).toBeFocused();
    });
  }

  test("type: display serif for headings, IBM Plex for the interface", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    expect(await page.evaluate(() => getComputedStyle(document.body).fontFamily)).toMatch(/Plex Sans/i);
    await page.getByTestId("official-pill").click();
    expect(await page.locator("#official-drawer-h").evaluate((n) => getComputedStyle(n).fontFamily)).toMatch(/Newsreader/i);
  });

  test("reading pages are on paper; the map keeps the dark sea", async ({ page }) => {
    await open(page, `${OK}/fisheries`);
    expect(await page.locator("main").evaluate((n) => getComputedStyle(n).backgroundColor)).not.toBe("rgb(4, 11, 23)");
    expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).toBe("rgb(245, 243, 238)");
    await open(page, `${OK}/`);
    await expect(page.locator("main")).toHaveClass(/theme-dark/);
  });

  test("with no data, the pill and drawer still route people to the agencies", async ({ page }) => {
    await page.goto(`${NODATA}/`);
    const pill = page.getByTestId("official-pill");
    await expect(pill).toContainText("Unavailable");
    await pill.click();
    const drawer = page.getByRole("dialog");
    await expect(drawer).toContainText("Official notices unavailable");
    await expect(drawer.locator("a[href='https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories']")).toBeVisible();
    await expect(drawer.locator("a[href^='tel:']")).toHaveCount(2);
  });
});

async function axeClean(page: Page) {
  const r = await new AxeBuilder({ page }).exclude(".maplibregl-canvas").exclude(".maplibregl-ctrl-attrib").analyze();
  const bad = r.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
}

test("drawer open: no serious or critical axe violations", async ({ page }) => {
  await open(page, `${OK}/bloom`);
  await page.waitForLoadState("networkidle");
  await page.getByTestId("official-pill").click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await axeClean(page);
});

test.describe("mobile shell", () => {
  test.use({ viewport: { width: 390, height: 844 } });
  for (const [name, url] of PAGES) {
    test(`${name}: tab bar with Notices, no horizontal overflow`, async ({ page }) => {
      await open(page, `${OK}${url}`);
      const bar = page.getByTestId("tabbar");
      await expect(bar).toBeVisible();
      await expect(page.getByRole("navigation", { name: "Primary" }).filter({ visible: true })).toHaveCount(1);
      if (name !== "sources") await expect(page.getByTestId(`tab-${name}`)).toHaveAttribute("aria-current", "page");
      await expect(page.getByTestId("tab-notices")).toContainText(String(active.length));
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
      await page.getByTestId("tab-notices").click();
      await expect(page.getByRole("dialog")).toBeVisible();
    });
  }

  test("map and bloom on a phone: no serious or critical axe violations", async ({ page }) => {
    for (const url of ["/", "/bloom"]) {
      await open(page, `${OK}${url}`);
      await page.waitForLoadState("networkidle");
      await axeClean(page);
    }
  });

  test("map: the slide-up sheet sits above the tab bar", async ({ page }) => {
    await open(page, `${OK}/`);
    const sheet = await page.getByTestId("mobile-sheet").boundingBox();
    const bar = await page.getByTestId("tabbar").boundingBox();
    expect(sheet!.y + sheet!.height).toBeLessThanOrEqual(bar!.y + 1);
    await expect(page.getByTestId("sheet-handle")).toBeVisible();
  });
});
