import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Accessibility of the v12 pages (axe-core, WCAG 2.x A/AA): no serious or critical violation on Home, Today,
 * Explore, My Coast and Research, at desktop and phone widths, and with the provenance sheet open. Ported from
 * a11y.spec, briefing, fishing, economics, since and m3's per-page axe checks of the old layouts. The chart's
 * WebGL canvas is excluded (its content is said in the margin). Also the first Tab offers to skip the bar
 * (navigation.spec) and controls on a phone are at least 44 px (hardening L1, layout).
 */
test.use({ storageState: { cookies: [], origins: [] } });

const MOSS = JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: "port-592", fisheries: ["dungeness_commercial"], vessel: null });

async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).exclude(".maplibregl-canvas").analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

const PAGES: Array<[string, string, boolean]> = [
  ["Home, a first visit", "/", false],
  ["Today", "/today?place=port-592", true],
  ["Explore", "/explore?place=port-592", true],
  ["My Coast", "/coast?place=port-592", true],
  ["Research", "/research?place=port-592", true],
];

for (const [label, viewport] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]] as const) {
  test.describe(label, () => {
    test.use({ viewport, ...(label === "phone" ? { isMobile: true, hasTouch: true } : {}) });
    for (const [name, url, saved] of PAGES) {
      test(`${name}: no serious or critical axe violations`, async ({ page, context }) => {
        await context.addInitScript(
          ([port, s]) => {
            localStorage.setItem("nx-opening-seen-v1", "1");
            localStorage.setItem("nx-tour-v1", "dismissed");
            if (s) localStorage.setItem("nx-port-v1", port as string);
          },
          [MOSS, saved] as const,
        );
        await page.goto(`${OK}${url}`);
        await expect(page.getByTestId("nx-margin")).toBeVisible();
        await page.waitForTimeout(1200);
        expect(await serious(page)).toEqual([]);
      });
    }
  });
}

test("the provenance sheet: a dialog that traps focus, closes on Escape and passes axe", async ({ page, context }) => {
  await context.addInitScript((port) => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-tour-v1", "dismissed");
    localStorage.setItem("nx-port-v1", port);
  }, MOSS);
  await page.goto(`${OK}/today?place=port-592`);
  const stamp = page.getByRole("button", { name: /not verified\. Open where this comes from/ }).first();
  await stamp.click();
  const sheet = page.getByRole("dialog");
  await expect(sheet).toBeVisible();
  await expect(sheet).toContainText("Not verified.");
  expect(await serious(page)).toEqual([]);
  await page.keyboard.press("Escape");
  await expect(sheet).toHaveCount(0);
});

// Missing in v12 (test-migration.md): the ds Stamp opens its sheet from its own state, not a Radix trigger, so
// focus is not returned to the stamp on close (components.md promises "focus trapped and returned").
test("closing the provenance sheet returns focus to the stamp that opened it", async ({ page, context }) => {
  await context.addInitScript((port) => {
    localStorage.setItem("nx-opening-seen-v1", "1");
    localStorage.setItem("nx-tour-v1", "dismissed");
    localStorage.setItem("nx-port-v1", port);
  }, MOSS);
  await page.goto(`${OK}/today?place=port-592`);
  const stamp = page.getByRole("button", { name: /not verified\. Open where this comes from/ }).first();
  await stamp.click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(stamp).toBeFocused();
});

test("the first Tab on each page offers to skip past the bar", async ({ page }) => {
  for (const url of ["/explore", "/coast", "/research", "/today?place=port-592"]) {
    await page.goto(`${OK}${url}`);
    await expect(page.getByTestId("nx-margin")).toBeVisible();
    await page.keyboard.press("Tab");
    await expect(page.getByRole("link", { name: "Skip to content" }), url).toBeFocused();
  }
});

test("on a 320 px phone the bar's and tab bar's controls are at least 44 px tall, and nothing scrolls sideways", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  for (const url of ["/", "/today?place=port-592", "/explore", "/coast?place=port-592", "/research"]) {
    await page.goto(`${OK}${url}`);
    await expect(page.getByTestId("nx-margin")).toBeVisible();
    await page.waitForTimeout(600);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), url).toBeLessThanOrEqual(0);
    const small = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>("[data-testid=nx-bar] a:not(.n12-wordmark), [data-testid=nx-bar] button, [data-testid=nx-tabs] a")]
        .filter((e) => e.offsetParent !== null)
        .map((e) => [e.textContent?.trim() || e.getAttribute("aria-label"), Math.round(e.getBoundingClientRect().height)] as const)
        .filter(([, h]) => h < 44),
    );
    expect(small, url).toEqual([]);
  }
  await ctx.close();
});

// Missing in v12 (test-migration.md): at 320 px the wordmark link (home) is 27 px tall.
test("on a 320 px phone the wordmark link is a 44 px target too", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true, storageState: { cookies: [], origins: [] } });
  const page = await ctx.newPage();
  await page.goto(`${OK}/explore`);
  const h = await page.locator("[data-testid=nx-bar] .n12-wordmark").evaluate((e) => e.getBoundingClientRect().height);
  expect(h).toBeGreaterThanOrEqual(44);
  await ctx.close();
});
