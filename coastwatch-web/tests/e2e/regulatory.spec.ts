import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/** While the notice registry is not human-verified, every legacy page says so in words on the trust pill, at
 *  every width, and the agencies are one tap away in its panel. The Del Norte razor clam records
 *  (2026-10-09) are listed, unverified. The v12 pages (/, Today, Explore, My Coast, Research) carry
 *  "not verified" on each official stamp instead (tests/e2e/v12-honesty.spec.ts). */
const WIDTHS: [number, number][] = [[1440, 900], [1024, 768], [768, 1024], [390, 844], [320, 640]];
for (const path of ["/fisheries", "/sources", "/about", "/ask", "/account", "/privacy", "/terms"]) {
  test(`${path}: the trust pill says "not verified" in words at every width and opens the agencies`, async ({ page }) => {
    for (const [width, height] of WIDTHS) {
      await page.setViewportSize({ width, height });
      await page.goto(`${OK}${path}`);
      const d = page.getByTestId("registry-disclosure").first();
      await expect(d, `${width}`).toBeVisible();
      // what is visible on the pill, not only what is in the page: an icon alone is not enough
      await expect(d, `${width}`).toContainText(/not verified/i, { useInnerText: true });
      await expect(d, `${width}`).toContainText(/preview/i, { useInnerText: true });
      await expect(d, `${width}`).toBeInViewport();
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(`${OK}${path}`);
    await expect(page.getByTestId("registry-disclosure").first()).toContainText("Research preview", { useInnerText: true });
    // the full statement and the agencies are one tap away, in the pill's panel
    await page.getByTestId("notices-button").click();
    const panel = page.getByTestId("notices-panel");
    await expect(panel).toContainText("may be incomplete");
    await expect(panel.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
    await expect(panel.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx");
  });
}

test("on a phone the pill is two short lines that still say not verified", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(`${OK}/sources`);
  const d = page.getByTestId("registry-disclosure").first();
  await expect(d).toContainText("Not verified", { useInnerText: true });
  expect((await d.boundingBox())!.height).toBeLessThan(44);
  await page.getByTestId("notices-button").click();
  await expect(page.getByTestId("notices-panel").getByRole("link", { name: "CDFW" })).toBeVisible();
  await ctx.close();
});

test("the Del Norte razor clam closure and CDPH warning are listed, not verified", async ({ page }) => {
  await page.goto(`${OK}/sources`);
  await page.getByTestId("notices-button").click();
  const drawer = page.getByTestId("notices-panel");
  await expect(drawer).toContainText("Recreational razor clam fishery closed in Del Norte County");
  await expect(drawer).toContainText("Do not eat sport-harvested razor clams from Del Norte County");
  await expect(drawer).toContainText("Not verified");
});
