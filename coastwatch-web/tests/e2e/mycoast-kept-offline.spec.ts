import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * A kept briefing (My Coast, "Keep a dated copy", and the home port's own copy) opens offline (v10 pass 2):
 * keeping it stores /coast/kept and its code in the service worker's kept cache; offline, that page reloads
 * from the device, and every other page's offline state links to it, in the site's own style, with the
 * agencies' tap-to-call lines and "no notice listed does not mean open or safe".
 */
const SANTA_CRUZ = `${OK}/coast?place=port-593&for=commercial&species=salmon`;

test("a kept briefing opens offline: /coast/kept reloads from the device, and the offline page links to it with the agencies' lines", async ({ browser }) => {
  test.setTimeout(90_000);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(SANTA_CRUZ);
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await page.reload(); // controlled by the worker from here
  await page.getByTestId("kept-make").click();
  await expect(page.getByTestId("kept-offline")).toBeVisible({ timeout: 30_000 });
  await ctx.setOffline(true);
  await page.goto(`${OK}/coast/kept`);
  await expect(page.getByTestId("kept-title")).toHaveText("Santa Cruz · Commercial fishing · Salmon");
  await expect(page.getByTestId("kept-warning")).toContainText("This is a saved copy, not current information");
  await expect(page.getByTestId("kept-agencies")).toContainText("No notice listed does not mean open or safe");
  await expect(page.getByTestId("kept-tel-cdfw")).toHaveAttribute("href", "tel:+18316492883");
  // a page that was not kept says so, in the site's own style, and leads to the kept copy
  await page.goto(SANTA_CRUZ);
  await expect(page.getByTestId("offline-page")).toBeVisible();
  await expect(page.getByTestId("offline-kept")).toContainText("Santa Cruz · Commercial fishing · Salmon");
  await expect(page.locator('a[href="tel:+18005534133"]')).toBeVisible();
  await expect(page.getByTestId("offline-page")).toContainText("No notice listed does not mean open or safe");
  await page.getByTestId("offline-kept").click();
  await expect(page.getByTestId("kept-title")).toBeVisible();
  await ctx.setOffline(false);
  await ctx.close();
});
