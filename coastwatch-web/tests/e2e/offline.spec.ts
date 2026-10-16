import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Offline field pack (v4): the reader sees what will be saved and its size, downloads it, and the
 * pack is verified (every file, and the basemap's checksum against the published one). The proof
 * that it then opens with no network at all, after a browser restart, is scripts/offline-proof.mjs:
 * Playwright's offline emulation does not stop a service worker's own requests, so it cannot be
 * shown honestly inside this suite.
 */
test("the Monterey pack: plan, download, verify, remove", async ({ browser }) => {
  test.setTimeout(120_000);
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`${OK}/coast?place=port-550`);
  await page.evaluate(() => navigator.serviceWorker.register("/sw.js").then(() => navigator.serviceWorker.ready));
  await page.getByTestId("pack-open").click();
  const plan = page.getByTestId("pack-plan");
  await expect(plan).toContainText("Map of the region", { timeout: 60_000 });
  await expect(plan).toContainText("no day beyond its horizon");
  await expect(plan).toContainText("Not for navigation");
  const go = page.getByTestId("pack-download");
  await expect(go).toBeEnabled({ timeout: 60_000 });
  await go.click();
  const status = page.getByTestId("pack-status");
  await expect(status).toHaveAttribute("data-complete", "true", { timeout: 90_000 });
  await expect(status).toContainText("verified");
  const rec = await page.evaluate(async () => {
    const c = await caches.open("neritic-pack-monterey");
    return (await (await c.match("/__pack__/record.json"))!.json()) as { complete: boolean; verified: { basemapSha256: boolean; missing: number }; items: { url: string }[] };
  });
  expect(rec.verified).toMatchObject({ basemapSha256: true, missing: 0 });
  expect(rec.items.some((i) => i.url.endsWith("basemap.pmtiles"))).toBe(true);
  expect(rec.items.some((i) => i.url.startsWith("/api/evidence"))).toBe(true);
  // removing it is one action, and it is gone
  await page.getByTestId("pack-delete").click();
  await expect(page.getByTestId("pack-status")).toHaveCount(0);
  expect(await page.evaluate(() => caches.has("neritic-pack-monterey"))).toBe(false);
  await ctx.close();
});
