import { expect, test } from "@playwright/test";
import { OK } from "../ports";
import { newContext, seedTrip } from "./helpers";

/**
 * The workspace at desk and laptop widths: the departure windows dock and the trip's ticket never overlap, and the
 * ticket's state and its one filled action are readable (never left at zero opacity by a swap).
 */
test.use({ storageState: { cookies: [], origins: [] } });

for (const width of [1440, 1280, 1100, 1024]) {
  test(`layout at ${width} px: the departure dock stops before the ticket`, async ({ browser }) => {
    const seeded = await seedTrip({ id: `l-${width}` });
    const { ctx, page, errors } = await newContext(browser, { width, height: 800 }, seeded.storage);
    await page.goto(`${OK}/coast/trips?id=l-${width}`);
    const ticket = page.getByTestId("trip-workspace");
    await expect(ticket).toBeVisible({ timeout: 20_000 });
    const dock = page.getByTestId("trip-departures").locator("xpath=ancestor-or-self::*[@data-align][1]");
    await expect(dock).toBeVisible();
    const [d, t] = [await dock.boundingBox(), await ticket.boundingBox()];
    expect(d!.x + d!.width, "the dock's right edge is left of the ticket").toBeLessThanOrEqual(t!.x - 8);
    for (const id of ["trip-lifecycle", "trip-next"]) {
      const el = page.getByTestId(id).first();
      await expect(el).toBeVisible();
      expect(Number(await el.evaluate((n) => { let o = 1; for (let e: Element | null = n; e; e = e.parentElement) o *= Number(getComputedStyle(e).opacity); return o; }))).toBeGreaterThan(0.95);
    }
    expect(errors).toEqual([]);
    await ctx.close();
  });
}
