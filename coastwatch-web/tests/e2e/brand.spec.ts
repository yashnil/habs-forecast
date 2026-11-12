import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * The product is Neritic (formerly CoastWatch). NOAA CoastWatch, the NOAA program whose data
 * it uses, keeps its name everywhere it is credited. Stable identifiers (storage keys, links, the
 * data URL) are unchanged, so saved settings and shared links keep working.
 */
test.describe("Neritic brand", () => {
  test("About says plainly who makes it, the old name, and that it is not NOAA's", async ({ page }) => {
    await page.goto(`${OK}/about`);
    const p = page.getByTestId("about-independence");
    await expect(p).toContainText("Neritic is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.");
    await expect(p).toContainText("was called CoastWatch until October 2026");
    await expect(p).toContainText("not NOAA CoastWatch");
    await expect(page.getByTestId("footer-independence")).toContainText("not affiliated with or endorsed by NOAA");
  });
});
