import { expect, test } from "@playwright/test";
import { NODATA, OK } from "./ports";

/**
 * "Since you last checked" and kept briefings (fishermen-first workstream), on My Coast in Coastal
 * mode (mounted under the answer line). Fixture data; nothing is fetched from a third party.
 */
const CRAB = "/coast?place=port-201&for=commercial&species=dungeness";
const coastal = (origin: string) => ({ origin, localStorage: [{ name: "cw-intro-dismissed-v1", value: "1" }, { name: "cw-mode-v1", value: "coastal" }] });
test.use({ storageState: { cookies: [], origins: [coastal(OK), coastal(NODATA)] } });

test.describe("since you last checked", () => {
  test("when Neritic data are unavailable nothing is listed or claimed", async ({ page }) => {
    await page.goto(`${NODATA}${CRAB}`);
    await page.waitForTimeout(1500);
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
    expect((await page.request.get(`${NODATA}/api/briefing?place=port-201`)).status()).toBe(503);
  });

  test("a v2 Research-mode link opens the Research workspace, which has no such block", async ({ page }) => {
    await page.goto(`${OK}${CRAB}&mode=research`);
    await expect(page.getByTestId("research-title")).toBeVisible();
    await page.waitForTimeout(1500);
    await expect(page.getByTestId("since-last-check")).toHaveCount(0);
  });
});
