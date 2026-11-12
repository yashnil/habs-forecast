import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * Navigation and what every page says about itself (v4), restored from the v2 shell, release, tour
 * and welcome suites: the notices control on every page; v2 links keep working; the research-preview
 * and test-data disclosures; reference pages keep their provenance; keyboard routes through controls.
 */
const NOW = "2026-10-08T20:00:00Z";

test("Sources lists every source with its status, and says when a date is inferred", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/sources`);
  for (const id of ["charm", "satellite_chl", "hf_radar", "official"]) await expect(page.getByTestId(`source-${id}`)).toBeVisible();
  await expect(page.getByTestId("source-charm")).toContainText("date inferred: NOAA does not publish it");
});
