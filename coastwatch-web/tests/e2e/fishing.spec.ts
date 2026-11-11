import { expect, test } from "@playwright/test";
import { OK } from "./ports";

/**
 * v5 fishermen audit: the trip briefing answers "is it in season here, what is the sea doing, what has
 * it been worth" at a glance, each answer quoted from its agency with its date, and never as Neritic's
 * verdict (no "open to fish", no "safe", no go/no-go).
 */

test("fisheries: each species shows CDFW's lists region by region and its dock value", async ({ page }) => {
  // the species panel is the Coastal layout (the suite's stored mode is Research)
  await page.addInitScript(() => window.localStorage.setItem("cw-mode-v1", "coastal"));
  await page.goto(`${OK}/fisheries`);
  await page.getByTestId("fish-species-groundfish").click();
  const f = page.getByTestId("fish-seasons");
  for (const r of ["northern", "mendocino", "san-francisco", "central", "southern"]) await expect(f.getByTestId(`fish-season-${r}`)).toBeVisible();
  await expect(f.getByTestId("fish-season-southern")).toContainText("alert");
  await page.getByTestId("fish-species-dungeness").click();
  await expect(page.getByTestId("fish-worth")).toContainText("2024: Dungeness Crab $49.7M");
});
