import { expect, test } from "@playwright/test";

test("the data sources page cites the published research and says how it relates to the app", async ({ page }) => {
  await page.goto("http://localhost:3200/sources");
  const r = page.getByTestId("research");
  await expect(r).toContainText("Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System");
  await expect(r.getByRole("link", { name: "doi:10.33422/ccgconf.v2i2.1619" })).toHaveAttribute("href", "https://doi.org/10.33422/ccgconf.v2i2.1619");
  await expect(r).toContainText("CoastWatch does not run those research models");
});
