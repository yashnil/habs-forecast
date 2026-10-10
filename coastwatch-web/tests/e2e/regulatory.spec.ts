import { expect, test } from "@playwright/test";

/** While the notice registry is not human-verified, every page says so and links to the agencies;
 *  the Del Norte razor clam records (2026-10-09) are listed, unverified. */
const OK = "http://localhost:3200";

for (const path of ["/", "/bloom", "/fisheries", "/sources"]) {
  test(`${path}: the unverified-notices disclosure is on the page with agency links`, async ({ page }) => {
    await page.goto(`${OK}${path}`);
    const d = page.getByTestId("registry-disclosure");
    await expect(d).toContainText("Official notices are not verified.");
    await expect(d).toContainText("may be incomplete");
    await expect(d.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories");
    await expect(d.getByRole("link", { name: "CDPH" })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx");
  });
}

test("on a phone the disclosure is one short line that still links to the agencies", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto(OK);
  const d = page.getByTestId("registry-disclosure");
  await expect(d).toContainText("Notices not verified");
  expect((await d.boundingBox())!.height).toBeLessThan(60);
  await expect(d.getByRole("link", { name: "CDFW" }).first()).toBeVisible();
  await ctx.close();
});

test("the Del Norte razor clam closure and CDPH warning are listed, not verified", async ({ page }) => {
  await page.goto(OK);
  await page.getByTestId("registry-disclosure").getByRole("button", { name: "See the list" }).click();
  const drawer = page.getByTestId("official-drawer");
  await expect(drawer).toContainText("Recreational razor clam fishery closed in Del Norte County");
  await expect(drawer).toContainText("Do not eat sport-harvested razor clams from Del Norte County");
  await expect(drawer).toContainText("Not verified");
});
