import { expect, test, type Page } from "@playwright/test";
import { FAILED, OK } from "./ports";

/**
 * An agency release appears to lift a notice the registry still lists as active (CDPH SN26-021 and
 * SN26-018, Oct 2026). Until a person reviews it, the notice is never shown as in effect: it is set
 * apart, uncounted, and always shown with the release. The registry itself is not changed.
 */
const NOW = "2026-10-12T20:00:00Z";
const ID = "cdph-2026-sn26-018-monterey-bivalves";

async function open(page: Page, url: string) {
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(url);
}

test.describe("agency source discrepancy", () => {
  test("the notices drawer sets the reported-lifted notice apart and does not count it", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-550`);
    await page.getByTestId("official-pill").click();
    const drawer = page.getByTestId("official-drawer");
    const disputed = drawer.getByTestId("drawer-disputed");
    await expect(disputed).toContainText("Reported lifted by the agency · awaiting review · 1");
    const card = disputed.getByTestId(`notice-${ID}`);
    await expect(card).toHaveAttribute("data-disputed", "true");
    await expect(card.getByTestId("notice-critical")).toContainText("SN26-021");
    await expect(card.getByRole("link", { name: /SN26-021/ })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx");
    // it appears once, in the disputed group only, and the heading counts the others
    await expect(drawer.getByTestId(`notice-${ID}`)).toHaveCount(1);
    const pill = Number(await page.getByTestId("official-pill").locator("b").innerText());
    await expect(drawer.locator("#official-drawer-h")).toHaveText(`${pill} active notices in California`);
    await expect(drawer).toContainText("release SN26-021 appears to lift a notice the list still shows as active");
  });

  test("My Coast never lists it as applying to Monterey; it says the agency reports it lifted", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-550`);
    const group = page.getByTestId("coast-notices-disputed");
    await expect(group).toContainText("Reported lifted by the agency · awaiting review");
    await expect(group.getByRole("link", { name: /SN26-021/ })).toHaveAttribute("href", /SN26-021\.aspx$/);
    await expect(page.getByTestId("coast-notices-geo").getByTestId(`coast-notice-${ID}`)).toHaveCount(0);
    await expect(page.getByTestId(`brief-disputed-${ID}`)).toContainText("Reported lifted by CDPH · awaiting review");
    await expect(page.getByTestId(`brief-notice-${ID}`)).toHaveCount(0);
    await expect(page.getByTestId("coast-unknowns")).toContainText("release SN26-021 appears to lift");
  });

  test("the map's port panel flags it where it is listed", async ({ page }) => {
    await open(page, `${FAILED}/?region=monterey_bay&port=550`);
    const card = page.getByTestId("port-official").getByTestId(`notice-${ID}`);
    await expect(card).toHaveAttribute("data-disputed", "true");
    await expect(card.getByTestId("notice-critical")).toContainText("appears to lift this notice");
  });

  test("without a discrepancy nothing changes", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-550`);
    await expect(page.getByTestId("coast-notices-geo").getByTestId(`coast-notice-${ID}`)).toBeVisible();
    await expect(page.getByTestId("coast-notices-disputed")).toHaveCount(0);
  });
});
