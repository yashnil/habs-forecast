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
  test("the notices panel sets the reported-lifted notice apart and does not count it", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-550`);
    const button = page.getByTestId("notices-button");
    const count = Number(await button.locator(".nx-data").innerText());
    await button.click();
    const panel = page.getByTestId("notices-panel");
    const disputed = panel.getByTestId("notices-disputed");
    await expect(disputed).toContainText("Reported lifted by the agency · awaiting review · 1");
    const card = disputed.getByTestId(`notice-${ID}`);
    await expect(card).toHaveAttribute("data-disputed", "true");
    await expect(card.getByRole("link", { name: /SN26-021/ })).toHaveAttribute("href", "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx");
    // once, in the disputed group only; the count and the heading leave it out
    await expect(panel.getByTestId(`notice-${ID}`)).toHaveCount(1);
    await expect(panel.getByTestId("notices-heading")).toHaveText(`${count} active notices in California`);
    await expect(panel.getByTestId("notices-unverified")).toContainText("Not verified");
  });

  test("the briefing says the agency reports it lifted, links the release, and lists it as unknown", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-550`);
    const claim = page.getByTestId(`claim-disputed-${ID}`);
    await expect(claim).toContainText("appears to have lifted");
    await expect(claim.getByRole("link", { name: /SN26-021/ })).toHaveAttribute("href", /SN26-021\.aspx$/);
    await expect(page.getByTestId("unknowns")).toContainText("release SN26-021 appears to lift");
    // in the official list it carries the release, never a plain "in effect"
    await expect(page.getByTestId(`notice-official:${ID}`)).toContainText("SN26-021");
  });

  test("Explore's harbour panel flags it", async ({ page }) => {
    await open(page, `${FAILED}/?sel=harbor:550&layer=forecast`);
    await expect(page.getByTestId("harbor-claims").getByTestId(`claim-disputed-${ID}`)).toContainText("appears to have lifted");
  });

  test("without a discrepancy nothing changes", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-550`);
    await expect(page.getByTestId(`claim-disputed-${ID}`)).toHaveCount(0);
    await page.getByTestId("notices-button").click();
    await expect(page.getByTestId("notices-disputed")).toHaveCount(0);
  });
});
