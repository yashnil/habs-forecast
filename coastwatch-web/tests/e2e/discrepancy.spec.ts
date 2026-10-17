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
    // the reason is always visible beside it, never behind "details"
    await expect(card.getByTestId("notice-critical")).toContainText("release SN26-021");
    await expect(card.getByTestId("notice-critical")).toContainText("appears to lift this notice");
    await expect(card.getByTestId("notice-critical")).toContainText("not shown as in effect");
    await expect(panel.getByTestId("notices-current").getByTestId(`notice-${ID}`)).toHaveCount(0);
    // once, in the disputed group only; neither the place's count nor the statewide heading counts it
    await expect(panel.getByTestId(`notice-${ID}`)).toHaveCount(1);
    await expect(panel.getByTestId("notices-local").getByTestId(`notice-${ID}`)).toHaveCount(0);
    const listed = await panel.locator('li[data-testid^="notice-"]:not([data-disputed])').count();
    await expect(panel.getByTestId("notices-heading")).toHaveText(`${listed} active notices in California`);
    expect(count).toBe(await panel.getByTestId("notices-local").locator('li[data-testid^="notice-"]').count());
    await expect(panel.getByTestId("notices-unverified")).toContainText("Not verified");
  });

  test("the briefing says the agency reports it lifted, links the release, and lists it as unknown", async ({ page }) => {
    await open(page, `${FAILED}/coast?place=port-550`);
    const claim = page.getByTestId(`claim-disputed-${ID}`);
    await expect(claim).toContainText("appears to have lifted");
    await expect(claim.getByRole("link", { name: /SN26-021/ })).toHaveAttribute("href", /SN26-021\.aspx$/);
    await expect(page.getByTestId("unknowns")).toContainText("release SN26-021 appears to lift");
    // in the official list it carries the release, never a plain "in effect"
    await expect(page.getByTestId(`notice-official:${ID}`)).toContainText("appears to lift it; awaiting review");
    // and no ordinary official statement counts it as applying
    expect(await page.locator(`[data-testid="matters"] [data-kind="official"][data-supports~="official:${ID}"]:not([data-testid^="claim-disputed-"])`).count()).toBe(0);
  });

  test("Explore's harbour panel flags it", async ({ page }) => {
    await open(page, `${FAILED}/?sel=harbor:550&layer=forecast`);
    await expect(page.getByTestId("harbor-claims").getByTestId(`claim-disputed-${ID}`)).toContainText("appears to have lifted");
  });

  test("without a discrepancy nothing changes", async ({ page }) => {
    await open(page, `${OK}/coast?place=port-550`);
    await expect(page.getByTestId(`claim-disputed-${ID}`)).toHaveCount(0);
    // the notice is listed for Monterey as an ordinary, unverified notice
    const row = page.getByTestId(`notice-official:${ID}`);
    await expect(row).toContainText("Not verified");
    await expect(row).not.toContainText("appears to lift");
    await page.getByTestId("notices-button").click();
    await expect(page.getByTestId("notices-disputed")).toHaveCount(0);
    // listed once, as a notice covering Monterey's waters (the briefing's place)
    await expect(page.getByTestId("notices-local").getByTestId(`notice-${ID}`)).toHaveCount(1);
    await expect(page.getByTestId("notices-panel").locator(`li[data-testid="notice-${ID}"]`)).toHaveCount(1);
  });
});
