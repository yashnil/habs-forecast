import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";

/**
 * Ask Neritic (/ask) on the port facts Today shows (src/pilot/today/portToday.ts; here the test
 * fixture, labelled "Sample data"), readiness "demo" (no ANTHROPIC_API_KEY, no TWILIO_*):
 * deterministic grounded answers, citations as chips, data age, safety templates, the
 * SMS preview, and axe at phone and desktop widths. Keep the number of questions low: /api/ask is
 * rate limited per client (30 / 10 min) and every test here shares one server.
 */

async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.filter((v) => v.impact === "serious" || v.impact === "critical").map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

async function ask(page: Page, question: string) {
  const before = await page.getByTestId("ask-answer").count();
  await page.getByTestId("ask-input").fill(question);
  await page.getByTestId("ask-send").click();
  await expect(page.getByTestId("ask-answer")).toHaveCount(before + 1);
  return page.getByTestId("ask-answer").last();
}

test.describe("desktop", () => {
  test("answers a common question with citations, data age and an SMS preview", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-592`);
    await expect(page.getByRole("heading", { level: 1, name: "Ask Neritic" })).toBeVisible();
    await expect(page.getByTestId("ask-readiness")).toContainText("Pilot demo — answers are assembled from Neritic's data without an AI model");
    await expect(page.getByTestId("ask-sample")).toHaveText("Sample data");
    await expect(page.getByTestId("ask-disclaimer")).toContainText("not an official source");
    await expect(page.getByTestId("ask-sms-label")).toHaveText("SMS in carrier review — preview only");

    await page.getByTestId("ask-suggestions").getByRole("button", { name: "What's the status of Dungeness crab?" }).click();
    const a = page.getByTestId("ask-answer").last();
    await expect(a).toContainText("Commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements)");
    await expect(a).toContainText("Next review Dec 15");
    await expect(a).toContainText("not yet verified by a person");
    await expect(a.getByTestId("ask-notice")).toContainText("not an official source");
    await expect(a.getByTestId("ask-data-age")).toContainText("Data age:");
    await expect(a.getByTestId("ask-made")).toContainText("without an AI model");

    const chip = a.getByTestId("ask-chip").first();
    await expect(chip).toContainText("CDFW");
    await expect(chip).toContainText("not verified");
    await chip.click();
    await expect(chip).toHaveAttribute("aria-expanded", "true");
    const detail = a.getByTestId("ask-chip-detail").first();
    await expect(detail.getByRole("link", { name: /Open CDFW/ })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
    await expect(detail.getByRole("link", { name: "See it on Today" })).toHaveAttribute("href", "/today?place=port-592#fact-status%3Adungeness_commercial");

    await expect(page.getByTestId("ask-sms-text")).toContainText("[Src: CDFW 10/12, unverified]");
    expect(await serious(page)).toEqual([]);
  });

  test("safety questions get the fixed template and official referrals, never 'safe'", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-550`);
    const a = await ask(page, "Is it safe to eat mussels?");
    await expect(a).toHaveAttribute("data-outcome", "safety");
    await expect(a.getByTestId("ask-preface")).toContainText("doesn't decide whether seafood can be eaten");
    await expect(a.getByTestId("ask-referrals").getByRole("link").first()).toHaveAttribute("href", /cdph\.ca\.gov/);
    await expect(a.getByTestId("ask-lines")).toContainText("For information only");
    await expect(a).not.toContainText(/\b(is|are) safe to eat\b/);
  });

  test("switching port answers from that port's records; out-of-area is declined", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-592`);
    await page.getByTestId("ask-port").selectOption("port-452");
    const a = await ask(page, "Sea, wind and tide tomorrow?");
    await expect(a).toContainText("PZZ545");
    await expect(a).toContainText("not a go/no-go");
    const b = await ask(page, "What's the crab status in Crescent City?");
    await expect(b).toHaveAttribute("data-outcome", "out_of_area");
    await expect(b).toContainText("only Moss Landing");
  });

  test("what changed compares with this device's previous visit", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-593`);
    const first = await ask(page, "What changed since I last checked?");
    await expect(first).toContainText("no earlier visit");
    await page.reload();
    const second = await ask(page, "What changed since I last checked?");
    await expect(second).toContainText("has changed since your last check");
  });
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test("empty state and an answer pass axe, with no sideways scroll", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-592`);
    await expect(page.getByTestId("ask-chat")).toBeVisible();
    expect(await serious(page)).toEqual([]);
    const a = await ask(page, "What's the latest crab toxin test?");
    await expect(a).toContainText("5.5 ppm");
    await expect(a).toContainText("This is a past lab sample, not a forecast.");
    // an out-of-date record is flagged in the answer (the fixture's C-HARM run ended two days before its clock)
    const b = await ask(page, "What's the bloom outlook this week?");
    await expect(b.getByTestId("ask-stale").filter({ hasText: "NOAA C-HARM" })).toContainText("out of date");
    expect(await serious(page)).toEqual([]);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });
});
