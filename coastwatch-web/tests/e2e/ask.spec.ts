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
    // the answer first, in one sentence, with who says it and when beside it; then what is next
    await expect(a.getByTestId("ask-lead")).toContainText("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
    await expect(a.getByTestId("ask-lead").getByTestId("ask-by")).toHaveText("CDFW, checked Oct 12");
    await expect(a).toContainText("Next CDFW risk assessments (subject to change): December 15-18 and January 13-15.");
    await expect(a).toContainText("not yet verified by a person");
    // nothing is said twice, and dates are in the reader's terms
    expect((await a.getByTestId("ask-lines").innerText()).match(/humpback/g)).toHaveLength(1);
    await expect(a).not.toContainText(/\d{4}-\d{2}-\d{2}/);
    await expect(a.getByTestId("ask-data-age")).toHaveText("Data age: CDFW, Oct 12 (today).");
    // the standing notice is short and comes AFTER the answer and its sources
    const notice = a.getByTestId("ask-notice");
    await expect(notice).toHaveText("Neritic is not an official source: CDFW and CDPH decide, and their notices take precedence. Never use Neritic to decide whether seafood is safe or whether it is safe to go out.");
    expect(await a.evaluate((el) => [...el.querySelectorAll("[data-testid]")].map((n) => n.getAttribute("data-testid")).filter((t) => ["ask-lines", "ask-citations", "ask-notice", "ask-made"].includes(t!)))).toEqual(["ask-lines", "ask-citations", "ask-notice", "ask-made"]);
    await expect(a.getByTestId("ask-made")).toContainText("without an AI model");

    const chip = a.getByTestId("ask-chip").first();
    await expect(chip).toContainText("CDFW");
    await expect(chip).toContainText("Oct 12");
    await expect(chip).toContainText("not verified");
    await chip.click();
    await expect(chip).toHaveAttribute("aria-expanded", "true");
    const detail = a.getByTestId("ask-chip-detail").first();
    await expect(detail.getByRole("link", { name: /Open CDFW/ })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
    await expect(detail.getByRole("link", { name: "See it on Today" })).toHaveAttribute("href", "/today?place=port-592#fact-status%3Adungeness_commercial");

    await expect(page.getByTestId("ask-sms-text")).toContainText("[Src: CDFW 10/12, unverified]");
    expect(await serious(page)).toEqual([]);

    // a yes/no question about one fishery gets its answer as the first word (the notice is not repeated on a second answer)
    const b = await ask(page, "Is the Monterey crab season open?");
    await expect(b.getByTestId("ask-lead")).toContainText("No — Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
    await expect(b.getByTestId("ask-lines").locator("p")).toHaveCount(4);
    await expect(b.getByTestId("ask-notice")).toHaveCount(0);
    await expect(page.getByTestId("ask-sms-text")).toContainText("No - Dungeness crab: commercial season delayed to at least Jan 1, 2027");
  });

  test("safety questions get the fixed template and official referrals, never 'safe'", async ({ page }) => {
    await page.goto(`${OK}/ask?place=port-550`);
    const a = await ask(page, "Is it safe to eat mussels?");
    await expect(a).toHaveAttribute("data-outcome", "safety");
    await expect(a.getByTestId("ask-preface")).toContainText("doesn't decide whether seafood can be eaten");
    await expect(a.getByTestId("ask-referrals").getByRole("link").first()).toHaveAttribute("href", /cdph\.ca\.gov/);
    await expect(a.getByTestId("ask-lines")).toContainText("For information only");
    await expect(a).not.toContainText(/\b(is|are) safe to eat\b/);
    // the refusal comes first and is never softened into an answer: no line is set as "the answer", none opens with yes or no
    await expect(a.getByTestId("ask-lead")).toHaveCount(0);
    for (const line of await a.getByTestId("ask-lines").locator("p").allInnerTexts()) expect(line).not.toMatch(/^\s*(yes|no)\b/i);
    expect(await a.evaluate((el) => [...el.querySelectorAll("[data-testid]")].map((n) => n.getAttribute("data-testid")).filter((t) => ["ask-preface", "ask-referrals", "ask-lines", "ask-notice"].includes(t!)))).toEqual(["ask-preface", "ask-referrals", "ask-lines", "ask-notice"]);
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
