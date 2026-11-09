import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { PILOT } from "./ports";

/**
 * The pilot as ONE product, end to end, on the server started with NERITIC_PILOT=1 (fixture data,
 * no cloud credentials): the front door opens Today at Moss Landing → a status row's detail → the
 * toxin test → a question to Ask Neritic in the sheet → its citation → the Today card it names →
 * "Set an alert" → the account page with the port and its honest readiness labels → the legal
 * pages → Today saved, then opened with no connection. One shell throughout; axe finds nothing
 * (every impact level) at 390 and 1440 px.
 */
async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

/** The same top bar on every pilot page: Today first, the bell, the ⋯ menu; the dock on a phone. */
async function oneShell(page: Page, phone: boolean) {
  await expect(page.getByTestId("alerts-bell")).toBeVisible();
  await expect(page.getByTestId("more-menu")).toBeVisible();
  const nav = page.getByTestId(phone ? "nav-dock" : "nav-bar");
  await expect(nav.locator("a")).toHaveText(["Today", "Explore", "My Coast", "Research"]);
  await expect(page.locator(".env-instrument.today").first()).toBeVisible();
}

for (const [label, viewport] of [["phone 390", { width: 390, height: 844 }], ["desktop 1440", { width: 1440, height: 900 }]] as const) {
  test(`the pilot journey, ${label}`, async ({ browser }) => {
    test.setTimeout(180_000);
    const phone = viewport.width < 600;
    const ctx = await browser.newContext({ viewport, ...(phone ? { isMobile: true, hasTouch: true } : {}) });
    const page = await ctx.newPage();
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));

    // ---- the front door: a bare address opens Today; the reader chooses Moss Landing once
    await page.goto(`${PILOT}/`);
    await expect(page).toHaveURL(`${PILOT}/today`);
    await page.getByTestId("pick-port-port-592").click();
    await expect(page.locator("h1")).toHaveText("Moss Landing");
    await expect(page.getByTestId("sea-lanes")).toBeVisible({ timeout: 15_000 });
    await oneShell(page, phone);
    // official status first, in words
    await expect(page.getByTestId("today-lede")).toContainText("Dungeness crab: commercial season delayed to at least Jan 1, 2027");
    expect(await axe(page)).toEqual([]);

    // ---- a status row's detail: the record, its source, and that no person has verified it
    await page.getByTestId("status-dungeness_commercial").click();
    const sheet = page.getByTestId("status-sheet");
    await expect(sheet.getByRole("heading", { level: 2 })).toHaveText("Dungeness crab (commercial)");
    await expect(sheet).toContainText("Not verified: no person has checked this record against the source yet");
    await expect(sheet).toContainText("Next review");
    await expect(sheet.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", /wildlife\.ca\.gov/);
    await sheet.getByTestId("status-sheet-close").click();
    await expect(sheet).toBeHidden();

    // ---- the toxin test: measured, dated, against the limit
    const tests = page.getByTestId("today-tests");
    await tests.scrollIntoViewIfNeeded();
    await expect(page.getByTestId("tests-lede")).toHaveText("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
    await expect(tests.getByTestId("toxin-chart").first()).toHaveAttribute("role", "img");
    await expect(tests).toContainText("Dungeness crab (commercial) is still delayed (CDFW); a test result does not change that.");

    // ---- Ask Neritic, in place: the same facts, each line with its source
    await page.getByTestId("action-ask").click();
    const ask = page.getByTestId("ask-sheet");
    await expect(ask).toBeVisible();
    await expect(ask.getByTestId("ask-readiness")).toContainText("Pilot demo — answers are assembled from Neritic's data without an AI model");
    await expect(ask.getByTestId("ask-sample")).toHaveText("Sample data");
    await ask.getByTestId("ask-input").fill("What's the latest crab toxin test?");
    await ask.getByTestId("ask-send").click();
    const answer = ask.getByTestId("ask-answer");
    await expect(answer).toContainText("Monterey Bay Dungeness tested well under the limit on Sep 28: avg 5.5 ppm, limit 30.");
    // the hierarchy holds in the answer too: the official delay comes before the test, and is marked unverified
    expect(await answer.getByTestId("ask-lines").locator("p").first().innerText()).toContain("Dungeness crab: commercial season delayed to at least Jan 1, 2027 (humpback whale entanglements).");
    await expect(answer).toContainText("not yet verified by a person");
    // the answer's own lines never call anything safe (the standing notice under them says Neritic never decides that)
    await expect(answer.getByTestId("ask-lines")).not.toContainText(/\bsafe\b/i);
    await expect(answer.getByTestId("ask-notice")).toContainText("Never use Neritic to decide whether seafood is safe");
    // dates as Today words them, in the answer and on its source chips
    await expect(answer).not.toContainText(/\d{4}-\d{2}-\d{2}/);
    expect(await axe(page)).toEqual([]);

    // ---- its citation → the card on Today that it names
    const chip = answer.getByTestId("ask-chip").filter({ hasText: "CDPH" });
    await chip.click();
    const see = answer.getByTestId("ask-chip-detail").getByTestId("ask-see-today");
    await expect(see).toHaveAttribute("href", "/today?place=port-592#fact-test%3Abma-g-monterey-bay-dungeness-crab");
    await see.click();
    await expect(ask).toBeHidden();
    const card = page.locator('[data-fact-hit="test:bma-g-monterey-bay-dungeness-crab"]');
    await expect(card).toBeInViewport();
    await expect(card).toHaveAttribute("data-testid", "test-bma-g-monterey-bay-dungeness-crab");
    await expect(card).toHaveClass(/fact-hit/);
    expect(new URL(page.url()).pathname + new URL(page.url()).search).toBe("/today?place=port-592");

    // ---- set an alert: the account page opens with this port first, one tap from its alerts
    await expect(page.getByTestId("readiness-alerts")).toHaveText("Not yet connected: accounts, push and text. Alerts work in the app on this device.");
    await page.getByTestId("action-alert").click();
    await expect(page).toHaveURL(`${PILOT}/account?place=port-592`);
    await expect(page.locator("h1")).toHaveText("Your ports and alerts");
    await oneShell(page, phone);
    const want = page.getByTestId("want-port");
    await expect(want).toContainText("Alerts for Moss Landing.");
    await page.getByTestId("want-save").click();
    await expect(page.getByTestId("port-port-592")).toHaveAttribute("data-saved", "1");
    await expect(want).toContainText("Moss Landing is saved on this device.");
    const alerts = page.getByTestId("alerts-port-592");
    await expect(alerts).toBeInViewport();
    await expect(alerts.getByTestId("trigger-port-592-status_change")).toBeChecked();
    await expect(alerts.getByTestId("trigger-port-592-new_toxin_result")).toBeChecked();
    // readiness, exactly as configured: nothing is claimed
    for (const id of ["readiness-accounts", "readiness-push", "readiness-sms"]) await expect(page.getByTestId(id)).toHaveText("Not yet connected");
    await expect(page.getByTestId("readiness-alerts")).toHaveText("Pilot demo");
    await expect(page.getByTestId("device-only-banner")).toContainText("Saved on this device — accounts not yet connected");
    await expect(alerts).toContainText("Text message(not yet available: carrier registration pending)");
    await expect(page.getByTestId("sample-facts-note")).toContainText("Sample data");
    await expect(page.getByTestId("account-page")).not.toContainText(/\bLive\b/);
    expect(await axe(page)).toEqual([]);

    // ---- the legal pages, from the footer every pilot page shares
    for (const [link, testid, title] of [
      ["footer-privacy", "privacy-page", "Privacy Policy"],
      ["footer-terms", "terms-page", "Terms of Use"],
      ["footer-sms-terms", "sms-terms-page", "SMS Terms"],
    ] as const) {
      await page.getByTestId(link).click();
      await expect(page.getByTestId(testid)).toBeVisible();
      await expect(page.locator("h1")).toHaveText(title);
      await expect(page.getByTestId("legal-draft-banner")).toContainText("Draft — pending owner/legal review");
      await oneShell(page, phone);
    }
    await expect(page.getByTestId("sms-status")).toContainText("not being sent yet");
    expect(await axe(page)).toEqual([]);

    // ---- Ask's own page says the same about SMS, in the same shell
    await page.getByTestId("footer-ask").click();
    await expect(page.locator("h1")).toHaveText("Ask Neritic");
    await expect(page.getByTestId("ask-port")).toHaveValue("port-592");
    await expect(page.getByTestId("ask-sms-label")).toHaveText("SMS in carrier review — preview only");
    await oneShell(page, phone);
    expect(await axe(page)).toEqual([]);

    // ---- back to Today (the reader's own port), save it, lose the connection, open it again
    await page.getByTestId("ask-today-link").click();
    await expect(page.locator("h1")).toHaveText("Moss Landing");
    await expect(page.getByTestId("sea-lanes")).toBeVisible({ timeout: 15_000 });
    // the port saved for alerts is now on the bell's panel (no alert yet: alerts begin from what was already seen)
    await page.getByTestId("save-offline").click();
    await expect(page.getByTestId("save-status")).toContainText("Saved", { timeout: 30_000 });
    await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
    await ctx.setOffline(true);
    // the front door itself, with no connection: the saved Today is offered by the port's name
    await page.goto(`${PILOT}/`);
    await expect(page.getByTestId("offline-page")).toBeVisible();
    await page.getByTestId("offline-today").click();
    await expect(page.getByTestId("today-port")).toHaveText("Moss Landing");
    await expect(page.getByTestId("today-saved-copy")).toContainText("Saved copy");
    await expect(page.getByTestId("status-dungeness_commercial")).toContainText("Commercial season delayed to at least Jan 1, 2027");
    await expect(page.getByTestId("sea-lanes")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("tests-lede")).toContainText("avg 5.5 ppm");
    await ctx.setOffline(false);

    expect(errors).toEqual([]);
    await ctx.close();
  });
}
