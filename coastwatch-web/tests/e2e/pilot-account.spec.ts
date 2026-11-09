import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";
import { SMS_CONSENT_TEXT } from "../../src/pilot/account/consent";

/**
 * Pilot accounts and alerts with NO cloud credentials (the e2e servers have no Supabase env): every
 * choice is kept on the device and labelled so, alerts come from the device's own comparison of the
 * port's real facts (the same ones Today shows; here from the test fixture, and labelled so), and the
 * legal pages are drafts. axe: no violations at 390 and 1440.
 */
async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.impact} ${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

/** a device that saved Moss Landing yesterday and saw an older Dungeness status then */
async function seedReturningDevice(page: Page) {
  const account = {
    v: 1,
    role: "commercial_fisher",
    fisheries: ["dungeness_commercial"],
    home_place_id: "port-592",
    vessel_length_class: null,
    age_13_plus: { at: "2026-11-07T15:00:00.000Z", text: "I am 13 or older." },
    saved_places: ["port-592"],
    alerts: [{ place_id: "port-592", triggers: ["status_change", "new_toxin_result", "ramp_action"], channels: ["in_app"] }],
    sms: null,
    created_at: "2026-11-07T15:00:00.000Z",
  };
  await page.addInitScript(
    ([a]) => {
      if (sessionStorage.getItem("seeded")) return;
      sessionStorage.setItem("seeded", "1");
      localStorage.setItem("nrt-pilot-account-v1", a);
    },
    [JSON.stringify(account)],
  );
  // the first visit keeps a baseline of the port's facts as the server has them (no alert yet)…
  await page.goto(`${OK}/account`);
  await page.waitForFunction(() => !!JSON.parse(localStorage.getItem("nrt-pilot-alert-snap-v1") ?? "{}")["port-592"]);
  await expect(page.getByTestId("inbox-empty")).toBeVisible();
  // …then that visit becomes yesterday's, when the commercial Dungeness delay ran to Dec 15
  await page.evaluate(() => {
    const k = "nrt-pilot-alert-snap-v1";
    const s = JSON.parse(localStorage.getItem(k)!);
    const id = "status:dungeness_commercial";
    s["port-592"].seen_at = "2026-11-07T15:00:00.000Z";
    s["port-592"].facts[id] = { v: "00000000", sig: { ...s["port-592"].facts[id].sig, headline: "Dungeness crab (commercial): Commercial season delayed to at least Dec 15 (humpback whale entanglements)." } };
    localStorage.setItem(k, JSON.stringify(s));
  });
}

test.describe("/account without accounts connected", () => {
  test("says everything is on this device, with honest readiness labels", async ({ page }) => {
    await page.goto(`${OK}/account`);
    await expect(page.getByTestId("account-page")).toHaveAttribute("data-mode", "device");
    await expect(page.getByTestId("device-only-banner")).toContainText("Saved on this device — accounts not yet connected");
    for (const id of ["readiness-accounts", "readiness-push", "readiness-sms"]) await expect(page.getByTestId(id)).toHaveText("Not yet connected");
    await expect(page.getByTestId("push-panel")).toHaveAttribute("data-advice", "not_configured");
    await expect(page.getByTestId("signin-email")).toHaveCount(0);
  });

  test("set up: 13+ is required, the SMS box starts unticked with the exact disclosure, and choices survive a reload", async ({ page }) => {
    await page.goto(`${OK}/account`);
    await expect(page.getByTestId("consent-sms")).not.toBeChecked();
    await expect(page.getByTestId("sms-disclosure")).toHaveText(SMS_CONSENT_TEXT);
    await expect(page.getByTestId("consent-age")).not.toBeChecked();
    await page.getByTestId("consent-age").check();
    await page.getByTestId("consent-sms").check();
    await page.getByTestId("sms-phone").fill("555");
    await page.getByTestId("device-setup-save").click();
    await expect(page.getByTestId("signin-msg")).toContainText("10-digit US mobile number");
    await page.getByTestId("sms-phone").fill("(831) 555-0100");
    await page.getByTestId("device-setup-save").click();
    await expect(page.getByTestId("signin-msg")).toHaveText("Saved on this device.");

    await page.getByTestId("save-port-592").click();
    await page.getByTestId("acct-fisheries").getByLabel("Dungeness crab (commercial)").check();
    await expect(page.getByTestId("alerts-port-592")).toBeVisible();
    await expect(page.getByTestId("trigger-port-592-status_change")).toBeChecked();
    await expect(page.getByTestId("trigger-port-592-bloom_risk_jump")).not.toBeChecked();

    await page.reload();
    await expect(page.getByTestId("device-setup-done")).toContainText("Text alerts requested for +18315550100 (not sent yet)");
    await expect(page.getByTestId("port-port-592")).toHaveAttribute("data-saved", "1");
    await expect(page.getByTestId("acct-fisheries").getByLabel("Dungeness crab (commercial)")).toBeChecked();
    // the e2e servers read the test fixture, and the page says so
    await expect(page.getByTestId("sample-facts-note")).toContainText("test fixture, not today’s published data");
  });

  test("in-app alerts: a change since the last visit lands in the inbox once, with its source; feedback is kept", async ({ page }) => {
    await seedReturningDevice(page);
    await page.goto(`${OK}/account`);
    const items = page.getByTestId("inbox-item");
    await expect(items.first()).toBeVisible();
    const status = page.locator('[data-testid="inbox-item"][data-trigger="status_change"]').filter({ hasText: "Dungeness crab (commercial)" });
    await expect(status).toHaveCount(1);
    await expect(status).toContainText("Moss Landing: Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027");
    await expect(status).toContainText("Was: “Dungeness crab (commercial): Commercial season delayed to at least Dec 15");
    await expect(status).toContainText("not yet verified by a person");
    await expect(status).toContainText("Source: CDFW");
    // made from the test fixture's facts: the item itself says it is not live
    await expect(status.getByTestId("inbox-sample")).toContainText("Sample data (test fixture), not live");
    // the one change is the one alert (the other facts are what the baseline saw)
    await expect(items).toHaveCount(1);
    await expect(page.getByTestId("inbox-list")).not.toContainText(/\bsafe\b/i);
    const n = await items.count();
    await expect(page.getByTestId("inbox-unread")).toHaveText(`${n} unread`);

    await page.reload();
    await expect(page.getByTestId("inbox-item")).toHaveCount(n);

    await status.getByTestId("feedback-yes").click();
    await expect(status.getByTestId("feedback")).toContainText("Thanks — saved on this device.");
    await page.getByTestId("inbox-mark-read").click();
    await expect(page.getByTestId("inbox-unread")).toHaveCount(0);
  });

  test("trip log and deleting everything on this device", async ({ page }) => {
    await seedReturningDevice(page);
    await page.goto(`${OK}/account`);
    const trip = page.getByTestId("trip-log");
    await trip.getByText("Fair", { exact: true }).click();
    await trip.getByTestId("trip-save").click();
    await expect(trip.getByTestId("trip-status")).toHaveText("Trip saved on this device.");
    await expect(trip.getByTestId("trip-list")).toContainText("Moss Landing · Fair");

    await page.getByTestId("forget-device").click();
    await page.getByTestId("confirm-delete-yes").click();
    await expect(page.getByTestId("data-msg")).toHaveText("Deleted from this device.");
    await expect(page.getByTestId("port-port-592")).toHaveAttribute("data-saved", "0");
    await expect(page.getByTestId("inbox-empty")).toBeVisible();
    expect(await page.evaluate(() => Object.keys(localStorage).filter((k) => k.startsWith("nrt-pilot-")))).toEqual([]);
  });

  test("blocked storage: the page still works for this visit and says so", async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, "localStorage", {
        get() {
          throw new DOMException("blocked", "SecurityError");
        },
      });
    });
    await page.goto(`${OK}/account`);
    await expect(page.getByTestId("storage-blocked")).toBeVisible();
    await page.getByTestId("save-port-550").click();
    await expect(page.getByTestId("port-port-550")).toHaveAttribute("data-saved", "1");
  });

  test("the fan-out endpoint refuses without its secret and says it is not configured", async ({ request }) => {
    const r = await request.post(`${OK}/api/alerts/fanout`);
    expect(r.status()).toBe(503);
    expect((await r.json()).readiness).toBe("not_configured");
  });
});

test.describe("legal pages", () => {
  test("privacy: draft banner, effective date and the CalOPPA elements", async ({ page }) => {
    await page.goto(`${OK}/privacy`);
    await expect(page.getByTestId("legal-draft-banner")).toContainText("Draft — pending owner/legal review");
    await expect(page.getByTestId("legal-effective")).toContainText("Effective November 8, 2026");
    const body = page.getByTestId("privacy-page");
    for (const t of ["What we collect", "How we use it", "Who we share it with", "How long we keep it", "Seeing, changing and deleting", "Do Not Track", "Children", "Changes to this policy", "We do not sell personal information", "never shared with or sold to third parties", "[OWNER: add a contact email before launch]"])
      await expect(body).toContainText(t);
    await expect(page.getByTestId("legal-attribution")).toContainText("CC BY 4.0");
  });

  test("terms: as-is, not for navigation, not official, AI may be wrong, chlorophyll is not toxin, California law", async ({ page }) => {
    await page.goto(`${OK}/terms`);
    const body = page.getByTestId("terms-page");
    for (const t of ["Not an official source", "CDFW", "CDPH", "NWS", "USCG", "Not for navigation", "chlorophyll is not toxin", "AI answers may be wrong", "Your decisions are yours", "AS IS", "Limitation of liability", "State of California"])
      await expect(body).toContainText(t, { ignoreCase: true });
  });

  test("SMS terms: program, frequency, rates, STOP/HELP, consent not a condition, no sharing for marketing", async ({ page }) => {
    await page.goto(`${OK}/sms-terms`);
    const body = page.getByTestId("sms-terms-page");
    for (const t of ["Neritic Alerts", "Frequency varies", "Msg & data rates may apply", "STOP", "HELP", "Consent is not a condition", "never shared with or sold to third parties or affiliates for marketing"])
      await expect(body).toContainText(t);
    await expect(page.getByTestId("sms-terms-disclosure")).toHaveText(SMS_CONSENT_TEXT);
    await expect(page.getByTestId("sms-status")).toContainText("not being sent yet");
  });
});

for (const [label, viewport] of [["phone", { width: 390, height: 844 }], ["desktop", { width: 1440, height: 900 }]] as const) {
  test.describe(`axe at ${viewport.width}`, () => {
    test.use({ viewport, ...(label === "phone" ? { isMobile: true, hasTouch: true } : {}) });
    test("/account with an inbox: no violations", async ({ page }) => {
      await seedReturningDevice(page);
      await page.goto(`${OK}/account`);
      await expect(page.getByTestId("inbox-item").first()).toBeVisible();
      expect(await axe(page)).toEqual([]);
    });
    for (const path of ["/privacy", "/terms", "/sms-terms"]) {
      test(`${path}: no violations`, async ({ page }) => {
        await page.goto(`${OK}${path}`);
        await expect(page.getByTestId("legal-draft-banner")).toBeVisible();
        expect(await axe(page)).toEqual([]);
      });
    }
  });
}
