import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { OK } from "./ports";
import { SMS_CONSENT_TEXT } from "../../src/pilot/account/consent";

/**
 * The pilot's account features with NO cloud credentials (the e2e servers have no Supabase env).
 *
 * v13: the device-only account page these tests used to walk (age confirmation, a phone number for texts that
 * are never sent, pilot-port-only alerts) is gone. Without accounts there is nothing to sign in to: the pilot's
 * page is not served, the server fan-out says it is not configured, and what a reader keeps on the device is
 * `/alerts` and `/account` in the v13 shell (tests/e2e/v13/alerts.spec.ts, personal.spec.ts). The legal pages
 * are drafts. axe: no violations at 390 and 1440.
 */
async function axe(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  return r.violations.map((v) => `${v.impact} ${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

test.describe("without accounts connected", () => {
  test("the pilot's account page is not served, and no page of the v13 shell links to it", async ({ page }) => {
    const r = await page.goto(`${OK}/account/pilot`);
    expect(r!.status()).toBe(404);
    await page.goto(`${OK}/account`);
    await expect(page.getByTestId("nx-account")).toBeVisible();
    await expect(page.locator('a[href^="/account/pilot"]')).toHaveCount(0);
    await expect(page.getByTestId("account-pilot-link")).toHaveCount(0);
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
    await expect(page.getByTestId("legal-effective")).toContainText("Effective November 14, 2026");
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
    for (const path of ["/privacy", "/terms", "/sms-terms"]) {
      test(`${path}: no violations`, async ({ page }) => {
        await page.goto(`${OK}${path}`);
        await expect(page.getByTestId("legal-draft-banner")).toBeVisible();
        expect(await axe(page)).toEqual([]);
      });
    }
  });
}
