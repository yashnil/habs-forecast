import { expect, test } from "@playwright/test";
import { OK } from "../ports";
import { BODEGA, MOSS, RETURNING, axe, dungenessOpens, halfAnHourLater, noSidewaysScroll, portRecord, secondState, seed, stored } from "./helpers";

/**
 * Alerts in the v13 shell (src/nx/alerts; PLAN decision 3): follow any port, what is watched, the inbox, and
 * exactly what an alert is: a comparison this browser makes when Neritic is opened. Nothing is checked while
 * it is closed and nothing is sent anywhere, and the page says so in those words.
 *
 * The controlled change is two states of the published records for one port: the fixture as it is, then the
 * same answer with one status record changed and a later publish time (helpers.ts `secondState`), which is how
 * a change reaches a browser.
 */
const DELIVERY = "Checked on this device when you open Neritic. Nothing is checked while it's closed. No texts, emails or phone notifications.";

test.beforeEach(async ({ context }) => {
  await seed(context, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS }) });
});

test("arriving from a port outside the pilot: it is offered, followed in one tap, and what alerts you can be changed", async ({ page }) => {
  await page.goto(`${OK}/alerts?place=${BODEGA}`);
  await expect(page.getByTestId("nx-app")).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("What changed at the ports you follow");
  const offer = page.getByTestId("alerts-offer");
  await expect(offer).toHaveAttribute("data-port", BODEGA);
  await expect(offer.getByRole("heading")).toHaveText("Follow Bodega Bay");
  await expect(page.getByTestId("alerts-empty")).toContainText("Nothing is being followed");

  await page.getByTestId("alerts-follow").click();
  const row = page.getByTestId(`alerts-port-${BODEGA}`);
  await expect(row).toContainText("Bodega Bay");
  await expect(row).toContainText("3 of 5 kinds of change alert you");
  await expect(offer).toHaveCount(0);
  // the first look is a baseline, said in words: no alert is made from what is already there
  await expect(page.getByTestId("alerts-said")).toContainText("Bodega Bay: noted as it is now. Changes from here on will be listed.", { timeout: 15_000 });
  await expect(page.getByTestId("alerts-empty")).toContainText("No alerts yet");
  await expect(page.getByTestId("alerts-last-checked")).toContainText(/^Last checked .+ PT\.$/);

  // the port the reader came for opens with its choices showing; each is a switch that says its state
  const stale = page.getByTestId(`alerts-trigger-${BODEGA}-data_stale`);
  await expect(stale).toHaveAttribute("aria-checked", "false");
  await stale.click();
  await expect(stale).toHaveAttribute("aria-checked", "true");
  await page.getByTestId(`alerts-trigger-${BODEGA}-status_change`).click();
  expect((await stored(page, "nx-follow-v1")).ports).toMatchObject([{ port: BODEGA, triggers: ["new_toxin_result", "ramp_action", "data_stale"] }]);

  // it survives a reload
  await page.reload();
  await expect(page.getByTestId(`alerts-port-${BODEGA}`)).toContainText("3 of 5 kinds of change alert you");
  await expect(page.getByTestId("alerts-offer")).toHaveCount(0);

  // another port from the list ("What alerts you", opened from the dock), then stop following the first: its baseline goes with it
  await expect(page.getByTestId("alerts-kinds")).toHaveAttribute("aria-expanded", "false");
  await page.getByTestId("alerts-kinds").click();
  await page.getByTestId("alerts-add-port").selectOption(MOSS);
  await page.getByTestId("alerts-add").click();
  await expect(page.getByTestId(`alerts-port-${MOSS}`)).toBeVisible();
  await expect.poll(async () => Object.keys((await stored(page, "nx-alert-snap-v1")) ?? {}).sort()).toEqual([BODEGA, MOSS].sort());
  // (the port the address names is already open; any other opens from "What alerts you")
  await expect(page.getByTestId(`alerts-port-${BODEGA}`).getByRole("button", { name: "What alerts you" })).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId(`alerts-port-${MOSS}`).getByRole("button", { name: "What alerts you" })).toHaveAttribute("aria-expanded", "false");
  await page.getByTestId(`alerts-unfollow-${BODEGA}`).click();
  await expect(page.getByTestId(`alerts-port-${BODEGA}`)).toHaveCount(0);
  expect(Object.keys(await stored(page, "nx-alert-snap-v1"))).toEqual([MOSS]);
  // and Bodega Bay is offered again, since the address still names it
  await expect(page.getByTestId("alerts-offer")).toHaveAttribute("data-port", BODEGA);
});

test("a controlled change: two states of the records make exactly one alert, counted on the bar when Explore is opened, never twice", async ({ page }) => {
  const records = await secondState(page, BODEGA, dungenessOpens);
  await page.goto(`${OK}/alerts?place=${BODEGA}`);
  await page.getByTestId("alerts-follow").click();
  await expect(page.getByTestId("alerts-said")).toContainText("noted as it is now", { timeout: 15_000 });
  await expect(page.getByTestId("nx-unread")).toHaveCount(0);

  // the records change while Neritic is closed; half an hour later the reader opens Explore, not Alerts
  records.publish();
  await halfAnHourLater(page);
  await page.goto(`${OK}/explore?place=${BODEGA}`);
  await expect(page.getByTestId("nx-unread")).toHaveText("1", { timeout: 15_000 });
  await expect(page.getByTestId("nx-alerts-link")).toHaveAttribute("aria-label", "Alerts, 1 unread");
  expect(await stored(page, "nx-inbox-v1")).toHaveLength(1);

  // a reload, and another due check against the same records: still one
  await halfAnHourLater(page);
  const checked = (await stored(page, "nx-alert-check-v1")).at;
  await page.reload();
  await expect.poll(async () => (await stored(page, "nx-alert-check-v1")).at).not.toBe(checked);
  await expect(page.getByTestId("nx-unread")).toHaveText("1");
  expect(await stored(page, "nx-inbox-v1")).toHaveLength(1);

  // the alert: what changed, its source, its date, and that the record is not verified
  await page.getByTestId("nx-alerts-link").click();
  const items = page.getByTestId("alerts-item");
  await expect(items).toHaveCount(1);
  const item = items.first();
  await expect(item).toHaveAttribute("data-port", BODEGA);
  await expect(item).toContainText("Unread · Bodega Bay · Season status");
  await expect(item.getByRole("heading")).toHaveText("Commercial Dungeness status changed");
  await expect(item.getByRole("img", { name: /was Jan 1, 2027, now Jan 1, 2027 \(sample data\)\. Same date\./ })).toBeVisible();
  await expect(item.getByRole("link", { name: "CDFW" })).toHaveAttribute("href", /^https:\/\/wildlife\.ca\.gov\//);
  await expect(item).toContainText("Not verified");
  await expect(item).toContainText("Sample data");
  await expect(item).not.toContainText(/\d{4}-\d{2}-\d{2}/);
  await expect(page.getByTestId("alerts-unread")).toHaveText(" · 1 unread");

  // "Check now" against the same records adds nothing and says so
  await page.getByTestId("alerts-check-now").click();
  await expect(page.getByTestId("alerts-said")).toHaveText("Checked: nothing has changed.");
  await expect(items).toHaveCount(1);

  // read ("What alerts you" holds the list's actions): the count leaves the page and the bar
  await page.getByTestId("alerts-kinds").click();
  await page.getByTestId("alerts-mark-read").click();
  await expect(page.getByTestId("alerts-unread")).toHaveCount(0);
  await expect(page.getByTestId("nx-unread")).toHaveCount(0);
  await expect(item).toHaveAttribute("data-read", "");

  // the record, one tap away: the kind of change, the record's sentence and what it said before
  await page.getByTestId("alerts-open").click();
  await expect(item).toContainText("Bodega Bay · Season status and official notices");
  await expect(item).toContainText("Dungeness crab (commercial): Commercial season opens Jan 1, 2027 with a 30-fathom depth limit.");
  await expect(item).toContainText("Before: Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027");
});

test("the ticket says the change: the new date with the old struck, before → after on a date axis, and opening the record clears the count", async ({ page }) => {
  // a followed port with one alert on this device (as check.ts writes it), a date that moved two weeks
  const FORT = "port-223";
  await page.goto(`${OK}/about`);
  await page.evaluate((kv) => Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, v)), {
    "nx-follow-v1": JSON.stringify({ v: 1, ports: [{ port: FORT, triggers: ["status_change", "new_toxin_result", "ramp_action"], since: "2026-10-01T17:00:00Z" }] }),
    "nx-inbox-v1": JSON.stringify([
      {
        key: `${FORT}|status|v2`, place_id: FORT, place_label: "Fort Bragg", fact_id: "status", fact_version: "v2", trigger: "status_change", severity: "red",
        title: "Fort Bragg: Dungeness crab (commercial): Commercial season delayed to at least Jan 15, 2027",
        body: "Changed since 2026-10-01. Was: “Commercial season delayed to at least Jan 1, 2027”. Source: CDFW, 2026-10-12.",
        basis: "official", verified: false, source: { name: "CDFW", url: "https://wildlife.ca.gov/" }, as_of: "2026-10-12T18:00:00Z", created_at: "2026-10-13T05:00:00Z",
        read_at: null, sample: true, change: "changed", was: "Dungeness crab (commercial): Commercial season delayed to at least Jan 1, 2027",
      },
    ]),
    // checked a moment ago, so opening the page does not check (and re-baseline) again
    "nx-alert-check-v1": JSON.stringify({ at: new Date().toISOString(), ports: [FORT], failed: [], data_at: null }),
  });
  await page.goto(`${OK}/alerts?place=${FORT}`);
  const ticket = page.locator("[data-nx-ticket]");
  const item = ticket.getByTestId("alerts-item");
  await expect(item).toHaveAttribute("data-port", FORT);
  await expect(item).toContainText("Unread · Fort Bragg · Season status");
  await expect(item.getByRole("heading")).toHaveText("Commercial Dungeness delay moved");
  await expect(item.getByRole("img", { name: /was Jan 1, 2027, now Jan 15, 2027 \(sample data\)\. \+14 days later\./ })).toBeVisible();
  await expect(item).toContainText("Not verified");
  await expect(item).toContainText("Noticed on this device");
  await expect(item).toContainText("following since Oct 1");
  await expect(page.getByTestId("alerts-delivery")).toHaveText(DELIVERY);
  await expect(page.getByTestId("nx-unread")).toHaveText("1");

  // the one filled action opens the record and marks the alert read: the count leaves the bar
  await page.getByTestId("alerts-open").click();
  await expect(page.getByTestId("alerts-open")).toHaveText(/Hide the record/);
  await expect(item.getByRole("link", { name: /The record at CDFW/ })).toHaveAttribute("href", "https://wildlife.ca.gov/");
  await expect(item).toHaveAttribute("data-read", "");
  await expect(item).toContainText("Read · Fort Bragg");
  await expect(item).toContainText("Dungeness crab (commercial): Commercial season delayed to at least Jan 15, 2027");
  await expect(page.getByTestId("nx-unread")).toHaveCount(0);
  expect((await stored(page, "nx-inbox-v1"))[0].read_at).toBeTruthy();

  // "What alerts you" (the dock) shows the port and its switches; Following there stops following, and its alert goes with it
  const kinds = page.getByTestId("alerts-kinds");
  await expect(kinds).toHaveAttribute("aria-expanded", "false");
  await kinds.click();
  await expect(page.getByTestId(`alerts-port-${FORT}`)).toContainText("3 of 5 kinds of change alert you");
  await page.getByTestId(`alerts-unfollow-${FORT}`).click();
  await expect(page.getByTestId("alerts-item")).toHaveCount(0);
  expect((await stored(page, "nx-follow-v1")).ports).toEqual([]);
  await expect(page.getByTestId("alerts-offer")).toHaveAttribute("data-port", FORT);
});

test("it says exactly what happens: when it checks, that nothing is sent, and no way to give a number or an address", async ({ page }) => {
  const outside: string[] = [];
  page.on("request", (r) => {
    const u = new URL(r.url());
    if (u.origin !== OK || /supabase|twilio|anthropic|push/i.test(u.href)) outside.push(u.href);
  });
  await page.goto(`${OK}/alerts?place=${MOSS}`);
  await expect(page.getByTestId("alerts-delivery")).toHaveText(DELIVERY);
  await page.getByTestId("alerts-follow").click();
  await expect(page.getByTestId("alerts-last-checked")).toContainText("Last checked", { timeout: 15_000 });
  await expect(page.getByTestId("nx-alerts")).toContainText("Neritic has no server watching your ports");

  const channels = page.getByTestId("alerts-channels");
  await expect(channels.locator('[data-channel="in_app"]')).toContainText("On");
  for (const c of ["push", "sms", "email"]) await expect(channels.locator(`[data-channel="${c}"]`)).toContainText("Not available in this preview");
  await expect(page.getByTestId("alerts-local")).toContainText("Everything here stays in this browser.");

  // nothing to type a phone number, an email address or a code into; no sign-in
  await expect(page.locator('input[type="tel"], input[type="email"], input[type="password"], input[autocomplete="one-time-code"]')).toHaveCount(0);
  await expect(page.getByTestId("nx-alerts").locator("input")).toHaveCount(0);
  await expect(page.getByTestId("nx-alerts")).not.toContainText(/sign in|log in|verify your/i);
  // the browser was never asked for permission to notify
  expect(await page.evaluate(() => (typeof Notification === "undefined" ? "default" : Notification.permission))).not.toBe("granted");
  expect(outside).toEqual([]);
});

test("a browser that keeps nothing: the page says so and still works for this visit", async ({ context, page }) => {
  await context.addInitScript(() => {
    const refuse = () => {
      throw new DOMException("blocked", "SecurityError");
    };
    Object.defineProperty(window, "localStorage", { get: refuse });
    Object.defineProperty(window, "sessionStorage", { get: refuse });
  });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${OK}/alerts?place=${BODEGA}`);
  await expect(page.getByTestId("alerts-not-kept")).toContainText("This browser is not keeping data");
  await page.getByTestId("alerts-follow").click();
  await expect(page.getByTestId(`alerts-port-${BODEGA}`)).toBeVisible();
  await expect(page.getByTestId("alerts-said")).toContainText("lasts only until you close the page", { timeout: 15_000 });
  await page.goto(`${OK}/today?place=${BODEGA}`);
  await expect(page.getByTestId("nx-today")).toBeVisible();
  expect(errors).toEqual([]);
});

test("offline: the saved Alerts page lists the alerts already on the device and checks nothing", async ({ page, context }) => {
  test.setTimeout(150_000);
  const records = await secondState(page, MOSS, dungenessOpens);
  await page.goto(`${OK}/alerts?place=${MOSS}`);
  await page.getByTestId("alerts-follow").click();
  await expect(page.getByTestId("alerts-said")).toContainText("noted as it is now", { timeout: 15_000 });
  records.publish();
  await page.getByTestId("alerts-check-now").click();
  await expect(page.getByTestId("alerts-item")).toHaveCount(1);
  const lastChecked = await page.getByTestId("alerts-last-checked").innerText();

  // save Moss Landing for offline use (Menu, Offline)
  await page.getByTestId("nx-menu-button").click();
  await page.getByTestId("nx-pack-save").click();
  await expect(page.getByTestId("nx-pack-status")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("nx-pack-contents")).toContainText("Alerts");
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);

  await context.setOffline(true);
  const asked: string[] = [];
  page.on("request", (r) => r.url().includes("/today/data") && asked.push(r.url()));
  await page.goto(`${OK}/alerts`);
  await expect(page.getByTestId("nx-alerts")).toBeVisible();
  await expect(page.getByTestId("alerts-item")).toHaveCount(1);
  await page.getByTestId("alerts-open").click();
  await expect(page.getByTestId("alerts-item").first()).toContainText("opens Jan 1, 2027");
  await expect(page.getByTestId("alerts-last-checked")).toHaveText(lastChecked);
  await expect(page.getByTestId("alerts-said")).toContainText("You're offline: nothing can be checked.");
  await expect(page.getByTestId("alerts-check-now")).toBeDisabled();
  await expect(page.getByTestId("nx-offline-notice")).toContainText("A new check of the ports you follow waits for a connection.");
  expect(asked).toEqual([]);
  await context.setOffline(false);
});

for (const [name, viewport] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }], ["small phone", { width: 320, height: 640 }]] as const) {
  test.describe(name, () => {
    test.use({ viewport, ...(viewport.width < 500 ? { isMobile: true, hasTouch: true } : {}) });
    test("with a port followed and an alert listed: no axe violations, no sideways scroll, 44 px controls", async ({ page }) => {
      const records = await secondState(page, MOSS, dungenessOpens);
      await page.goto(`${OK}/alerts?place=${MOSS}`);
      await page.getByTestId("alerts-follow").click();
      await expect(page.getByTestId("alerts-said")).toContainText("noted as it is now", { timeout: 15_000 });
      records.publish();
      await page.getByTestId("alerts-check-now").click();
      await expect(page.getByTestId("alerts-item")).toHaveCount(1);
      // the settings and the list's actions open from the dock's "What alerts you"
      await page.getByTestId("alerts-kinds").click();
      expect(await axe(page)).toEqual([]);
      await noSidewaysScroll(page);
      for (const id of [`alerts-trigger-${MOSS}-status_change`, "alerts-check-now", "alerts-mark-read", "alerts-add-port", `alerts-unfollow-${MOSS}`, "alerts-open", "alerts-kinds"]) {
        const b = await page.getByTestId(id).boundingBox();
        expect(b!.height, id).toBeGreaterThanOrEqual(44);
      }
    });
  });
}
