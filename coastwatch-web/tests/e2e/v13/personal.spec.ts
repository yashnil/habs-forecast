import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { OK } from "../ports";
import { BODEGA, MOSS, RETURNING, axe, noSidewaysScroll, portRecord, seed, stored } from "./helpers";

/**
 * What Neritic remembers about a reader, all of it in their browser (src/nx/personal, src/nx/account): the
 * home port and the port last opened, the "since your last visit" line, preferences, a copy of the data, and
 * deleting all of it. No account, no sign-in, nothing sent.
 */
const MONTEREY = "port-550";

test.describe("the home port", () => {
  test("a link to another port is only the port last opened: `/` still opens home", async ({ context, page }) => {
    await seed(context, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS }) });
    await page.goto(`${OK}/today?place=${MONTEREY}`);
    await expect(page.getByTestId("nx-port-control")).toContainText("Monterey");
    await expect.poll(async () => (await stored(page, "nx-port-v1")).lastPort).toBe(MONTEREY);
    expect((await stored(page, "nx-port-v1")).place).toBe(MOSS);
    await page.goto(`${OK}/`);
    await expect(page.locator("html")).toHaveAttribute("data-nx-port", MOSS);
    await expect(page.getByTestId("nx-port-control")).toContainText("Moss Landing");
  });

  test("the first port a reader chooses becomes home; a port chosen later in the bar does not replace it", async ({ page }) => {
    await page.addInitScript(() => localStorage.getItem("nx-opening-seen-v1") || localStorage.setItem("nx-opening-seen-v1", "1"));
    await page.goto(`${OK}/`);
    await page.getByTestId("nx-find-port").click();
    await page.keyboard.type("Bodega");
    await page.keyboard.press("Enter");
    await expect.poll(async () => (await stored(page, "nx-port-v1"))?.place).toBe(BODEGA);
    await page.locator('[data-testid="nx-fishery-chooser"] button').first().click();
    await expect(page.getByTestId("nx-today")).toContainText("Bodega Bay");
    // a port opened for the first time never says "since"
    await expect(page.getByTestId("nx-change")).toHaveCount(0);

    await page.getByTestId("nx-port-control").click();
    await page.keyboard.type("Moss Landing");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(new RegExp(`place=${MOSS}`));
    await expect.poll(async () => (await stored(page, "nx-port-v1")).lastPort).toBe(MOSS);
    expect((await stored(page, "nx-port-v1")).place).toBe(BODEGA);
    // and that first look at Moss Landing says nothing about an earlier visit either
    await expect(page.getByTestId("nx-today")).toContainText("Moss Landing");
    await expect(page.getByTestId("nx-change")).toHaveCount(0);
  });
});

test.describe("Your data and preferences", () => {
  test.beforeEach(async ({ context }) => {
    await seed(context, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS, last: BODEGA, fisheries: ["dungeness_commercial", "rock_crab"] }) });
  });

  test("in the v13 shell, with no sign-in and no way to give a number, an address or an age", async ({ page }) => {
    await page.goto(`${OK}/account`);
    await expect(page.getByTestId("nx-app")).toBeVisible();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Your data and preferences");
    await expect(page.getByTestId("account-local")).toHaveText("Everything here stays in this browser. There is no account to sign in to in this preview.");
    await expect(page.locator('input[type="tel"], input[type="email"], input[type="password"], input[type="checkbox"]')).toHaveCount(0);
    await expect(page.getByTestId("nx-account")).not.toContainText(/13 or older|text me|mobile number|trip log|trip feedback/i);
    await expect(page.getByTestId("account-pilot-link")).toHaveCount(0);
    // the Menu's "Your data" is this page
    await page.getByTestId("nx-menu-button").click();
    await expect(page.getByRole("navigation", { name: "More" }).getByRole("link", { name: "Your data" })).toHaveAttribute("href", "/account");
  });

  test("the old alerts address is sent on to Alerts with its port", async ({ page }) => {
    await page.goto(`${OK}/account?place=${BODEGA}#alerts`);
    await expect(page).toHaveURL(new RegExp(`/alerts\\?place=${BODEGA}$`));
    await expect(page.getByTestId("alerts-offer")).toHaveAttribute("data-port", BODEGA);
  });

  test("home port: set, moved and cleared here, and `/` follows it", async ({ page }) => {
    await page.goto(`${OK}/account`);
    const home = page.getByTestId("account-home");
    await expect(home).toHaveValue(MOSS);
    await expect(page.getByTestId("account-opens")).toContainText("Neritic opens on Moss Landing. Opening a link to another port does not change this.");
    await home.selectOption(MONTEREY);
    await expect(page.getByTestId("account-opens")).toContainText("Neritic opens on Monterey.");
    expect((await stored(page, "nx-port-v1")).place).toBe(MONTEREY);
    await page.goto(`${OK}/`);
    await expect(page.locator("html")).toHaveAttribute("data-nx-port", MONTEREY);

    await page.goto(`${OK}/account`);
    await page.getByTestId("account-home").selectOption("");
    await expect(page.getByTestId("account-opens")).toContainText("With no home port, Neritic opens on the port you used last (Bodega Bay).");
    await page.goto(`${OK}/`);
    await expect(page.locator("html")).toHaveAttribute("data-nx-port", BODEGA);
  });

  test("fisheries in the reader's order, the chart mode, and the tour offered again", async ({ page }) => {
    await page.goto(`${OK}/account`);
    const list = page.getByTestId("account-fisheries");
    await expect(list.locator("li")).toHaveText([/Dungeness crab \(commercial\)/, /Rock crab/]);
    await page.getByRole("button", { name: "Move Rock crab up" }).click();
    await expect(list.locator("li")).toHaveText([/Rock crab/, /Dungeness crab \(commercial\)/]);
    await page.getByTestId("account-add-fishery").selectOption("market_squid");
    await page.getByRole("button", { name: "Remove Dungeness crab (commercial)" }).click();
    expect((await stored(page, "nx-port-v1")).fisheries).toEqual(["rock_crab", "market_squid"]);
    // Today leads with the first
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-today")).toContainText(/Moss Landing · Rock crab/i);

    await page.goto(`${OK}/account`);
    await page.getByRole("radio", { name: /Night/ }).click();
    await expect(page.locator("html")).toHaveAttribute("data-chart-mode", "night");
    expect(await page.evaluate(() => localStorage.getItem("nx-chart-mode-v1"))).toBe("night");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-chart-mode", "night");
    await expect(page.getByRole("radio", { name: /Night/ })).toBeChecked();

    await page.getByTestId("account-replay-tour").click();
    await expect(page.getByTestId("account-said")).toContainText("offered again on Today");
    expect(await page.evaluate(() => localStorage.getItem("nx-tour-v1"))).toBeNull();
  });

  test("what is kept is listed in plain words, and a copy can be saved", async ({ page }) => {
    await page.goto(`${OK}/today?place=${MOSS}`);
    await expect(page.getByTestId("nx-today")).toContainText("Moss Landing");
    // Today records what it showed once the port's records have loaded; leave only after it has
    await expect
      .poll(() => page.evaluate(() => Object.keys(localStorage).some((k) => k.startsWith("nx-today-seen-v1:"))))
      .toBe(true);
    await page.goto(`${OK}/account`);
    const data = page.getByTestId("account-data");
    await expect(data).toContainText("Your home port, the port you opened last, your fisheries in order and your vessel length");
    await expect(data).toContainText("What Today showed at each port on your last visit");
    // in the reader's words: no storage key names, no file formats
    await expect(data).not.toContainText(/nx-|localStorage|JSON|cache/i);

    const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("account-export").click()]);
    expect(download.suggestedFilename()).toMatch(/^neritic-my-data-\d{4}-\d{2}-\d{2}\.json$/);
    const copy = JSON.parse(readFileSync((await download.path())!, "utf8"));
    expect(copy).toMatchObject({ app: "neritic-device", schema: 1 });
    const port = copy.local.find((l: { key: string }) => l.key === "nx-port-v1");
    expect(port.value).toMatchObject({ place: MOSS, fisheries: ["dungeness_commercial", "rock_crab"] });
    expect(port.holds).toContain("Your home port");
    await expect(page.getByTestId("account-data-said")).toContainText("A copy was saved as neritic-my-data-");
  });
});

test("Delete everything on this device: ports, follows, alerts, saved trips, visit history and offline copies all go", async ({ page }) => {
  test.setTimeout(90_000);
  await page.goto(`${OK}/about`);
  // a used device, as the journeys leave it (set once: the delete must not be undone by a seed on reload)
  await page.evaluate(
    async ([rec]) => {
      const set = (k: string, v: unknown) => localStorage.setItem(k, typeof v === "string" ? v : JSON.stringify(v));
      set("nx-port-v1", rec);
      set("nx-opening-seen-v1", "1");
      set("nx-tour-v1", "dismissed");
      set("nx-chart-mode-v1", "dusk");
      set("nx-follow-v1", { v: 1, ports: [{ port: "port-473", triggers: ["status_change"], since: "2026-10-01T00:00:00.000Z" }] });
      set("nx-inbox-v1", [{ key: "k", title: "Bodega Bay: a change", place_id: "port-473", place_label: "Bodega Bay", trigger: "status_change", body: "", created_at: "2026-10-12T00:00:00.000Z", read_at: null }]);
      set("nx-alert-snap-v1", { "port-473": { snap: { place_id: "port-473", seen_at: "2026-10-12T00:00:00.000Z", facts: {} }, data_at: null } });
      set("nx-trip-v1:scenarios", [{ id: "a", name: "Crab, Moss Landing" }]);
      set("nx-trip-v1:vessels", [{ id: "v", name: "F/V Anna" }]);
      set("nx-coast-v1", { fishery: "dungeness_commercial" });
      set("nx-today-seen-v1:port-592", { at: "2026-10-12T00:00:00.000Z", facts: [] });
      set("nx-ask-seen-v1:port-592", {});
      set("nx-field-v1", { v: 1 });
      set("nrt-pilot-account-v1", { v: 1, saved_places: ["port-592"], sms: { phone_e164: "+18315550100" } });
      set("some-key-nobody-registered", "x");
      sessionStorage.setItem("nx-today-visit-v1:port-592", "{}");
      for (const name of ["neritic-pack-field", "neritic-pack-today"]) await (await caches.open(name)).put("/saved", new Response("x"));
    },
    [portRecord({ home: MOSS, last: BODEGA })],
  );

  await page.goto(`${OK}/account`);
  await expect(page.getByTestId("account-data")).toContainText("Your saved trips, vessels and their costs and prices");
  await page.getByTestId("account-delete").click();
  const confirm = page.getByTestId("account-delete-confirm");
  await expect(confirm).toContainText("the 1 port you follow and your 1 alert");
  await expect(confirm).toContainText("every trip and vessel saved on My Coast");
  await expect(confirm).toContainText("It cannot be undone.");
  await page.getByTestId("account-delete-yes").click();

  await expect(page.getByTestId("account-deleted")).toHaveText("Everything Neritic kept in this browser has been deleted.");
  await expect(page.getByTestId("account-nothing")).toBeVisible();
  // nothing of the reader's is left: every key, the tab's session, both offline packs
  const left = await page.evaluate(async () => ({ local: Object.keys(localStorage), session: Object.keys(sessionStorage), caches: (await caches.keys()).filter((k) => !k.startsWith("neritic-static-")) }));
  expect(left.local.filter((k) => k !== "nx-today-probe")).toEqual([]);
  expect(left.session).toEqual([]);
  expect(left.caches).toEqual([]);
  await expect(page.getByTestId("nx-port-control")).toContainText("Find your port");
  await expect(page.getByTestId("nx-unread")).toHaveCount(0);

  // Neritic opens as it does on a first visit
  await page.goto(`${OK}/`);
  await expect(page.locator("html")).toHaveAttribute("data-nx-entry", "home");
  await page.goto(`${OK}/alerts`);
  await expect(page.getByTestId("alerts-no-follows")).toBeVisible();
  await expect(page.getByTestId("alerts-empty")).toBeVisible();
});

for (const [name, viewport] of [["desktop", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }], ["small phone", { width: 320, height: 640 }]] as const) {
  test.describe(name, () => {
    test.use({ viewport, ...(viewport.width < 500 ? { isMobile: true, hasTouch: true } : {}) });
    test("Your data: no axe violations, no sideways scroll, 44 px controls", async ({ context, page }) => {
      await seed(context, { ...RETURNING, "nx-port-v1": portRecord({ home: MOSS, fisheries: ["dungeness_commercial", "rock_crab"] }) });
      await page.goto(`${OK}/account`);
      await expect(page.getByTestId("account-data")).toHaveAttribute("data-count", /\d+/);
      expect(await axe(page)).toEqual([]);
      await noSidewaysScroll(page);
      for (const id of ["account-home", "account-add-fishery", "account-export", "account-delete", "account-replay-tour"]) expect((await page.getByTestId(id).boundingBox())!.height, id).toBeGreaterThanOrEqual(44);
      expect((await page.getByRole("button", { name: "Move Rock crab up" }).boundingBox())!.height).toBeGreaterThanOrEqual(44);
    });
  });
}
