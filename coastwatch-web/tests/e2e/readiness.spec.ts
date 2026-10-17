import { expect, test } from "@playwright/test";
import { NODATA, OK } from "./ports";

/**
 * Readiness (v4): the app always says what is ready, what is loading, whether it is fresh or saved,
 * whether the device is online, and what can be done next. No bare spinner, no invented progress.
 */
const NOW = "2026-10-08T20:00:00Z";

test("the shell renders at once with a start card listing what loads, then the status centre says Ready or why not", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  // hold the map's code (the lazily loaded MapCanvas chunk, recognised by its selection ring image)
  // until the start state has been checked; the style is inline, so nothing later holds the map back
  let release!: () => void;
  const held = new Promise<void>((res) => (release = res));
  // (on the context: the service worker fetches app code, and only a context route sees that)
  await page.context().route("**/_next/static/chunks/*.js", async (r) => {
    const res = await r.fetch();
    const body = await res.text();
    if (body.includes("v4-ring")) await held;
    await r.fulfill({ response: res, body });
  });
  await page.goto(`${OK}/?layer=forecast`);
  const card = page.getByTestId("start-card");
  await expect(card).toBeVisible();
  await expect(card).toContainText("Search a harbour");
  await expect(card.getByTestId("source-map")).toHaveAttribute("data-state", "loading");
  release();
  await expect(page.getByTestId("start-card")).toHaveCount(0, { timeout: 20000 });
  const status = page.getByTestId("status-center");
  await expect(status).not.toHaveAttribute("data-state", "initializing");
  await status.click();
  const panel = page.getByTestId("status-panel");
  await expect(panel.getByTestId("source-map")).toHaveAttribute("data-state", /ready|partial/);
  await expect(panel.getByTestId("source-notices")).toContainText("not verified");
  await expect(panel).toContainText("Online");
});

test("a provider that fails says so for that part only, offers Try again, and recovers", async ({ page }) => {
  await page.clock.setFixedTime(new Date(NOW));
  let fail = true;
  await page.route("**/api/evidence**", (r) => (fail ? r.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: "Neritic data are unavailable" }) }) : r.continue()));
  await page.goto(`${OK}/?layer=forecast&sel=harbor:550`);
  const claims = page.getByTestId("harbor-claims");
  await expect(claims.getByRole("alert")).toContainText("could not be loaded");
  await expect(claims.getByRole("alert")).toContainText("Everything else on this page is unaffected");
  // the rest of the panel still works
  await expect(page.getByTestId("harbor-stations")).toContainText("Monterey Wharf");
  fail = false;
  await claims.getByRole("button", { name: "Try again" }).click();
  await expect(claims.locator("[data-kind]").first()).toBeVisible();
});

test("no dataset: an explicit unavailable state with the agencies and a retry, never an empty map", async ({ page }) => {
  await page.goto(`${NODATA}/`);
  const u = page.getByTestId("data-unavailable");
  await expect(u).toContainText("could not be loaded");
  await expect(u.getByRole("link", { name: "CDFW advisories" })).toBeVisible();
  await expect(u.getByRole("link", { name: "Try again" })).toBeVisible();
});

test("offline is said persistently, with what is shown and when; never as a toast", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.clock.setFixedTime(new Date(NOW));
  await page.goto(`${OK}/coast?place=port-550`);
  await expect(page.getByTestId("matters").locator("li").first()).toBeVisible();
  await ctx.setOffline(true);
  await expect(page.getByTestId("offline-banner")).toContainText("Offline");
  await expect(page.getByTestId("status-center")).toHaveAttribute("data-state", "offline");
  await ctx.setOffline(false);
  await expect(page.getByTestId("offline-banner")).toHaveCount(0);
  await ctx.close();
});
